import mlflow
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from fraud_detection_mlops.feature_utils import add_engineered_features


def preprocess_data(raw_data: pd.DataFrame, test_size: float, random_state: int):
    """Podział train/test ze stratyfikacją (bez skalowania — to po FE)."""
    df = raw_data.copy()
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def engineer_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Inżynieria cech na surowych wartościach Amount/Time (przed skalowaniem)."""
    return add_engineered_features(X_train), add_engineered_features(X_test)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """StandardScaler na Amount i Time — fit na train, transform na test (bez data leakage)."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    return X_train, X_test, scaler


def select_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, selected_features: list
):
    """Filtruje do cech wybranych na podstawie feature_importance z AutoGluon (analiza w notebooku 02)."""
    return X_train[selected_features], X_test[selected_features]


def tune_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int, random_state: int
) -> dict:
    """Bayesian Optimization (Optuna) hiperparametru C dla LogisticRegression."""

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        model = LogisticRegression(
            C=C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
        return cross_val_score(
            model, X_train, y_train, scoring="average_precision", cv=3, n_jobs=-1
        ).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    mlflow.log_param("tuned_C", study.best_params["C"])
    mlflow.log_metric("cv_pr_auc", study.best_value)

    print(
        f"Najlepsze C: {study.best_params['C']:.4f} (CV PR-AUC: {study.best_value:.4f})"
    )

    return {"C": study.best_params["C"], "cv_pr_auc": study.best_value}


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: dict, random_state: int
):
    """Trening finalnego modelu z wytuningowanym C."""
    model = LogisticRegression(
        C=best_params["C"],
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", best_params["C"])

    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Ewaluacja: PR-AUC, ROC-AUC, classification report, confusion matrix."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, y_pred, target_names=["Normalna", "Fraud"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_fraud", report["Fraud"]["recall"])
    mlflow.log_metric("precision_fraud", report["Fraud"]["precision"])
    mlflow.sklearn.log_model(model, "model")

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return metrics


def prepare_serving_artifacts(raw_data: pd.DataFrame, selected_features: list) -> dict:
    """Metadane potrzebne do serwowania: lista cech + próbka referencyjna do drift detection."""
    reference_amounts = raw_data["Amount"].sample(2000, random_state=42).tolist()
    return {
        "selected_features": selected_features,
        "reference_amounts": reference_amounts,
    }
