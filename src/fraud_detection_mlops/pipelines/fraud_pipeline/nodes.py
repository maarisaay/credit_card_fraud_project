"""
This is a boilerplate pipeline 'fraud_pipeline'
generated using Kedro 1.4.0
"""
import mlflow
import optuna
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

def select_features(X_train, X_test, selected_features: list):
    """Filtruje do cech wybranych na podstawie feature_importance z AG"""
    return X_train[selected_features], X_test[selected_features]

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
    """Cechy czasowe, log-transform Amount, interakcje z V17/V14/V12/V10 — na surowych wartościach."""

    def _add_features(X):
        X = X.copy()
        X["hour_of_day"] = (X["Time"] // 3600) % 24
        X["amount_log"] = np.log1p(X["Amount"])
        for col in ["V17", "V14", "V12", "V10"]:
            X[f"amount_x_{col}"] = X["Amount"] * X[col]
        return X

    return _add_features(X_train), _add_features(X_test)


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

    # mlflow logging
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_fraud", report["Fraud"]["recall"])
    mlflow.log_metric("precision_fraud", report["Fraud"]["precision"])
    mlflow.sklearn.log_model(model, "model")

    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return metrics


def tune_hyperparameters(X_train, y_train, n_trials: int, random_state: int) -> dict:
    """Bayesian Optimization (Optuna) hiperparametru C dla LogisticRegression."""

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        model = LogisticRegression(
            C=C, solver="liblinear", class_weight="balanced",
            max_iter=5000, random_state=random_state
        )
        return cross_val_score(
            model, X_train, y_train, scoring="average_precision", cv=3, n_jobs=-1
        ).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    mlflow.log_param("tuned_C", study.best_params["C"])
    mlflow.log_metric("cv_pr_auc", study.best_value)

    print(f"Najlepsze C: {study.best_params['C']:.4f} (CV PR-AUC: {study.best_value:.4f})")

    return {"C": study.best_params["C"], "cv_pr_auc": study.best_value}


def train_model(X_train, y_train, best_params: dict, random_state: int):
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

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """StandardScaler na Amount i Time — fit na train, transform na test (bez data leakage)."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    return X_train, X_test