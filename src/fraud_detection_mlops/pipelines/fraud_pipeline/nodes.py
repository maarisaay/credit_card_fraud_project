"""
This is a boilerplate pipeline 'fraud_pipeline'
generated using Kedro 1.4.0
"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


def preprocess_data(raw_data: pd.DataFrame, test_size: float, random_state: int):
    """Skalowanie Amount/Time i podział train/test ze stratyfikacją."""
    df = raw_data.copy()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    """Trening baseline Logistic Regression."""
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state
    )
    model.fit(X_train, y_train)

    # mlflow logging
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", random_state)

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

    # mlflow logging
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_fraud", report["Fraud"]["recall"])
    mlflow.log_metric("precision_fraud", report["Fraud"]["precision"])
    mlflow.sklearn.log_model(model, "model")

    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return metrics