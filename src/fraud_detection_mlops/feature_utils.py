import numpy as np
import pandas as pd

INTERACTION_COLS = ["V17", "V14", "V12", "V10"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cechy czasowe, log-transform Amount, interakcje z najbardziej skorelowanymi V."""
    X = X.copy()
    X["hour_of_day"] = (X["Time"] // 3600) % 24
    X["amount_log"] = np.log1p(X["Amount"])
    for col in INTERACTION_COLS:
        X[f"amount_x_{col}"] = X["Amount"] * X[col]
    return X