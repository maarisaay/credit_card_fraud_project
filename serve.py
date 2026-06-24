import json
import pickle
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import create_model
from scipy.stats import ks_2samp

from fraud_detection_mlops.feature_utils import add_engineered_features

DATA_DIR = Path("data")

with open(DATA_DIR / "06_models" / "fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(DATA_DIR / "06_models" / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(DATA_DIR / "07_model_output" / "serving_metadata.json") as f:
    metadata = json.load(f)

selected_features = metadata["selected_features"]
reference_amounts = metadata["reference_amounts"]

# dynamiczny model Pydantic - Time, V1-V28, Amount (30 surowych cech wejściowych)
fields = {f"V{i}": (float, ...) for i in range(1, 29)}
fields["Time"] = (float, ...)
fields["Amount"] = (float, ...)
Transaction = create_model("Transaction", **fields)

app = FastAPI(title="Fraud Detection API")

predictions_total = Counter(
    "fraud_predictions_total", "Liczba predykcji", ["predicted_class"]
)
prediction_latency = Histogram("fraud_prediction_latency_seconds", "Czas predykcji")
fraud_probability = Histogram(
    "fraud_probability",
    "Rozkład prawdopodobieństwa fraud",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
amount_drift_score = Gauge(
    "fraud_amount_drift_ks_statistic",
    "KS statistic: rozkład Amount (ostatnie predykcje) vs trening",
)

recent_amounts: list[float] = []
MAX_BUFFER = 500


def preprocess(transaction) -> pd.DataFrame:
    raw = pd.DataFrame([transaction.model_dump()])
    fe = add_engineered_features(raw)
    fe[["Amount", "Time"]] = scaler.transform(fe[["Amount", "Time"]])
    return fe[selected_features]


@app.post("/predict")
def predict(transaction: Transaction):
    start = time.time()

    X = preprocess(transaction)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    latency = time.time() - start
    predictions_total.labels(predicted_class=str(pred)).inc()
    prediction_latency.observe(latency)
    fraud_probability.observe(proba)

    log_entry = {
        "timestamp": time.time(),
        "amount": transaction.Amount,
        "predicted_class": pred,
        "probability": proba,
    }
    with open("predictions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    recent_amounts.append(transaction.Amount)
    if len(recent_amounts) > MAX_BUFFER:
        recent_amounts.pop(0)
    if len(recent_amounts) >= 50:
        stat, _ = ks_2samp(recent_amounts, reference_amounts)
        amount_drift_score.set(stat)

    return {"fraud_probability": proba, "predicted_class": pred}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
