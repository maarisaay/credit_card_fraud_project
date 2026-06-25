from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = Path("data/06_models/fraud_model.pkl")
SCALER_PATH = Path("data/06_models/scaler.pkl")
METADATA_PATH = Path("data/07_model_output/serving_metadata.json")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists()
    or not SCALER_PATH.exists()
    or not METADATA_PATH.exists(),
    reason="Serving artifacts are not available in CI.",
)

from serve import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint():
    response = client.get("/version")

    assert response.status_code == 200

    data = response.json()
    assert data["project"] == "fraud-detection-mlops"
    assert data["api_version"] == "1.0.0"


def test_predict_rejects_negative_amount():
    payload = {"Time": 1000, "Amount": -10}

    for i in range(1, 29):
        payload[f"V{i}"] = 0.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Amount must be non-negative."


def test_predict_rejects_negative_time():
    payload = {"Time": -1, "Amount": 10}

    for i in range(1, 29):
        payload[f"V{i}"] = 0.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Time must be non-negative."
