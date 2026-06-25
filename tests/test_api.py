from fastapi.testclient import TestClient

from serve import app

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