FROM python:3.12-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY data/06_models/fraud_model.pkl ./data/06_models/fraud_model.pkl
COPY data/06_models/scaler.pkl ./data/06_models/scaler.pkl
COPY data/07_model_output/serving_metadata.json ./data/07_model_output/serving_metadata.json
COPY src/fraud_detection_mlops/__init__.py ./fraud_detection_mlops/__init__.py
COPY src/fraud_detection_mlops/feature_utils.py ./fraud_detection_mlops/feature_utils.py
COPY serve.py .

EXPOSE 8008

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8008"]