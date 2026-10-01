import torch
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import TransactionAutoencoder

app = FastAPI(title="Fraud & Anomaly Detector Service")

# Загружаем сохраненный scaler и чекпоинт модели при старте
scaler = joblib.load("scaler.pkl")
checkpoint = torch.load("autoencoder_fraud.pth", map_location=torch.device('cpu'))

input_dim = checkpoint["input_dim"]
threshold = checkpoint["threshold"]

model = TransactionAutoencoder(input_dim=input_dim, latent_dim=2)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class TransactionFeatures(BaseModel):
    amount: float
    hour: int
    day_of_week: int
    time_delta_minutes: float
    user_avg_spend: float

class FraudPredictionResponse(BaseModel):
    is_anomaly: bool
    risk_score: float
    reconstruction_error: float
    threshold: float

@app.post("/predict", response_model=FraudPredictionResponse)
def predict_anomaly(features: TransactionFeatures):
    try:
        # Преобразуем входящие данные в вектор
        raw_vector = np.array([[
            features.amount,
            features.hour,
            features.day_of_week,
            features.time_delta_minutes,
            features.user_avg_spend
        ]])

        # Нормализация
        scaled_vector = scaler.transform(raw_vector)
        x_tensor = torch.tensor(scaled_vector, dtype=torch.float32)

        # Инференс PyTorch
        with torch.no_grad():
            reconstructured = model(x_tensor)
            # MSE loss между входом и восстановлением
            mse_loss = float(torch.mean((x_tensor - reconstructured) ** 2).item())

        # Расчет Risk Score (нормализованное значение относительно порога)
        risk_score = min(1.0, mse_loss / (threshold * 2))
        is_anomaly = mse_loss > threshold

        return FraudPredictionResponse(
            is_anomaly=is_anomaly,
            risk_score=round(risk_score, 4),
            reconstruction_error=round(mse_loss, 6),
            threshold=round(threshold, 6)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")