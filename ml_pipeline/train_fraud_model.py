import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data_preprocessing import prepare_data_loaders
from evaluate import calculate_anomaly_threshold

import mlflow
import mlflow.pytorch

from data_preprocessing import prepare_data_loaders
from evaluate import calculate_anomaly_threshold
from services.fraud_detector.model import TransactionAutoencoder


def run_pipeline(epochs: int = 20, lr: float = 1e-3, latent_dim: int = 2):
    print("Старт ML-пайплайна обучения Fraud Detector...")

    # 0. Настройка подключения к MLFlow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Fraud_Detector_Autoencoder")

    with mlflow.start_run():
        # Логируем гиперпараметры
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": lr,
            "latent_dim": latent_dim,
            "optimizer": "Adam",
            "loss_function": "MSELoss"
        })

    # 1. Загрузка данных
    train_loader, val_loader, input_dim = prepare_data_loaders()

    # 2. Инициализация модели и оптимизатора
    model = TransactionAutoencoder(input_dim=input_dim, latent_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # Логируем метрику каждой эпохи
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # 4. Вычисление порога на валидации
    threshold = calculate_anomaly_threshold(model, val_loader)
    mlflow.log_metric("anomaly_threshold", threshold)

    # 5. Сохранение чекпоинта
    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "threshold": threshold,
    }
    torch.save(checkpoint, "autoencoder_fraud.pth")
    print("Пайплайн завершен! Веса и порог сохранены в autoencoder_fraud.pth")

if __name__ == "__main__":
    run_pipeline()