import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from model import TransactionAutoencoder

def generate_synthetic_transactions(n_samples: int = 10000):
    """
    Генерация синтетических признаков транзакцийЖ
    0: Сумма транзакций
    1: Час суток (0-23)
    2: День-недели (0-6)
    3: Время с предыдущей транзакции (в мин)
    4: Средний чек пользователя
    """
    np.random.seed(42)
    amount = np.random.exponential(scale=2000, size=n_samples)
    hour = np.random.randint(0, 24, size=n_samples)
    day_of_week = np.random.randint(0, 7, size=n_samples)
    time_delta = np.random.exponential(scale=120, size=n_samples)
    avg_user_spend = np.random.normal(loc=2500, scale=500, size=n_samples)

    X = np.column_stack([amount, hour, day_of_week, time_delta, avg_user_spend])
    return X

def train_pipeline():
    # 1. Загрузка и подготовка нормальных данных
    X_train_raw = generate_synthetic_transactions(n_samples=10000)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # Сохраняем Scaler для фазы Inference
    joblib.dump(scaler, "scaler.pkl")

    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Инициализация модели и оптимизатора
    input_dim = X_train_raw.shape[1]
    model = TransactionAutoencoder(input_dim=input_dim, latent_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 3. Цикл обучения (Training Loop)
    model.train()
    epochs = 20
    print("Старт обучения PyTorch Autoencoder...")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructured = model(x_batch)
            loss = criterion(reconstructured, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss}/{len(dataloader):.6f}")

    # 4. Расчет порога аномальности (Threshold = Mean Loss + 3 * Std)
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        errors = torch.mean((X_tensor - preds) ** 2, dim=1).numpy()
        threshold = float(np.mean(errors) + 3 * np.std(errors))
        print(f"Порог аномальности установлен: {threshold:.6f}")

    # 5. Сохранение весов и порога
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "threshold": threshold
    }, "autoencoder_fraud.pth")
    print("Модель и метаданные сохранены в autoencoder_fraud.pth и scaler.pkl")

if __name__ == "__main__":
    train_pipeline()