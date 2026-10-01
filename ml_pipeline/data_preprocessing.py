import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_transactions(n_samples: int = 12000):
    """Генерация нормальных финтех-транзакций для обучения Autoencoder."""
    np.random.seed(42)
    amount = np.random.exponential(scale=1500, size=n_samples)
    hour = np.random.randint(0, 24, size=n_samples)
    day_of_week = np.random.randint(0, 7, size=n_samples)
    time_delta = np.random.exponential(scale=60, size=n_samples)
    user_avg_spend = np.random.normal(loc=2000, scale=400, size=n_samples)

    return np.column_stack([amount, hour, day_of_week, time_delta, user_avg_spend])

def prepare_data_loaders(batch_size: int = 64, scaler_path: str = "scaler.pkl"):
    """Подготовка и масштабирование данных, сохранение scaler и возврат DataLoader'ов."""
    X_raw = generate_synthetic_transactions()

    X_train, X_val = train_test_split(X_raw, text_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Сохраняем Scaler для фазы Inference
    joblib.dump(scaler, scaler_path)

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_tensor), batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader, X_train_scaled.shape[1]