import numpy as np
import torch

def calculate_anomaly_threshold(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, n_std: float = 3.0) -> float:
    """Расчет MSE loss порога аномальности по формуле : Mean + n_std * Std"""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in val_loader:
            x_batch = batch[0]
            reconstructed = model(x_batch)
            mse = torch.mean((x_batch - reconstructed) ** 2, dim=1)
            losses.extend(mse.numpy())

    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    threshold = mean_loss + (n_std * std_loss)

    print(f" Val Mean MSE: {mean_loss:.6f} | Std: {std_loss:.6f}")
    print(f" Вычислительный порог (Threshold): {threshold:.6f}")

    return threshold