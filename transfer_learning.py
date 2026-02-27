"""Transfer learning routines for fine-tuning the LSTM on Region B."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMRegressor, set_global_seed

GLOBAL_SEED = 123


def _build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def fine_tune_lstm(
    model: LSTMRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 5e-4,
    epochs: int = 10,
    batch_size: int = 64,
    freeze_backbone: bool = False,
    seed: int = GLOBAL_SEED,
    device: str | None = None,
) -> Dict[str, List[float]]:
    """
    Fine-tune a pretrained LSTM on limited target data.

    freeze_backbone=True freezes the LSTM layers and updates only the head.
    """
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "lstm" in name:
                param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    train_loader, val_loader = _build_loaders(X_train, y_train, X_val, y_val, batch_size)

    train_loss, val_loss = [], []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)
        train_loss.append(epoch_loss)

        model.eval()
        val_epoch = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                val_epoch += loss.item() * len(xb)
        val_epoch /= len(val_loader.dataset)
        val_loss.append(val_epoch)

    return {"train_loss": train_loss, "val_loss": val_loss}
