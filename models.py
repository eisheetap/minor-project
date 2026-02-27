"""Models: Random Forest baseline and LSTM sequence regressor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, TensorDataset

GLOBAL_SEED = 123


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = None,
    seed: int = GLOBAL_SEED,
) -> RandomForestRegressor:
    """Fit a RandomForest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        out = output[:, -1, :]
        return self.fc(out)


@dataclass
class TrainHistory:
    losses: List[float]
    val_losses: List[float]


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


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = GLOBAL_SEED,
    device: str | None = None,
) -> Tuple[LSTMRegressor, TrainHistory]:
    """Train an LSTM regressor with early-stable defaults."""
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader, val_loader = _build_loaders(X_train, y_train, X_val, y_val, batch_size)

    losses: List[float] = []
    val_losses: List[float] = []
    for epoch in range(epochs):
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
        losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                val_loss += loss.item() * len(xb)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
    return model, TrainHistory(losses=losses, val_losses=val_losses)


def predict_lstm(model: LSTMRegressor, X: np.ndarray, device: str | None = None) -> np.ndarray:
    """Predict using a trained LSTM."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor_x = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(tensor_x).squeeze().cpu().numpy()
    return preds
