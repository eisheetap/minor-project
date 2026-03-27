"""Generic training helpers for baselines and LSTM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation_engine.metrics import regression_metrics
from modeling_engine import baseline_models
from modeling_engine.lstm_model import LSTMRegressor
from training_engine.reproducibility import set_global_seed


@dataclass
class TrainingHistory:
    losses: list
    val_losses: list


@dataclass
class TrainingResult:
    model: Any
    history: TrainingHistory
    val_metrics: Dict[str, float]
    val_predictions: np.ndarray


def _build_loaders(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, batch_size: int):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def predict_model(kind: str, model: Any, X: np.ndarray, device: str | None = None) -> np.ndarray:
    if kind == "rf":
        return baseline_models.predict_random_forest(model, X)
    if kind == "linear":
        return baseline_models.predict_linear_regression(model, X)
    if kind == "lstm":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32, device=device)
            preds = model(tensor_x).squeeze().cpu().numpy()
        return preds
    raise ValueError(f"Unsupported model kind for prediction: {kind}")


def train_model(
    kind: str,
    model: Any,
    train_cfg: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> TrainingResult:
    set_global_seed(seed)
    if kind == "rf":
        rf_model = baseline_models.train_random_forest(X_train, y_train, baseline_models.RFConfig(**train_cfg))
        val_pred = predict_model("rf", rf_model, X_val)
        return TrainingResult(
            model=rf_model,
            history=TrainingHistory(losses=[], val_losses=[]),
            val_metrics=regression_metrics(y_val, val_pred),
            val_predictions=val_pred,
        )

    if kind == "linear":
        lin_model = baseline_models.train_linear_regression(X_train, y_train, baseline_models.LinearConfig(**train_cfg))
        val_pred = predict_model("linear", lin_model, X_val)
        return TrainingResult(
            model=lin_model,
            history=TrainingHistory(losses=[], val_losses=[]),
            val_metrics=regression_metrics(y_val, val_pred),
            val_predictions=val_pred,
        )

    if kind == "lstm":
        device = train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        lr = train_cfg.get("lr", 1e-3)
        epochs = train_cfg.get("epochs", 25)
        batch_size = train_cfg.get("batch_size", 64)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_loader, val_loader = _build_loaders(X_train, y_train, X_val, y_val, batch_size)
        losses, val_losses = [], []

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
            losses.append(epoch_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).squeeze()
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * len(xb)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

        val_pred = predict_model("lstm", model, X_val, device=device)
        return TrainingResult(
            model=model,
            history=TrainingHistory(losses=losses, val_losses=val_losses),
            val_metrics=regression_metrics(y_val, val_pred),
            val_predictions=val_pred,
        )

    raise ValueError(f"Unsupported model kind: {kind}")
