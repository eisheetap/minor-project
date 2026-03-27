"""Robustness checks via noise and missingness."""
from __future__ import annotations

from typing import Dict

import numpy as np

from evaluation_engine.metrics import regression_metrics
from training_engine.trainer import predict_model


def inject_noise(X: np.ndarray, std: float) -> np.ndarray:
    noisy = X.copy()
    noise = np.random.normal(0, std, size=noisy.shape)
    return noisy + noise


def simulate_missing(X: np.ndarray, missing_frac: float) -> np.ndarray:
    masked = X.copy()
    total = masked.size
    n_missing = int(total * missing_frac)
    if n_missing == 0:
        return masked
    flat_indices = np.random.choice(total, size=n_missing, replace=False)
    masked = masked.reshape(-1)
    masked[flat_indices] = np.nan
    masked = masked.reshape(X.shape)
    # Simple mean imputation
    col_means = np.nanmean(masked, axis=(0, 1))
    inds = np.where(np.isnan(masked))
    masked[inds] = np.take(col_means, inds[2] if len(inds) > 2 else inds[1])
    return masked


def evaluate_robustness(
    model_kind: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_std: float,
    missing_frac: float,
) -> Dict[str, Dict[str, float]]:
    noisy = inject_noise(X_test, noise_std)
    noisy_preds = predict_model(model_kind, model, noisy)
    noisy_metrics = regression_metrics(y_test, noisy_preds)

    missing = simulate_missing(X_test, missing_frac)
    missing_preds = predict_model(model_kind, model, missing)
    missing_metrics = regression_metrics(y_test, missing_preds)

    return {"noisy": noisy_metrics, "missing": missing_metrics}
