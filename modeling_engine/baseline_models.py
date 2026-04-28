"""Classical baseline regressors for cross-domain comparison experiments.

Purpose:
- Provides Random Forest and Linear Regression training/prediction helpers to benchmark
  against the LSTM transfer-learning path.

Inputs:
- Sequence arrays (optionally 3D), training targets, and model-specific configs.

Outputs:
- Trained sklearn model objects and predicted target vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def _flatten_sequences(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X.reshape(len(X), -1)
    return X


@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    seed: int = 123


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, config: RFConfig) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(_flatten_sequences(X_train), y_train)
    return model


def predict_random_forest(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(_flatten_sequences(X))


@dataclass
class LinearConfig:
    fit_intercept: bool = True
    n_jobs: int = -1


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray, config: LinearConfig) -> LinearRegression:
    model = LinearRegression(fit_intercept=config.fit_intercept, n_jobs=config.n_jobs)
    model.fit(_flatten_sequences(X_train), y_train)
    return model


def predict_linear_regression(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(_flatten_sequences(X))
