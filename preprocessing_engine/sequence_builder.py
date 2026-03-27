"""Sliding-window sequence creation."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def build_sequences(df: pd.DataFrame, feature_cols: Iterable[str], target_col: str, window_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    values = df[list(feature_cols)].values
    targets = df[target_col].values
    for i in range(len(df) - window_size):
        X.append(values[i : i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)
