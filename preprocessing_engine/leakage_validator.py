"""Validation helpers to guard against leakage and scaling issues."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def assert_no_future_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if not (train_df["timestamp"].max() < test_df["timestamp"].min()):
        raise ValueError("Temporal leakage detected: train data overlaps or follows test data.")


def validate_scaling_boundaries(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    tolerance_std: float = 5.0,
) -> None:
    """Warn/raise if test values are far outside train distribution after scaling."""
    for col in feature_cols:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std() + 1e-8
        z_scores = (test_df[col] - train_mean) / train_std
        if np.any(np.abs(z_scores) > tolerance_std):
            raise ValueError(f"Potential scaling mismatch on feature '{col}' (>|{tolerance_std}| std from train).")
