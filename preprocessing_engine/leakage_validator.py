"""Data-quality guards for temporal leakage and distribution-bound violations.

Purpose:
- Validates that preprocessing preserves chronological integrity and flags extreme
  train/test scaling mismatches.

Inputs:
- Train/test DataFrames and feature list from preprocessing stage.

Outputs:
- Raises validation errors (or warnings) used to hard-stop or monitor risky runs.
"""
from __future__ import annotations

import logging
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
    warn_only: bool = True,
) -> None:
    """Warn if test values are far outside train distribution after scaling."""
    for col in feature_cols:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std() + 1e-8
        z_scores = (test_df[col] - train_mean) / train_std
        if np.any(np.abs(z_scores) > tolerance_std):
            msg = f"Potential scaling mismatch on feature '{col}' (>|{tolerance_std}| std from train)."
            if warn_only:
                logging.warning(msg)
            else:
                raise ValueError(msg)
