"""Feature/target normalization helpers with train-only fitting semantics.

Purpose:
- Applies standardization to features (and optionally target) while preventing
  test-set information leakage into scaler fitting.

Inputs:
- Train/test DataFrames, selected feature columns, target column, and fitted/fit-able scalers.

Outputs:
- Scaled train/test DataFrames plus scaler objects bundled in ``ScaledData``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ScaledData:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_scaler: StandardScaler
    target_scaler: Optional[StandardScaler] = None


def fit_scaler(train_df: pd.DataFrame, cols: Iterable[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[list(cols)])
    return scaler


def apply_feature_scaler(
    scaler: StandardScaler, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(feature_cols)
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[cols] = scaler.transform(train_df[cols])
    test_scaled[cols] = scaler.transform(test_df[cols])
    return train_scaled, test_scaled


def apply_scalers(
    feature_scaler: StandardScaler,
    target_scaler: Optional[StandardScaler],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
) -> ScaledData:
    train_scaled, test_scaled = apply_feature_scaler(feature_scaler, train_df, test_df, feature_cols)

    if target_scaler:
        train_scaled[target_col] = target_scaler.transform(train_df[[target_col]])
        test_scaled[target_col] = target_scaler.transform(test_df[[target_col]])

    return ScaledData(train=train_scaled, test=test_scaled, feature_scaler=feature_scaler, target_scaler=target_scaler)
