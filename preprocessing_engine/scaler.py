"""Scaling utilities that fit on training only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ScaledData:
    train: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


def fit_scaler(train_df: pd.DataFrame, feature_cols: Iterable[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[list(feature_cols)])
    return scaler


def apply_scaler(
    scaler: StandardScaler, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Iterable[str]
) -> ScaledData:
    cols = list(feature_cols)
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[cols] = scaler.transform(train_df[cols])
    test_scaled[cols] = scaler.transform(test_df[cols])
    return ScaledData(train=train_scaled, test=test_scaled, scaler=scaler)
