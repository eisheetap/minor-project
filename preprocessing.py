"""Preprocessing utilities: time-based splits, scaling, and sliding windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "rainfall",
    "evapotranspiration",
    "soil_moisture",
]
TARGET_COLUMN = "next_day_soil_moisture"


@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    scaler: StandardScaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def time_aware_split(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split without shuffling to avoid leakage."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    assert train_df["timestamp"].max() < test_df["timestamp"].min(), "Temporal leakage detected in split."
    return train_df, test_df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train only and transform both splits."""
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_features = scaler.transform(test_df[FEATURE_COLUMNS])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[FEATURE_COLUMNS] = train_features
    test_scaled[FEATURE_COLUMNS] = test_features
    return train_scaled, test_scaled, scaler


def create_windows(df: pd.DataFrame, window_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for sequence models."""
    X, y = [], []
    values = df[FEATURE_COLUMNS].values
    targets = df[TARGET_COLUMN].values
    for i in range(len(df) - window_size):
        X.append(values[i : i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)


def prepare_region_sequences(
    df: pd.DataFrame, train_ratio: float = 0.7, window_size: int = 7
) -> SplitData:
    """Time split, scale, and window a single regional dataset."""
    train_df, test_df = time_aware_split(df, train_ratio=train_ratio)
    train_scaled, test_scaled, scaler = scale_features(train_df, test_df)
    X_train, y_train = create_windows(train_scaled, window_size=window_size)
    X_test, y_test = create_windows(test_scaled, window_size=window_size)
    return SplitData(
        train_df=train_scaled,
        test_df=test_scaled,
        scaler=scaler,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def prepare_all_regions(
    datasets: Dict[str, pd.DataFrame], train_ratio: float = 0.7, window_size: int = 7
) -> Dict[str, SplitData]:
    """Convenience wrapper for both regions."""
    splits: Dict[str, SplitData] = {}
    for key, df in datasets.items():
        splits[key] = prepare_region_sequences(df, train_ratio=train_ratio, window_size=window_size)
    return splits
