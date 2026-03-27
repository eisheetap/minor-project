"""Time-based train/test splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.7) -> SplitResult:
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if not (train_df["timestamp"].max() < test_df["timestamp"].min()):
        raise ValueError("Temporal leakage detected: train max timestamp overlaps test min.")
    return SplitResult(train_df=train_df, test_df=test_df)
