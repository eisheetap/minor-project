"""Feature-distribution drift analysis utilities between source and target regions.

Purpose:
- Quantifies domain shift using mean/variance statistics and KL-divergence proxies.

Inputs:
- Two region DataFrames (source and target) and the list of feature columns to compare.

Outputs:
- Per-feature shift summary table and aggregate drift score used by evaluation/reporting.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.stats import entropy


def _kl_divergence(a: np.ndarray, b: np.ndarray, bins: int = 40, epsilon: float = 1e-8) -> float:
    """Approximate KL divergence using histograms."""
    hist_a, bin_edges = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bin_edges, density=True)
    hist_a = hist_a + epsilon
    hist_b = hist_b + epsilon
    return float(entropy(hist_a, hist_b))


def summarize_shift(df_a: pd.DataFrame, df_b: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Return mean/variance deltas and KL divergence per feature."""
    records: Dict[str, Dict[str, float]] = {}
    for col in feature_cols:
        a_vals = df_a[col].to_numpy()
        b_vals = df_b[col].to_numpy()
        records[col] = {
            "mean_diff": float(np.mean(b_vals) - np.mean(a_vals)),
            "var_ratio": float(np.var(b_vals) / (np.var(a_vals) + 1e-8)),
            "kl_divergence": _kl_divergence(a_vals, b_vals),
        }
    return pd.DataFrame.from_dict(records, orient="index")


def drift_score(df_a: pd.DataFrame, df_b: pd.DataFrame, feature_cols: Iterable[str]) -> float:
    """Aggregate drift score as average KL across features."""
    summary = summarize_shift(df_a, df_b, feature_cols)
    return float(summary["kl_divergence"].mean())
