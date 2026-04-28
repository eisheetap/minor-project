"""Cross-region score table builder for reporting and CSV export.

Purpose:
- Builds a standardized one-row performance matrix for train-region vs test-region
  evaluation with explicit degradation annotation.

Inputs:
- Train/test region labels and in-domain/cross-domain metric dictionaries.

Outputs:
- Pandas DataFrame row used in outputs/metrics and markdown reports.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from evaluation_engine.metrics import degradation_percentage


def build_matrix(
    train_region: str,
    test_region: str,
    in_domain_metrics: Dict[str, float],
    cross_domain_metrics: Dict[str, float],
) -> pd.DataFrame:
    row = {
        "Train Region": train_region,
        "Test Region": test_region,
        "RMSE": cross_domain_metrics["RMSE"],
        "MAE": cross_domain_metrics["MAE"],
        "R2": cross_domain_metrics["R2"],
        "Degradation_%": degradation_percentage(in_domain_metrics["RMSE"], cross_domain_metrics["RMSE"]),
    }
    return pd.DataFrame([row])
