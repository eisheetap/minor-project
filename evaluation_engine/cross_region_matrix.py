"""Cross-region performance matrix construction."""
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
