"""Evaluation metric primitives for domain-shift and transfer analysis.

Purpose:
- Computes core regression metrics and helper percentages for degradation/recovery.

Inputs:
- Ground-truth and predicted target arrays or scalar metric values.

Outputs:
- Numeric metric dictionaries and percentage indicators consumed by reports/plots.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def degradation_percentage(in_domain: float, cross_domain: float) -> float:
    return ((cross_domain - in_domain) / in_domain) * 100 if in_domain != 0 else np.nan


def recovery_percentage(pre: float, post: float) -> float:
    return ((pre - post) / pre) * 100 if pre != 0 else np.nan
