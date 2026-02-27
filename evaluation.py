"""Evaluation utilities: metrics, degradation, recovery, plots, and tables."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


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


def paired_t_test(errors_before: np.ndarray, errors_after: np.ndarray) -> Tuple[float, float]:
    """Paired t-test over per-sample absolute errors."""
    t_stat, p_value = stats.ttest_rel(errors_before, errors_after)
    return t_stat, p_value


def format_metrics_table(rows: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).T[["RMSE", "MAE", "R2"]]


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label="Actual", alpha=0.8)
    plt.plot(y_pred, label="Predicted", alpha=0.8)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Soil moisture")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_error_distribution(
    errors_a: np.ndarray, errors_b: np.ndarray, labels: Tuple[str, str], title: str, path: Path
) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(errors_a, bins=40, alpha=0.6, label=labels[0])
    plt.hist(errors_b, bins=40, alpha=0.6, label=labels[1])
    plt.title(title)
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_recovery_curve(history: Dict[str, list], title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history.get("train_loss", []), label="Train")
    plt.plot(history.get("val_loss", []), label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
