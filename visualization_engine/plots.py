"""Centralized plotting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def plot_error_comparison(errors_a: np.ndarray, errors_b: np.ndarray, labels: Tuple[str, str], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def plot_rmse_bar(labels: List[str], values: List[float], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_recovery_curve(history: Dict[str, List[float]], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if history.get("train_loss"):
        plt.plot(history["train_loss"], label="Train")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_feature_distributions(df_a: pd.DataFrame, df_b: pd.DataFrame, feature_cols: Iterable[str], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(list(feature_cols))
    n_rows = int(np.ceil(n_cols / 2))
    plt.figure(figsize=(10, 4 * n_rows))
    for idx, col in enumerate(feature_cols):
        plt.subplot(n_rows, 2, idx + 1)
        plt.hist(df_a[col], bins=40, alpha=0.6, label="Train region")
        plt.hist(df_b[col], bins=40, alpha=0.6, label="Target region")
        plt.title(col)
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, dpi=200)
    plt.close()
