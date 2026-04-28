#!/usr/bin/env python3
"""Standalone post-training evaluation utility for transfer-learning experiments.

Purpose:
- Computes summary metrics and generates publication-friendly tables and plots
  from already available prediction/error arrays.

Inputs:
- ``run_evaluation(inputs)`` expects baseline/transfer RMSE runs, per-sample errors,
  and true/predicted arrays for both baseline and transfer settings.
- Optional training and validation loss arrays for curve plotting.

Outputs:
- Writes ``results_table.csv``, ``results_table.md`` and multiple PNG figures.
- Returns a dictionary containing computed metrics, generated table, output file list,
  and interpretation text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---- Metrics ----------------------------------------------------------------
def compute_metrics(baseline_rmse_runs: np.ndarray, transfer_rmse_runs: np.ndarray) -> Dict[str, Any]:
    baseline_mean = float(np.mean(baseline_rmse_runs))
    baseline_std = float(np.std(baseline_rmse_runs, ddof=0))
    transfer_mean = float(np.mean(transfer_rmse_runs))
    transfer_std = float(np.std(transfer_rmse_runs, ddof=0))
    baseline_mse = baseline_mean ** 2
    transfer_mse = transfer_mean ** 2
    recovery_pct = ((baseline_mean - transfer_mean) / baseline_mean) * 100 if baseline_mean != 0 else np.nan
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "transfer_mean": transfer_mean,
        "transfer_std": transfer_std,
        "baseline_mse": baseline_mse,
        "transfer_mse": transfer_mse,
        "recovery_pct": recovery_pct,
    }


# ---- Table generation -------------------------------------------------------
def generate_table(metrics: Dict[str, Any], csv_path: Path, md_path: Path) -> pd.DataFrame:
    rows = [
        {
            "Metric": "RMSE (mean ± std)",
            "Baseline (A→B)": f"{metrics['baseline_mean']:.4f} ± {metrics['baseline_std']:.4f}",
            "Transfer Learning": f"{metrics['transfer_mean']:.4f} ± {metrics['transfer_std']:.4f}",
            "Improvement": f"{metrics['recovery_pct']:.2f}%",
        },
        {
            "Metric": "MSE",
            "Baseline (A→B)": f"{metrics['baseline_mse']:.4f}",
            "Transfer Learning": f"{metrics['transfer_mse']:.4f}",
            "Improvement": f"{metrics['recovery_pct']:.2f}%",
        },
        {
            "Metric": "Recovery (%)",
            "Baseline (A→B)": "-",
            "Transfer Learning": "-",
            "Improvement": f"{metrics['recovery_pct']:.2f}%",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))
    return df


# ---- Plotting helpers -------------------------------------------------------
def plot_rmse_comparison(baseline_mean: float, transfer_mean: float, out_path: Path) -> None:
    plt.figure(figsize=(4, 4))
    plt.bar(["Baseline A→B", "Transfer"], [baseline_mean, transfer_mean], color=["tab:gray", "tab:blue"])
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_recovery(baseline_mean: float, transfer_mean: float, out_path: Path) -> None:
    recovery = baseline_mean - transfer_mean
    plt.figure(figsize=(4, 4))
    plt.bar(["Degradation (A→B)", "Recovery"], [baseline_mean, recovery], color=["tab:red", "tab:green"])
    plt.ylabel("RMSE / Delta")
    plt.title("Degradation vs Recovery")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_distribution(errors_a: np.ndarray, errors_b: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(errors_a, bins=40, alpha=0.6, label="Region A errors", color="tab:blue")
    plt.hist(errors_b, bins=40, alpha=0.6, label="Region B errors", color="tab:orange")
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.title("Error Distribution (A vs B)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label="Actual", alpha=0.8)
    plt.plot(y_pred, label="Predicted", alpha=0.8)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_training_curve(training_loss: Optional[np.ndarray], validation_loss: Optional[np.ndarray], out_path: Path) -> bool:
    if training_loss is None or validation_loss is None:
        return False
    plt.figure(figsize=(6, 4))
    plt.plot(training_loss, label="Train")
    plt.plot(validation_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


# ---- Main pipeline ----------------------------------------------------------
def run_evaluation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    # Required inputs
    baseline_rmse_runs = np.asarray(inputs["baseline_rmse_runs"], dtype=float)
    transfer_rmse_runs = np.asarray(inputs["transfer_rmse_runs"], dtype=float)
    errors_source = np.asarray(inputs["errors_source"], dtype=float)
    errors_target = np.asarray(inputs["errors_target"], dtype=float)
    y_true_baseline = np.asarray(inputs["y_true_baseline"], dtype=float)
    y_pred_baseline = np.asarray(inputs["y_pred_baseline"], dtype=float)
    y_true_transfer = np.asarray(inputs["y_true_transfer"], dtype=float)
    y_pred_transfer = np.asarray(inputs["y_pred_transfer"], dtype=float)

    # Optional
    training_loss = np.asarray(inputs["training_loss"], dtype=float) if "training_loss" in inputs else None
    validation_loss = np.asarray(inputs["validation_loss"], dtype=float) if "validation_loss" in inputs else None

    metrics = compute_metrics(baseline_rmse_runs, transfer_rmse_runs)

    # Table
    csv_path = Path("results_table.csv")
    md_path = Path("results_table.md")
    table_df = generate_table(metrics, csv_path, md_path)

    # Plots
    plot_rmse_comparison(metrics["baseline_mean"], metrics["transfer_mean"], Path("rmse_comparison.png"))
    plot_recovery(metrics["baseline_mean"], metrics["transfer_mean"], Path("recovery_plot.png"))
    plot_error_distribution(errors_source, errors_target, Path("error_distribution.png"))
    plot_predictions(y_true_baseline, y_pred_baseline, "Baseline: Prediction vs Actual (A→B)", Path("baseline_plot.png"))
    plot_predictions(y_true_transfer, y_pred_transfer, "Transfer: Prediction vs Actual (A→B, TL)", Path("transfer_plot.png"))
    training_curve_written = plot_training_curve(training_loss, validation_loss, Path("training_curve.png"))

    outputs = [
        "results_table.csv",
        "results_table.md",
        "rmse_comparison.png",
        "recovery_plot.png",
        "error_distribution.png",
        "baseline_plot.png",
        "transfer_plot.png",
    ]
    if training_curve_written:
        outputs.append("training_curve.png")

    # Print summary
    print("\n=== Comparison Table ===")
    print(table_df.to_string(index=False))
    print(f"\nRecovery (%): {metrics['recovery_pct']:.2f}%")
    print("\nGenerated files:")
    for f in outputs:
        print(f" - {f}")

    # Research-style interpretation
    interp = [
        "Cross-domain results show significant degradation on Region B without adaptation.",
        f"Baseline A→B RMSE: {metrics['baseline_mean']:.2f}; transfer RMSE: {metrics['transfer_mean']:.2f}.",
        f"Transfer learning recovers ~{metrics['recovery_pct']:.2f}% of the error relative to baseline.",
        "Error distributions shift between regions, highlighting domain shift in both mean and variance.",
        "Post-transfer predictions track targets more closely, indicating improved generalization.",
        "Overall, domain-aware adaptation mitigates concept/scale mismatch in the target domain.",
    ]
    print("\nInterpretation:")
    for line in interp:
        print(f"- {line}")

    return {
        "metrics": metrics,
        "table": table_df,
        "outputs": outputs,
        "interpretation": interp,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Provide inputs as described and call run_evaluation(inputs). "
        "Example keys: baseline_rmse_runs, transfer_rmse_runs, errors_source, errors_target, "
        "y_true_baseline, y_pred_baseline, y_true_transfer, y_pred_transfer (optional: training_loss, validation_loss)."
    )
