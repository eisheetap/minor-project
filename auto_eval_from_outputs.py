#!/usr/bin/env python3
"""
Lightweight auto-evaluation that reads metrics produced by main.py and emits
comparison tables and a concise summary. This does NOT require supplying arrays
manually; it uses the CSVs generated in outputs/metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def load_metrics(metrics_dir: Path) -> Dict[str, Any]:
    cross_path = metrics_dir / "cross_region_matrix.csv"
    transfer_path = metrics_dir / "transfer_metrics.csv"

    if not cross_path.exists() or not transfer_path.exists():
        raise FileNotFoundError("Expected metrics files not found in outputs/metrics.")

    cross_df = pd.read_csv(cross_path)
    transfer_df = pd.read_csv(transfer_path)

    # Use first row as representative (main saves a single row)
    baseline_rmse = float(cross_df.iloc[0]["RMSE"])
    transfer_rmse = float(transfer_df.iloc[0]["RMSE"])
    baseline_mae = float(cross_df.iloc[0]["MAE"])
    transfer_mae = float(transfer_df.iloc[0]["MAE"])
    baseline_r2 = float(cross_df.iloc[0]["R2"])
    transfer_r2 = float(transfer_df.iloc[0]["R2"])

    recovery_pct = ((baseline_rmse - transfer_rmse) / baseline_rmse) * 100 if baseline_rmse != 0 else np.nan

    return {
        "baseline_rmse": baseline_rmse,
        "transfer_rmse": transfer_rmse,
        "baseline_mae": baseline_mae,
        "transfer_mae": transfer_mae,
        "baseline_r2": baseline_r2,
        "transfer_r2": transfer_r2,
        "recovery_pct": recovery_pct,
    }


def write_tables(metrics: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "Metric": "RMSE",
            "Baseline (A->B)": f"{metrics['baseline_rmse']:.4f}",
            "Transfer": f"{metrics['transfer_rmse']:.4f}",
            "Improvement": f"{metrics['recovery_pct']:.2f}%",
        },
        {
            "Metric": "MAE",
            "Baseline (A->B)": f"{metrics['baseline_mae']:.4f}",
            "Transfer": f"{metrics['transfer_mae']:.4f}",
            "Improvement": "-",
        },
        {
            "Metric": "R2",
            "Baseline (A->B)": f"{metrics['baseline_r2']:.4f}",
            "Transfer": f"{metrics['transfer_r2']:.4f}",
            "Improvement": "-",
        },
        {
            "Metric": "Recovery (%)",
            "Baseline (A->B)": "-",
            "Transfer": "-",
            "Improvement": f"{metrics['recovery_pct']:.2f}%",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "auto_results_table.csv", index=False, encoding="utf-8")
    (out_dir / "auto_results_table.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    return df


def main() -> None:
    metrics_dir = Path("outputs/metrics")
    results_dir = Path("outputs/metrics")

    metrics = load_metrics(metrics_dir)
    table = write_tables(metrics, results_dir)

    print("\n=== Auto-Eval (from outputs/metrics) ===")
    print(table.to_string(index=False))
    print(f"\nRecovery (%): {metrics['recovery_pct']:.2f}%")
    print("\nGenerated files:")
    print(" - outputs/metrics/auto_results_table.csv")
    print(" - outputs/metrics/auto_results_table.md")


if __name__ == "__main__":
    main()
