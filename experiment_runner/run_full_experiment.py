"""Primary orchestration module for the complete experimental pipeline.

Purpose:
- Coordinates config loading, dataset preparation, repeated runs, model training,
  transfer adaptation, statistical tests, robustness checks, visualization, and reporting.

Inputs:
- YAML config path and optional transfer override from CLI/wrapper scripts.

Outputs:
- Persisted metrics tables, plots, logs, and markdown report in configured directories.
- Console/log summaries describing aggregated baseline and transfer behavior.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data_engine.data_generator import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_regions_from_config,
    save_datasets,
)
from evaluation_engine.cross_region_matrix import build_matrix
from evaluation_engine.metrics import degradation_percentage, regression_metrics, recovery_percentage
from evaluation_engine.robustness_analysis import evaluate_robustness
from evaluation_engine.statistical_tests import ensure_min_runs, paired_t_test, summarize_runs
from experiment_runner.utils import ensure_output_dirs, load_config, setup_logging
from modeling_engine.model_factory import create_model
from training_engine.cross_region_trainer import run_cross_region_training
from training_engine.reproducibility import set_global_seed
from transfer_engine.fine_tuning import run_fine_tuning
from visualization_engine.plots import (
    plot_rmse_bar,
    plot_error_comparison,
    plot_feature_distributions,
    plot_predictions,
    plot_recovery_curve,
)
from visualization_engine.reports import write_markdown_report


def _aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(metric_list)
    return {f"mean_{col}": df[col].mean() for col in df.columns} | {f"std_{col}": df[col].std() for col in df.columns}


def run_pipeline(config_path: Path, transfer_override: bool | None = None, min_runs_required: int = 5) -> None:
    cfg = load_config(config_path)
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    transfer_cfg = cfg["transfer"].copy()
    eval_cfg = cfg["evaluation"]

    if transfer_override is not None:
        transfer_cfg["enabled"] = transfer_override

    output_dir = Path(run_cfg["output_dir"])
    log_dir = Path(run_cfg["log_dir"])
    metrics_dir = Path(run_cfg["metrics_dir"])
    plots_dir = Path(run_cfg["plots_dir"])
    ensure_output_dirs(output_dir, metrics_dir, plots_dir)

    log_path = setup_logging(run_cfg["run_name"], log_dir)
    logging.info("Logging to %s", log_path)

    feature_cols = data_cfg["features"]["columns"]
    target_col = data_cfg["features"]["target"]

    # Data loading
    datasets = load_regions_from_config(
        source=data_cfg["source"],
        synthetic_length=data_cfg["synthetic"]["length"],
        seed=run_cfg["seed"],
        region_seed_offset=data_cfg["synthetic"]["region_seed_offset"],
        external_a=data_cfg["external"]["region_a_path"],
        external_b=data_cfg["external"]["region_b_path"],
        timestamp_col=data_cfg["external"]["timestamp_col"],
    )
    if run_cfg.get("save_raw_data", False):
        save_datasets(datasets, Path("DATA"))

    eval_runs = eval_cfg.get("runs", 5)
    ensure_min_runs(list(range(eval_runs)), minimum=min_runs_required)

    in_metrics_runs: List[Dict[str, float]] = []
    cross_metrics_runs: List[Dict[str, float]] = []
    transfer_metrics_runs: List[Dict[str, float]] = []
    errors_before: List[np.ndarray] = []
    errors_after: List[np.ndarray] = []
    last_cross_result = None
    last_transfer = None

    for i in range(eval_runs):
        seed = run_cfg["seed"] + i
        set_global_seed(seed)
        model_spec = create_model(model_cfg, input_size=len(feature_cols))
        # Training config for model
        model_train_cfg = vars(model_spec.config) if hasattr(model_spec.config, "__dict__") else dict(model_spec.config)
        if model_spec.kind == "lstm":
            model_train_cfg = {
                "lr": train_cfg.get("lr", 1e-3),
                "epochs": train_cfg.get("epochs", 25),
                "batch_size": train_cfg.get("batch_size", 64),
                "grad_clip": train_cfg.get("grad_clip", 1.0),
                "early_stopping_patience": train_cfg.get("early_stopping_patience", 5),
            }

        cross_result = run_cross_region_training(
            datasets=datasets,
            model_kind=model_spec.kind,
            model=model_spec.model,
            model_train_cfg=model_train_cfg,
            feature_cols=feature_cols,
            target_col=target_col,
            train_ratio=prep_cfg["train_ratio"],
            window_size=prep_cfg["window_size"],
            val_ratio=train_cfg.get("val_ratio", 0.15),
            seed=seed,
            scaling_mode=prep_cfg.get("scaling_mode", "per_region"),
            tolerance_std=prep_cfg.get("leakage_std_tolerance", 5.0),
            warn_only=prep_cfg.get("leakage_warn_only", True),
        )
        in_metrics_runs.append(cross_result.in_domain_metrics)
        cross_metrics_runs.append(cross_result.cross_domain_metrics)

        errors_before.append(np.abs(cross_result.region_b.y_test - cross_result.cross_domain_predictions))

        transfer_result = None
        if transfer_cfg.get("enabled", True) and model_spec.kind == "lstm":
            transfer_result = run_fine_tuning(
                base_model=cross_result.training.model,
                region_b_prepared=cross_result.region_b,
                finetune_fraction=transfer_cfg["finetune_fraction"],
                freeze_backbone=transfer_cfg["freeze_backbone"],
                lr=transfer_cfg["lr"],
                epochs=transfer_cfg["epochs"],
                batch_size=transfer_cfg["batch_size"],
                seed=seed + 999,
                strategy=transfer_cfg.get("strategy", "full"),
                base_lr=transfer_cfg.get("base_lr"),
                head_lr=transfer_cfg.get("head_lr"),
                grad_clip=transfer_cfg.get("grad_clip", 1.0),
            )
            transfer_metrics_runs.append(transfer_result["metrics"])
            errors_after.append(np.abs(cross_result.region_b.y_test - transfer_result["predictions"]))
            last_transfer = transfer_result

        last_cross_result = cross_result

    # Aggregate metrics
    in_metrics_mean = _aggregate_metrics(in_metrics_runs)
    cross_metrics_mean = _aggregate_metrics(cross_metrics_runs)
    cross_table = build_matrix("A", "B", in_metrics_runs[0], cross_metrics_runs[0])
    cross_table.to_csv(metrics_dir / "cross_region_matrix.csv", index=False)

    # Plots and summaries use last run for concrete visuals
    plot_predictions(
        last_cross_result.region_b.y_test,
        last_cross_result.cross_domain_predictions,
        title="Region B: Prediction vs Actual (baseline)",
        path=plots_dir / "prediction_vs_actual_baseline.png",
    )
    plot_error_comparison(
        errors_a=np.abs(last_cross_result.region_a.y_test - last_cross_result.in_domain_predictions),
        errors_b=np.abs(last_cross_result.region_b.y_test - last_cross_result.cross_domain_predictions),
        labels=("Region A errors", "Region B errors"),
        title="Domain shift error distribution",
        path=plots_dir / "domain_shift_error.png",
    )
    plot_feature_distributions(
        last_cross_result.region_a.train_df,
        last_cross_result.region_b.train_df,
        feature_cols=feature_cols,
        title="Feature distributions: Train vs Target",
        path=plots_dir / "feature_distribution.png",
    )
    plot_rmse_bar(
        labels=["A→B RMSE"],
        values=[cross_metrics_runs[0]["RMSE"]],
        title="Cross-domain RMSE",
        path=plots_dir / "rmse_bar.png",
    )

    transfer_metrics_mean: Dict[str, float] | None = None
    stats_summary: Dict[str, float] | None = None
    robustness_summary: Dict[str, Dict[str, float]] | None = None

    if transfer_metrics_runs:
        transfer_metrics_mean = _aggregate_metrics(transfer_metrics_runs)
        # Recovery percentages using first-run numbers for interpretability
        recovery_rmse = recovery_percentage(cross_metrics_runs[0]["RMSE"], transfer_metrics_runs[0]["RMSE"])
        recovery_mae = recovery_percentage(cross_metrics_runs[0]["MAE"], transfer_metrics_runs[0]["MAE"])
        pd.DataFrame([transfer_metrics_runs[0]]).to_csv(metrics_dir / "transfer_metrics.csv", index=False)
        plot_predictions(
            last_cross_result.region_b.y_test,
            last_transfer["predictions"],
            title="Region B: Prediction vs Actual (transfer)",
            path=plots_dir / "prediction_vs_actual_transfer.png",
        )
        plot_recovery_curve(
            history=last_transfer["history"],
            title="Transfer learning recovery curve",
            path=plots_dir / "recovery_curve.png",
        )

        if len(errors_after) >= 1:
            t_stat, p_value = paired_t_test(errors_before[0], errors_after[0])
            stats_summary = {
                "t_stat": t_stat,
                "p_value": p_value,
                "recovery_RMSE_%": recovery_rmse,
                "recovery_MAE_%": recovery_mae,
            }
            pd.DataFrame([stats_summary]).to_csv(metrics_dir / "statistical_tests.csv", index=False)

        robustness_summary = evaluate_robustness(
            model_kind="lstm",
            model=last_transfer["model"],
            X_test=last_cross_result.region_b.X_test,
            y_test=last_cross_result.region_b.y_test,
            noise_std=eval_cfg["noise_std"],
            missing_frac=eval_cfg["missing_frac"],
        )
        pd.DataFrame.from_dict(robustness_summary, orient="index").to_csv(metrics_dir / "robustness.csv")

        plot_rmse_bar(
            labels=["A→B", "A→B (TL)"],
            values=[cross_metrics_runs[0]["RMSE"], transfer_metrics_runs[0]["RMSE"]],
            title="RMSE before/after transfer",
            path=plots_dir / "rmse_transfer_bar.png",
        )

    # Domain shift summary
    last_cross_result.domain_shift_summary.to_csv(metrics_dir / "domain_shift_summary.csv")

    write_markdown_report(
        path=output_dir / "report.md",
        run_name=run_cfg["run_name"],
        cross_region_table=cross_table,
        transfer_metrics=transfer_metrics_runs[0] if transfer_metrics_runs else None,
        stats_summary=stats_summary,
        robustness_summary=robustness_summary,
    )

    logging.info("Cross-region metrics (mean): %s", in_metrics_mean | cross_metrics_mean)
    if transfer_metrics_mean:
        logging.info("Transfer metrics (mean): %s", transfer_metrics_mean)
    logging.info("Finished run. Outputs saved to %s", output_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    run_pipeline(Path(args.config))


if __name__ == "__main__":
    main()
