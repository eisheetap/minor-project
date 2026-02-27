"""Orchestrate irrigation domain-shift experiment with synthetic data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import copy

import numpy as np
import pandas as pd

from data_generation import GLOBAL_SEED, generate_regions
from evaluation import (
    OUT_DIR,
    degradation_percentage,
    format_metrics_table,
    paired_t_test,
    plot_error_distribution,
    plot_predictions,
    plot_recovery_curve,
    recovery_percentage,
    regression_metrics,
)
from irrigation_simulation import simulate_irrigation
from models import predict_lstm, set_global_seed, train_lstm, train_random_forest
from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_windows,
    time_aware_split,
)
from transfer_learning import fine_tune_lstm


def temporal_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.15) -> Tuple[np.ndarray, ...]:
    val_size = max(1, int(len(X) * val_ratio))
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]


def scale_with_existing(df: pd.DataFrame, scaler) -> pd.DataFrame:
    scaled = df.copy()
    scaled[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
    return scaled


def run_experiment() -> Dict[str, any]:
    set_global_seed(GLOBAL_SEED)
    OUT_DIR.mkdir(exist_ok=True)

    # Generate synthetic data
    datasets = generate_regions(length=12_000, seed=GLOBAL_SEED)
    train_ratio = 0.7
    window_size = 7

    # Region A preprocessing
    from preprocessing import prepare_region_sequences  # local import to avoid cycle

    region_a = prepare_region_sequences(datasets["A"], train_ratio=train_ratio, window_size=window_size)

    # Region B with Region A scaler to avoid leakage and mimic deployment
    b_train_raw, b_test_raw = time_aware_split(datasets["B"], train_ratio=train_ratio)
    b_train_scaled = scale_with_existing(b_train_raw, region_a.scaler)
    b_test_scaled = scale_with_existing(b_test_raw, region_a.scaler)
    X_b_train, y_b_train = create_windows(b_train_scaled, window_size=window_size)
    X_b_test, y_b_test = create_windows(b_test_scaled, window_size=window_size)

    # In-domain (Region A) train/val split for LSTM
    X_a_train, y_a_train, X_a_val, y_a_val = temporal_val_split(region_a.X_train, region_a.y_train, val_ratio=0.15)

    # Train Random Forest on Region A
    rf_model = train_random_forest(region_a.X_train, region_a.y_train)
    rf_pred_a = rf_model.predict(region_a.X_test.reshape(len(region_a.X_test), -1))
    rf_pred_b = rf_model.predict(X_b_test.reshape(len(X_b_test), -1))

    rf_metrics_a = regression_metrics(region_a.y_test, rf_pred_a)
    rf_metrics_b = regression_metrics(y_b_test, rf_pred_b)

    # Train LSTM on Region A
    lstm_model, lstm_history = train_lstm(
        X_a_train,
        y_a_train,
        X_a_val,
        y_a_val,
        input_size=len(FEATURE_COLUMNS),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
        epochs=25,
        batch_size=64,
        seed=GLOBAL_SEED,
    )
    lstm_pred_a = predict_lstm(lstm_model, region_a.X_test)
    lstm_pred_b = predict_lstm(lstm_model, X_b_test)

    lstm_metrics_a = regression_metrics(region_a.y_test, lstm_pred_a)
    lstm_metrics_b = regression_metrics(y_b_test, lstm_pred_b)

    # Transfer learning on first 10% of Region B training (time-respecting)
    ft_size = max(1, int(0.10 * len(X_b_train)))
    ft_train_X = X_b_train[:ft_size]
    ft_train_y = y_b_train[:ft_size]
    # Use next slice as validation to avoid leakage
    ft_val_X = X_b_train[ft_size : ft_size * 2]
    ft_val_y = y_b_train[ft_size : ft_size * 2]
    if len(ft_val_X) == 0:
        ft_val_X, ft_val_y = ft_train_X, ft_train_y

    tl_model = copy.deepcopy(lstm_model)
    ft_history = fine_tune_lstm(
        tl_model,
        ft_train_X,
        ft_train_y,
        ft_val_X,
        ft_val_y,
        lr=5e-4,
        epochs=10,
        batch_size=32,
        freeze_backbone=False,
        seed=GLOBAL_SEED + 1,
    )
    tl_pred_b = predict_lstm(tl_model, X_b_test)
    tl_metrics_b = regression_metrics(y_b_test, tl_pred_b)

    # Statistics: degradation and recovery
    rf_degradation = degradation_percentage(rf_metrics_a["RMSE"], rf_metrics_b["RMSE"])
    lstm_degradation = degradation_percentage(lstm_metrics_a["RMSE"], lstm_metrics_b["RMSE"])
    recovery_rmse = recovery_percentage(lstm_metrics_b["RMSE"], tl_metrics_b["RMSE"])
    recovery_mae = recovery_percentage(lstm_metrics_b["MAE"], tl_metrics_b["MAE"])

    pre_errors = np.abs(y_b_test - lstm_pred_b)
    post_errors = np.abs(y_b_test - tl_pred_b)
    t_stat, p_value = paired_t_test(pre_errors, post_errors)

    # Irrigation simulation on Region B test
    soil_today = b_test_raw["soil_moisture"].values[window_size - 1 : -1]
    irrigation_results = simulate_irrigation(
        soil_moisture_today=soil_today,
        actual_next_day=y_b_test,
        predicted_next_day=tl_pred_b,
        baseline_threshold=30.0,
        ml_threshold=30.0,
    )

    # Tables
    perf_table = format_metrics_table(
        {
            "RF_in_domain_A": rf_metrics_a,
            "RF_cross_B": rf_metrics_b,
            "LSTM_in_domain_A": lstm_metrics_a,
            "LSTM_cross_B": lstm_metrics_b,
            "LSTM_cross_B_TL": tl_metrics_b,
        }
    )
    shift_table = pd.DataFrame(
        {
            "RF_degradation_%": [rf_degradation],
            "LSTM_degradation_%": [lstm_degradation],
            "Recovery_RMSE_%": [recovery_rmse],
            "Recovery_MAE_%": [recovery_mae],
            "t_stat": [t_stat],
            "p_value": [p_value],
        }
    )
    irrigation_table = pd.DataFrame([irrigation_results])

    perf_table.to_csv(OUT_DIR / "performance_table.csv")
    shift_table.to_csv(OUT_DIR / "degradation_recovery_table.csv", index=False)
    irrigation_table.to_csv(OUT_DIR / "irrigation_table.csv", index=False)

    # Plots
    plot_predictions(
        y_b_test,
        tl_pred_b,
        title="Region B: Prediction vs Actual (post-TL)",
        path=OUT_DIR / "prediction_vs_actual.png",
    )
    plot_error_distribution(
        errors_a=np.abs(region_a.y_test - lstm_pred_a),
        errors_b=pre_errors,
        labels=("Region A errors", "Region B errors"),
        title="Domain shift error distribution",
        path=OUT_DIR / "domain_shift_error.png",
    )
    plot_recovery_curve(
        history={"train_loss": ft_history["train_loss"], "val_loss": ft_history["val_loss"]},
        title="Transfer learning recovery curve",
        path=OUT_DIR / "recovery_curve.png",
    )

    summary = {
        "performance_table": perf_table,
        "shift_table": shift_table,
        "irrigation_table": irrigation_table,
        "rf_degradation_percent": rf_degradation,
        "lstm_degradation_percent": lstm_degradation,
        "recovery_rmse_percent": recovery_rmse,
        "recovery_mae_percent": recovery_mae,
        "t_stat": t_stat,
        "p_value": p_value,
        "irrigation_results": irrigation_results,
    }
    return summary


if __name__ == "__main__":
    results = run_experiment()
    print("=== Performance ===")
    print(results["performance_table"])
    print("\\n=== Degradation / Recovery ===")
    print(results["shift_table"])
    print("\\n=== Irrigation Simulation ===")
    print(results["irrigation_table"])
