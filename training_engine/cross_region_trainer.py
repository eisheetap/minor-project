"""Cross-region training and evaluation orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from data_engine.domain_shift import drift_score, summarize_shift
from evaluation_engine.metrics import regression_metrics
from preprocessing_engine.leakage_validator import assert_no_future_leakage, validate_scaling_boundaries
from preprocessing_engine.scaler import apply_scalers, fit_scaler
from preprocessing_engine.sequence_builder import build_sequences
from preprocessing_engine.splitter import time_based_split
from training_engine.trainer import TrainingResult, train_model


def _inverse_target(preds: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        return preds
    return scaler.inverse_transform(preds.reshape(-1, 1)).ravel()


@dataclass
class RegionPrepared:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    target_scaler: Any


@dataclass
class CrossRegionResult:
    train_region: str
    target_region: str
    region_a: RegionPrepared
    region_b: RegionPrepared
    training: TrainingResult
    in_domain_metrics: Dict[str, float]
    cross_domain_metrics: Dict[str, float]
    in_domain_predictions: np.ndarray
    cross_domain_predictions: np.ndarray
    domain_shift_summary: pd.DataFrame
    domain_drift_score: float
    scalers: Dict[str, Any]


def _temporal_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_size = max(1, int(len(X) * val_ratio))
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]


def _scale_and_window(
    split,
    feature_scaler,
    target_scaler,
    feature_cols,
    target_col: str,
    window_size: int,
    tolerance_std: float,
    warn_only: bool,
) -> RegionPrepared:
    scaled = apply_scalers(feature_scaler, target_scaler, split.train_df, split.test_df, feature_cols, target_col)
    validate_scaling_boundaries(scaled.train, scaled.test, feature_cols, tolerance_std=tolerance_std, warn_only=warn_only)
    X_train, y_train = build_sequences(scaled.train, feature_cols, target_col, window_size=window_size)
    X_test, y_test = build_sequences(scaled.test, feature_cols, target_col, window_size=window_size)
    # Raw targets for metrics in original scale
    y_train_raw = split.train_df[target_col].values[window_size:]
    y_test_raw = split.test_df[target_col].values[window_size:]
    return RegionPrepared(
        train_df=scaled.train,
        test_df=scaled.test,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        target_scaler=target_scaler,
    )


def run_cross_region_training(
    datasets: Dict[str, pd.DataFrame],
    model_kind: str,
    model,
    model_train_cfg: Dict,
    feature_cols,
    target_col: str,
    train_ratio: float,
    window_size: int,
    val_ratio: float,
    seed: int,
    scaling_mode: str = "per_region",
    tolerance_std: float = 5.0,
    warn_only: bool = True,
) -> CrossRegionResult:
    # Time-based splits
    split_a = time_based_split(datasets["A"], train_ratio=train_ratio)
    split_b = time_based_split(datasets["B"], train_ratio=train_ratio)
    assert_no_future_leakage(split_a.train_df, split_a.test_df)
    assert_no_future_leakage(split_b.train_df, split_b.test_df)

    scaling_mode = scaling_mode or "per_region"
    if scaling_mode == "combined":
        combined_train = pd.concat([split_a.train_df, split_b.train_df], axis=0)
        scaler_a = scaler_b = fit_scaler(combined_train, feature_cols)
        target_scaler_a = target_scaler_b = fit_scaler(combined_train, [target_col])
    elif scaling_mode == "per_region":
        scaler_a = fit_scaler(split_a.train_df, feature_cols)
        scaler_b = fit_scaler(split_b.train_df, feature_cols)
        target_scaler_a = fit_scaler(split_a.train_df, [target_col])
        target_scaler_b = fit_scaler(split_b.train_df, [target_col])
    else:
        raise ValueError(f"Unknown scaling_mode: {scaling_mode}")

    region_a = _scale_and_window(
        split_a,
        scaler_a,
        target_scaler_a,
        feature_cols,
        target_col,
        window_size,
        tolerance_std,
        warn_only,
    )
    region_b = _scale_and_window(
        split_b,
        scaler_b,
        target_scaler_b,
        feature_cols,
        target_col,
        window_size,
        tolerance_std,
        warn_only,
    )

    X_train, y_train, X_val, y_val = _temporal_val_split(region_a.X_train, region_a.y_train, val_ratio=val_ratio)
    training_result = train_model(
        kind=model_kind,
        model=model,
        train_cfg=model_train_cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=seed,
    )

    from training_engine.trainer import predict_model

    in_domain_predictions_scaled = predict_model(model_kind, training_result.model, region_a.X_test)
    cross_domain_predictions_scaled = predict_model(model_kind, training_result.model, region_b.X_test)

    in_domain_predictions = _inverse_target(in_domain_predictions_scaled, region_a.target_scaler)
    cross_domain_predictions = _inverse_target(cross_domain_predictions_scaled, region_b.target_scaler)

    in_domain_metrics = regression_metrics(region_a.y_test_raw, in_domain_predictions)
    cross_domain_metrics = regression_metrics(region_b.y_test_raw, cross_domain_predictions)

    shift_summary = summarize_shift(region_a.train_df, region_b.train_df, feature_cols)
    drift = drift_score(region_a.train_df, region_b.train_df, feature_cols)

    return CrossRegionResult(
        train_region="A",
        target_region="B",
        region_a=region_a,
        region_b=region_b,
        training=training_result,
        in_domain_metrics=in_domain_metrics,
        cross_domain_metrics=cross_domain_metrics,
        in_domain_predictions=in_domain_predictions,
        cross_domain_predictions=cross_domain_predictions,
        domain_shift_summary=shift_summary,
        domain_drift_score=drift,
        scalers={
            "A": {"features": scaler_a, "target": target_scaler_a},
            "B": {"features": scaler_b, "target": target_scaler_b},
        },
    )
