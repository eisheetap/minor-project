"""Cross-region training and evaluation orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_engine.domain_shift import drift_score, summarize_shift
from evaluation_engine.metrics import regression_metrics
from preprocessing_engine.leakage_validator import assert_no_future_leakage, validate_scaling_boundaries
from preprocessing_engine.scaler import apply_scaler, fit_scaler
from preprocessing_engine.sequence_builder import build_sequences
from preprocessing_engine.splitter import time_based_split
from training_engine.trainer import TrainingResult, train_model


@dataclass
class RegionPrepared:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


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
    scaler


def _temporal_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_size = max(1, int(len(X) * val_ratio))
    return X[:-val_size], y[:-val_size], X[-val_size:], y[-val_size:]


def _prepare_region(
    df: pd.DataFrame, feature_cols, target_col: str, train_ratio: float, window_size: int, scaler=None
) -> Tuple[RegionPrepared, any]:
    split = time_based_split(df, train_ratio=train_ratio)
    assert_no_future_leakage(split.train_df, split.test_df)

    if scaler is None:
        scaler = fit_scaler(split.train_df, feature_cols)
    scaled = apply_scaler(scaler, split.train_df, split.test_df, feature_cols)
    validate_scaling_boundaries(scaled.train, scaled.test, feature_cols)

    X_train, y_train = build_sequences(scaled.train, feature_cols, target_col, window_size=window_size)
    X_test, y_test = build_sequences(scaled.test, feature_cols, target_col, window_size=window_size)
    prepared = RegionPrepared(
        train_df=scaled.train,
        test_df=scaled.test,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return prepared, scaler


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
) -> CrossRegionResult:
    # Train region A
    region_a, scaler = _prepare_region(
        datasets["A"], feature_cols=feature_cols, target_col=target_col, train_ratio=train_ratio, window_size=window_size
    )
    # Target region B using Region A scaler
    region_b, _ = _prepare_region(
        datasets["B"],
        feature_cols=feature_cols,
        target_col=target_col,
        train_ratio=train_ratio,
        window_size=window_size,
        scaler=scaler,
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

    in_domain_predictions = predict_model(model_kind, training_result.model, region_a.X_test)
    cross_domain_predictions = predict_model(model_kind, training_result.model, region_b.X_test)

    in_domain_metrics = regression_metrics(region_a.y_test, in_domain_predictions)
    cross_domain_metrics = regression_metrics(region_b.y_test, cross_domain_predictions)

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
        scaler=scaler,
    )
