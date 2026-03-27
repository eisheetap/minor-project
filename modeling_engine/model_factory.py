"""Factory for creating models from config."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from modeling_engine.baseline_models import LinearConfig, RFConfig
from modeling_engine.lstm_model import LSTMRegressor


@dataclass
class ModelSpec:
    kind: str  # "rf", "linear", "lstm"
    model: Any
    config: Any


def create_model(model_cfg: Dict[str, Any], input_size: int) -> ModelSpec:
    kind = model_cfg.get("type", "lstm")
    if kind == "rf":
        rf_cfg = model_cfg.get("rf", {})
        cfg = RFConfig(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_depth=rf_cfg.get("max_depth", None),
            seed=model_cfg.get("seed", 123),
        )
        return ModelSpec(kind="rf", model=None, config=cfg)
    if kind == "linear":
        lin_cfg = model_cfg.get("linear", {})
        cfg = LinearConfig(
            fit_intercept=lin_cfg.get("fit_intercept", True),
            n_jobs=lin_cfg.get("n_jobs", -1),
        )
        return ModelSpec(kind="linear", model=None, config=cfg)
    if kind == "lstm":
        lstm_cfg = model_cfg.get("lstm", {})
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=lstm_cfg.get("hidden_size", 64),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.1),
        )
        return ModelSpec(kind="lstm", model=model, config=lstm_cfg)
    raise ValueError(f"Unsupported model type: {kind}")
