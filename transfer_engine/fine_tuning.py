"""Fine-tuning orchestration on target region."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

from evaluation_engine.metrics import regression_metrics
from transfer_engine.transfer_strategies import fine_tune_lstm
from training_engine.trainer import predict_model


def run_fine_tuning(
    base_model,
    region_b_prepared,
    finetune_fraction: float,
    freeze_backbone: bool,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    strategy: str = "full",
    base_lr: float | None = None,
    head_lr: float | None = None,
    grad_clip: float = 1.0,
) -> Dict:
    """Fine-tune a pretrained LSTM on a slice of Region B and evaluate."""
    ft_size = max(1, int(finetune_fraction * len(region_b_prepared.X_train)))
    X_train_ft = region_b_prepared.X_train[:ft_size]
    y_train_ft = region_b_prepared.y_train[:ft_size]
    X_val_ft = region_b_prepared.X_train[ft_size : ft_size * 2]
    y_val_ft = region_b_prepared.y_train[ft_size : ft_size * 2]
    if len(X_val_ft) == 0:
        X_val_ft, y_val_ft = X_train_ft, y_train_ft

    # Clone model to keep base weights untouched
    model_copy = deepcopy(base_model)
    history = fine_tune_lstm(
        model=model_copy,
        X_train=X_train_ft,
        y_train=y_train_ft,
        X_val=X_val_ft,
        y_val=y_val_ft,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        freeze_backbone=freeze_backbone,
        seed=seed,
        strategy=strategy,
        base_lr=base_lr,
        head_lr=head_lr,
        grad_clip=grad_clip,
    )
    preds = predict_model("lstm", model_copy, region_b_prepared.X_test)
    metrics = regression_metrics(region_b_prepared.y_test, preds)
    return {"model": model_copy, "history": history, "predictions": preds, "metrics": metrics}
