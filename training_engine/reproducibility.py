"""Determinism utilities for consistent experimental behavior across runs.

Purpose:
- Sets aligned random seeds for Python, NumPy, and PyTorch (CPU/GPU) and configures
  deterministic backend behavior where possible.

Inputs:
- Integer seed from run configuration.

Outputs:
- Process-level RNG/backend state updates that make training/evaluation reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
