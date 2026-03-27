"""Statistical testing utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


def paired_t_test(errors_before: np.ndarray, errors_after: np.ndarray) -> Tuple[float, float]:
    t_stat, p_value = stats.ttest_rel(errors_before, errors_after)
    return float(t_stat), float(p_value)


def summarize_runs(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values))
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def ensure_min_runs(run_values: List[float], minimum: int = 5) -> None:
    if len(run_values) < minimum:
        raise ValueError(f"Statistical tests require at least {minimum} runs; got {len(run_values)}.")
