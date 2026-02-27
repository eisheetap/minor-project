"""Irrigation simulation comparing rule-based vs ML policies."""
from __future__ import annotations

from typing import Dict

import numpy as np


def simulate_irrigation(
    soil_moisture_today: np.ndarray,
    actual_next_day: np.ndarray,
    predicted_next_day: np.ndarray,
    baseline_threshold: float = 30.0,
    ml_threshold: float = 30.0,
    irrigation_amount: float = 10.0,
    over_margin: float = 5.0,
) -> Dict[str, float]:
    """
    Simulate decisions and water use.

    over-irrigation: irrigate but next-day moisture exceeds threshold + margin.
    under-irrigation: skip irrigation and next-day moisture falls below threshold.
    """
    assert len(actual_next_day) == len(predicted_next_day) == len(soil_moisture_today)

    baseline_events = soil_moisture_today < baseline_threshold
    ml_events = predicted_next_day < ml_threshold

    baseline_water = baseline_events.sum() * irrigation_amount
    ml_water = ml_events.sum() * irrigation_amount

    over_irrigation = ((actual_next_day > (baseline_threshold + over_margin)) & ml_events).sum()
    under_irrigation = ((actual_next_day < baseline_threshold) & (~ml_events)).sum()

    return {
        "baseline_water": float(baseline_water),
        "ml_water": float(ml_water),
        "baseline_events": int(baseline_events.sum()),
        "ml_events": int(ml_events.sum()),
        "over_irrigation_events": int(over_irrigation),
        "under_irrigation_events": int(under_irrigation),
    }
