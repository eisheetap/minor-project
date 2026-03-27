"""Region parameterization for synthetic agro-climatic data."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionParams:
    name: str
    temp_mean: float
    temp_std: float
    humidity_base: float
    humidity_variance: float
    rainfall_shape: float
    rainfall_scale: float
    rainfall_spike_chance: float
    rainfall_spike_scale: float
    evap_temp_coeff: float
    evap_humidity_coeff: float
    decay_rate: float
    infiltration_eff: float


def get_region_a_params() -> RegionParams:
    """Semi-arid, hotter, sparse rainfall."""
    return RegionParams(
        name="A_semi_arid",
        temp_mean=30.0,
        temp_std=3.0,
        humidity_base=25.0,
        humidity_variance=3.0,
        rainfall_shape=0.8,
        rainfall_scale=1.0,
        rainfall_spike_chance=0.02,
        rainfall_spike_scale=6.0,
        evap_temp_coeff=0.12,
        evap_humidity_coeff=0.03,
        decay_rate=0.06,
        infiltration_eff=0.55,
    )


def get_region_b_params() -> RegionParams:
    """Humid, cooler, frequent rainfall."""
    return RegionParams(
        name="B_humid",
        temp_mean=22.0,
        temp_std=2.0,
        humidity_base=65.0,
        humidity_variance=5.0,
        rainfall_shape=1.2,
        rainfall_scale=2.5,
        rainfall_spike_chance=0.08,
        rainfall_spike_scale=12.0,
        evap_temp_coeff=0.08,
        evap_humidity_coeff=0.025,
        decay_rate=0.03,
        infiltration_eff=0.65,
    )


def get_region_registry() -> dict[str, RegionParams]:
    """Registry for all default synthetic regions."""
    return {"A": get_region_a_params(), "B": get_region_b_params()}
