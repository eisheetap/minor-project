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
    """Semi-arid, hotter, sparse rainfall; soil moisture stays within 20-100."""
    return RegionParams(
        name="A_semi_arid",
        temp_mean=30.0,
        temp_std=2.5,
        humidity_base=30.0,
        humidity_variance=2.5,
        rainfall_shape=1.0,
        rainfall_scale=1.2,
        rainfall_spike_chance=0.03,
        rainfall_spike_scale=5.0,
        evap_temp_coeff=0.10,
        evap_humidity_coeff=0.028,
        decay_rate=0.05,
        infiltration_eff=0.5,
    )


def get_region_b_params() -> RegionParams:
    """Humid, cooler, frequent rainfall; soil moisture stays within 20-100."""
    return RegionParams(
        name="B_humid",
        temp_mean=24.0,
        temp_std=2.0,
        humidity_base=60.0,
        humidity_variance=4.0,
        rainfall_shape=1.4,
        rainfall_scale=2.2,
        rainfall_spike_chance=0.07,
        rainfall_spike_scale=10.0,
        evap_temp_coeff=0.07,
        evap_humidity_coeff=0.022,
        decay_rate=0.035,
        infiltration_eff=0.6,
    )


def get_region_registry() -> dict[str, RegionParams]:
    """Registry for all default synthetic regions."""
    return {"A": get_region_a_params(), "B": get_region_b_params()}
