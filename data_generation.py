"""
Synthetic agro-climatic data generation for two contrasting regions.

Region A (semi-arid): higher temperature mean, low rainfall variance, faster soil moisture decay.
Region B (humid): lower temperature mean, frequent rainfall spikes, slower soil moisture decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

GLOBAL_SEED = 123


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


def _seasonal_signal(length: int, amplitude: float, phase: float = 0.0) -> np.ndarray:
    """Yearly sinusoidal signal to mimic seasonality."""
    days = np.arange(length)
    return amplitude * np.sin(2 * np.pi * days / 365 + phase)


def _generate_rainfall(rng: np.random.Generator, params: RegionParams, length: int) -> np.ndarray:
    """Gamma base rainfall with occasional spikes for stormy days."""
    base = rng.gamma(shape=params.rainfall_shape, scale=params.rainfall_scale, size=length)
    spikes = rng.uniform(0, 1, size=length) < params.rainfall_spike_chance
    spike_amounts = spikes * rng.exponential(scale=params.rainfall_spike_scale, size=length)
    rainfall = base + spike_amounts
    return rainfall


def generate_region_data(region: RegionParams, length: int = 10_000, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """Generate a time series with features and next-day soil moisture target."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start="2000-01-01", periods=length, freq="D")
    seasonal_temp = _seasonal_signal(length, amplitude=8.0, phase=0.0)
    temperature = (
        region.temp_mean
        + seasonal_temp
        + rng.normal(0, region.temp_std, size=length)
    )

    humidity = (
        region.humidity_base
        + 0.2 * (100 - temperature)
        + rng.normal(0, region.humidity_variance, size=length)
    )
    humidity = np.clip(humidity, 5, 100)

    rainfall = _generate_rainfall(rng, region, length)

    evapotranspiration = (
        region.evap_temp_coeff * temperature
        - region.evap_humidity_coeff * humidity
        + rng.normal(0, 0.2, size=length)
    )
    evapotranspiration = np.clip(evapotranspiration, 0, None)

    soil_moisture = np.zeros(length)
    soil_moisture[0] = rng.uniform(20, 50)
    for t in range(1, length):
        recharge = rainfall[t - 1] * region.infiltration_eff
        loss = evapotranspiration[t - 1] + region.decay_rate * soil_moisture[t - 1]
        soil_moisture[t] = soil_moisture[t - 1] + recharge - loss + rng.normal(0, 0.5)
        soil_moisture[t] = np.clip(soil_moisture[t], 0, 100)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "evapotranspiration": evapotranspiration,
            "soil_moisture": soil_moisture,
        }
    )
    df["next_day_soil_moisture"] = df["soil_moisture"].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)
    df["region"] = region.name
    return df


def generate_regions(length: int = 10_000, seed: int = GLOBAL_SEED) -> Dict[str, pd.DataFrame]:
    """Generate both regions with distinct statistics."""
    region_a = RegionParams(
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
    region_b = RegionParams(
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

    # Offset seeds to ensure different draws per region while remaining reproducible.
    data_a = generate_region_data(region_a, length=length, seed=seed)
    data_b = generate_region_data(region_b, length=length, seed=seed + 1)
    return {"A": data_a, "B": data_b}


def main() -> None:
    datasets = generate_regions()
    for key, df in datasets.items():
        print(f"Region {key}: shape={df.shape}, head=\\n{df.head()}")


if __name__ == "__main__":
    main()
