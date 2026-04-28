"""Data ingestion/generation layer for region-wise pipeline inputs.

Purpose:
- Generates synthetic agro-climatic time-series for multiple regions or loads
  external region files with the required schema.

Inputs:
- Region parameter definitions, random seeds, sequence length, and optional external
  CSV/JSON paths defined by configuration.

Outputs:
- In-memory pandas DataFrames for regions (typically keys ``A`` and ``B``) containing
  feature columns and next-day target.
- Optional persisted region CSV/JSON files for traceability/debugging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from data_engine.region_simulator import RegionParams, get_region_registry

GLOBAL_SEED = 123
FEATURE_COLUMNS = ["temperature", "humidity", "rainfall", "evapotranspiration", "soil_moisture"]
TARGET_COLUMN = "next_day_soil_moisture"
SOIL_MIN = 20.0
SOIL_MAX = 100.0


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

    temperature = region.temp_mean + seasonal_temp + rng.normal(0, region.temp_std, size=length)
    humidity = region.humidity_base + 0.2 * (100 - temperature) + rng.normal(0, region.humidity_variance, size=length)
    humidity = np.clip(humidity, 5, 100)

    rainfall = _generate_rainfall(rng, region, length)

    evapotranspiration = (
        region.evap_temp_coeff * temperature
        - region.evap_humidity_coeff * humidity
        + rng.normal(0, 0.2, size=length)
    )
    evapotranspiration = np.clip(evapotranspiration, 0, None)

    soil_moisture = np.zeros(length)
    soil_moisture[0] = rng.uniform(SOIL_MIN, (SOIL_MIN + SOIL_MAX) / 2)
    for t in range(1, length):
        recharge = rainfall[t - 1] * region.infiltration_eff
        loss = evapotranspiration[t - 1] + region.decay_rate * soil_moisture[t - 1]
        noise = rng.normal(0, 0.8)
        soil_moisture[t] = soil_moisture[t - 1] + recharge - loss + noise
        soil_moisture[t] = np.clip(soil_moisture[t], SOIL_MIN, SOIL_MAX)

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


def generate_regions(
    length: int = 10_000,
    seed: int = GLOBAL_SEED,
    region_seed_offset: int = 1,
    custom_params: Optional[Mapping[str, RegionParams]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate multiple regions with distinct statistics."""
    registry = dict(get_region_registry())
    if custom_params:
        registry.update(custom_params)

    datasets: Dict[str, pd.DataFrame] = {}
    for idx, (key, params) in enumerate(registry.items()):
        datasets[key] = generate_region_data(params, length=length, seed=seed + idx * region_seed_offset)
    return datasets


def save_datasets(datasets: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df in datasets.items():
        df.to_csv(output_dir / f"region_{key}.csv", index=False)
        df.to_json(output_dir / f"region_{key}.json", orient="records", date_format="iso")


def load_region_from_file(path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load a region dataset from CSV or JSON with the expected schema."""
    if not path.exists():
        raise FileNotFoundError(f"Region file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=[timestamp_col])
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path, convert_dates=[timestamp_col])
    else:
        raise ValueError(f"Unsupported file format for region data: {path.suffix}")
    if timestamp_col != "timestamp":
        df = df.rename(columns={timestamp_col: "timestamp"})
    required_cols: Iterable[str] = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Region data missing columns: {missing}")
    return df.reset_index(drop=True)


def load_regions_from_config(
    source: str,
    synthetic_length: int,
    seed: int,
    region_seed_offset: int,
    external_a: Optional[str],
    external_b: Optional[str],
    timestamp_col: str,
) -> Dict[str, pd.DataFrame]:
    """Load regions per config: synthetic or external."""
    if source == "synthetic":
        return generate_regions(length=synthetic_length, seed=seed, region_seed_offset=region_seed_offset)
    if source in {"csv", "json"}:
        if not external_a or not external_b:
            raise ValueError("External region paths must be provided when source is csv/json.")
        return {
            "A": load_region_from_file(Path(external_a), timestamp_col=timestamp_col),
            "B": load_region_from_file(Path(external_b), timestamp_col=timestamp_col),
        }
    raise ValueError(f"Unknown data source: {source}")
