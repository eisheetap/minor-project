"""Lightweight benchmark runner that executes a fast config and writes outputs to benchmark_output."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import yaml

from experiment_runner.run_full_experiment import run_pipeline
from experiment_runner.utils import ensure_output_dirs, load_config


def _write_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f)


def build_benchmark_config(base_config_path: Path, benchmark_root: Path) -> Path:
    cfg = load_config(base_config_path)

    # Override for a fast, deterministic smoke benchmark
    cfg["run"]["run_name"] = "benchmark_run"
    cfg["run"]["output_dir"] = str(benchmark_root)
    cfg["run"]["log_dir"] = str(benchmark_root / "logs")
    cfg["run"]["metrics_dir"] = str(benchmark_root / "metrics")
    cfg["run"]["plots_dir"] = str(benchmark_root / "plots")
    cfg["data"]["synthetic"]["length"] = 800  # keep small for speed
    cfg["evaluation"]["runs"] = 1
    cfg["model"]["type"] = "linear"  # fastest option
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 128
    cfg["transfer"]["enabled"] = False  # skip fine-tuning in benchmark

    benchmark_cfg_path = benchmark_root / "benchmark_config.yaml"
    _write_config(cfg, benchmark_cfg_path)
    return benchmark_cfg_path


def run_benchmarks(base_config_path: Path = Path("configs/experiment_config.yaml")) -> Dict[str, Any]:
    benchmark_root = Path("benchmark_output")
    ensure_output_dirs(benchmark_root, benchmark_root / "metrics", benchmark_root / "plots")

    bench_cfg_path = build_benchmark_config(base_config_path, benchmark_root)
    start = time.perf_counter()
    run_pipeline(bench_cfg_path, transfer_override=False, min_runs_required=1)
    duration = time.perf_counter() - start

    summary = {
        "config": str(bench_cfg_path),
        "output_dir": str(benchmark_root),
        "duration_sec": duration,
    }
    (benchmark_root / "benchmark_summary.yaml").write_text(yaml.safe_dump(summary))
    return summary
