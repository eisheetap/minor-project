"""Entry point for research-grade domain-shift pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment_runner.run_full_experiment import run_pipeline
from benchmark.benchmark_runner import run_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full cross-region + transfer experiment.")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip running the fast benchmarks.")
    args = parser.parse_args()
    run_pipeline(Path(args.config))
    if not args.skip_benchmarks:
        run_benchmarks(Path(args.config))


if __name__ == "__main__":
    main()
