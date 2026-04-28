"""Top-level CLI entry for the full domain-shift experiment pipeline.

Purpose:
- Launches the end-to-end workflow (data -> preprocessing -> training -> evaluation -> reporting).

Inputs:
- CLI argument ``--config`` pointing to YAML configuration.
- Optional ``--skip-benchmarks`` flag.

Outputs:
- Delegates creation of metrics, plots, logs, and reports under configured output folders.
- Optionally triggers benchmark artifacts under benchmark output paths.
"""
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
