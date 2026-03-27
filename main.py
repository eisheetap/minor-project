"""Entry point for research-grade domain-shift pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment_runner.run_full_experiment import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full cross-region + transfer experiment.")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    run_pipeline(Path(args.config))


if __name__ == "__main__":
    main()
