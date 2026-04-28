"""CLI wrapper that executes baseline cross-region evaluation only.

Purpose:
- Runs the main pipeline while explicitly disabling transfer adaptation to establish
  a pure cross-domain baseline.

Inputs:
- CLI config path argument.

Outputs:
- Baseline metrics/plots/report artifacts without fine-tuning outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment_runner.run_full_experiment import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    run_pipeline(Path(args.config), transfer_override=False)


if __name__ == "__main__":
    main()
