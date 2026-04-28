"""CLI wrapper that executes the pipeline with transfer learning forced on.

Purpose:
- Provides a convenient entrypoint for transfer-enabled experiments regardless of
  default config transfer toggles.

Inputs:
- CLI config path argument.

Outputs:
- Same artifact set as full runner, with transfer stage guaranteed to execute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment_runner.run_full_experiment import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    args = parser.parse_args()
    run_pipeline(Path(args.config), transfer_override=True)


if __name__ == "__main__":
    main()
