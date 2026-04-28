"""Shared runner utilities for configuration, logging, and output setup.

Purpose:
- Provides reusable helpers that keep runner scripts minimal and consistent.

Inputs:
- Config file paths, run names, and directory paths from runner modules.

Outputs:
- Parsed config dictionaries, initialized logging targets, and created output folders.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def setup_logging(run_name: str, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def ensure_output_dirs(output_dir: Path, metrics_dir: Path, plots_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
