"""Markdown report writer that consolidates experiment outcomes.

Purpose:
- Converts cross-region, transfer, statistical, and robustness outputs into a
  single human-readable markdown report.

Inputs:
- Run metadata plus DataFrame/dictionary summaries produced during evaluation.

Outputs:
- Markdown report file saved to configured output path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def write_markdown_report(
    path: Path,
    run_name: str,
    cross_region_table: pd.DataFrame,
    transfer_metrics: Optional[Dict[str, float]] = None,
    stats_summary: Optional[Dict[str, float]] = None,
    robustness_summary: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Experiment Report — {run_name}", ""]
    lines.append("## Cross-Region Performance")
    lines.append(cross_region_table.to_markdown(index=False))
    lines.append("")

    if transfer_metrics:
        lines.append("## Transfer Learning (Target Region)")
        lines.append(pd.DataFrame([transfer_metrics]).to_markdown(index=False))
        lines.append("")

    if stats_summary:
        lines.append("## Statistical Tests")
        lines.append(pd.DataFrame([stats_summary]).to_markdown(index=False))
        lines.append("")

    if robustness_summary:
        lines.append("## Robustness Checks")
        robustness_rows = []
        for key, metrics in robustness_summary.items():
            row = {"scenario": key, **metrics}
            robustness_rows.append(row)
        lines.append(pd.DataFrame(robustness_rows).to_markdown(index=False))

    path.write_text("\n".join(lines))
