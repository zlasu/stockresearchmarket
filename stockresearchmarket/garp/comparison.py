from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def compare_garp_runs(root: Path = Path("experiments/garp"), output_dir: Path = Path("reports")) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        config_path = summary_path.parent / "resolved_config.yaml"
        config = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        rows.append(
            {
                "experiment_id": summary_path.parent.name,
                "name": config.get("experiment", {}).get("name", summary_path.parent.name),
                "output_dir": str(summary_path.parent),
                **metrics,
            }
        )
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values(["sharpe", "calmar", "cagr"], ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "garp_experiment_comparison.csv", index=False)
    (output_dir / "garp_experiment_comparison.md").write_text(_markdown_table(comparison), encoding="utf-8")
    return comparison


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No GARP runs found.\n"
    columns = [
        "experiment_id",
        "name",
        "total_return",
        "benchmark_total_return",
        "alpha_total_return",
        "cagr",
        "sharpe",
        "calmar",
        "max_drawdown",
        "avg_annual_turnover",
        "output_dir",
    ]
    existing = [column for column in columns if column in frame.columns]
    lines = ["| " + " | ".join(existing) + " |", "| " + " | ".join(["---"] * len(existing)) + " |"]
    for _, row in frame[existing].iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"

