from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_METRICS = [
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "trades",
    "win_rate",
    "profit_factor",
    "avg_trade_return",
    "avg_annual_turnover",
    "alpha_total_return",
    "benchmark_total_return",
    "capital",
    "drawdown_magnitude",
    "research_score",
]

PERCENT_FIELDS = {
    "return_pct": "total_return",
    "cagr_pct": "cagr",
    "max_drawdown_pct": "max_drawdown",
    "win_rate_pct": "win_rate",
    "alpha_pct_points": "alpha_total_return",
    "benchmark_return_pct": "benchmark_total_return",
}

DIRECT_ALIASES = {
    "ending_equity": "capital",
    "daily_sharpe": "sharpe",
    "closed_trades": "trades",
    "winning_trades": "winning_trades",
}


def stable_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "item"


def experiment_id_for(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    return stable_id("__".join(rel.parts))


def to_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return to_number(value.item())
        except Exception:
            return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return None
    text = text.replace(",", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if percent:
        number /= 100
    if number.is_integer() and not percent:
        return int(number)
    return number


def normalize_metrics(raw: dict[str, Any], *, equity_last: float | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in CANONICAL_METRICS:
        value = to_number(raw.get(key))
        if value is not None:
            metrics[key] = value

    for source, target in DIRECT_ALIASES.items():
        if target not in metrics:
            value = to_number(raw.get(source))
            if value is not None:
                metrics[target] = value

    for source, target in PERCENT_FIELDS.items():
        if target in metrics:
            continue
        value = to_number(raw.get(source))
        if value is None:
            continue
        metrics[target] = value / 100 if abs(float(value)) > 1 else value

    if "max_drawdown" in metrics:
        max_dd = float(metrics["max_drawdown"])
        if max_dd > 0:
            metrics["max_drawdown"] = -max_dd
        metrics["drawdown_magnitude"] = abs(float(metrics["max_drawdown"]))

    if "capital" not in metrics:
        if equity_last is not None:
            metrics["capital"] = equity_last
        elif "ending_equity" in raw:
            metrics["capital"] = raw["ending_equity"]
        elif "total_return" in metrics:
            metrics["capital"] = 1 + float(metrics["total_return"])

    if "trades" in metrics and isinstance(metrics["trades"], float):
        metrics["trades"] = int(metrics["trades"])

    return metrics


def params_from_row(row: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in row.items():
        text = str(key)
        if text.startswith("param_"):
            params[text.removeprefix("param_")] = value
    if isinstance(row.get("params"), dict):
        params.update(row["params"])
    elif isinstance(row.get("variant"), dict):
        params.update(row["variant"])
    return params


def first_numeric_last(path: Path, preferred_column: str | None = None) -> float | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None
    columns = [preferred_column] if preferred_column and preferred_column in frame.columns else []
    columns += [column for column in frame.columns if column not in columns]
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1])
    return None


def infer_created_at(path: Path) -> str | None:
    text = path.name
    match = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})[_-](\d{2})(\d{2})(\d{2})", text)
    if match:
        y, m, d, hh, mm, ss = match.groups()
        return f"{y}-{m}-{d}T{hh}:{mm}:{ss}"
    match = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", text)
    if match:
        y, m, d = match.groups()
        return f"{y}-{m}-{d}"
    try:
        return pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()
    except OSError:
        return None

