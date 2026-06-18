from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stockresearchmarket.ui.models import ExperimentRecord, VariantRecord
from stockresearchmarket.ui.serialization import clean_value

TABLE_ALIASES = {
    "positions": "position",
    "position": "position",
    "holdings": "holdings",
    "trades": "trades",
    "weights": "weights",
    "scores": "scores",
    "data_quality": "data_quality",
}


def load_table(path: Path, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    total = len(frame)
    preview = frame.iloc[max(offset, 0) : max(offset, 0) + max(limit, 1)]
    return {
        "columns": [{"key": str(column), "label": str(column)} for column in preview.columns],
        "rows": clean_value(preview.to_dict("records")),
        "row_count": total,
        "offset": offset,
        "limit": limit,
        "truncated": offset + limit < total,
    }


def load_variant_series(experiment: ExperimentRecord, variant: VariantRecord, kind: str) -> list[dict[str, Any]]:
    path = variant.series_paths.get(kind) or experiment.series_paths.get(kind)
    if path is None or not path.exists():
        if kind == "drawdown":
            equity = load_variant_series(experiment, variant, "equity")
            return _drawdown_from_equity(equity)
        return []
    column = variant.series_columns.get(kind)
    return _read_series(path, column=column, kind=kind)


def compare_variants(experiment: ExperimentRecord, variants: list[VariantRecord]) -> dict[str, Any]:
    series: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for kind in ["equity", "drawdown", "turnover", "monthly_returns", "yearly_returns"]:
        series[kind] = {variant.id: load_variant_series(experiment, variant, kind) for variant in variants}
    return {
        "metrics": [
            {
                "id": variant.id,
                "name": variant.name,
                "metrics": variant.metrics,
                "params": variant.params,
                "role": variant.role,
                "computed_score": variant.computed_score,
            }
            for variant in variants
        ],
        "series": series,
    }


def _read_series(path: Path, *, column: str | None, kind: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    date_column = _date_column(frame)
    value_column = _value_column(frame, column=column, kind=kind, date_column=date_column)
    if date_column is None or value_column is None:
        return []
    dates = frame[date_column]
    values = pd.to_numeric(frame[value_column], errors="coerce")
    data = pd.DataFrame({"date": dates.astype(str), "value": values}).dropna(subset=["value"])
    if len(data) > 2500:
        step = max(len(data) // 2500, 1)
        data = data.iloc[::step]
    return clean_value(data.to_dict("records"))


def _date_column(frame: pd.DataFrame) -> str | None:
    candidates = ["date", "Date", "timestamp", "Timestamp", "index", "Unnamed: 0"]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0]) if len(frame.columns) >= 2 else None


def _value_column(frame: pd.DataFrame, *, column: str | None, kind: str, date_column: str | None) -> str | None:
    if column and column in frame.columns:
        return column
    preferred = {
        "equity": ["equity", "strategy", "portfolio"],
        "drawdown": ["drawdown", "strategy"],
        "turnover": ["turnover"],
        "monthly_returns": ["return", "0", "strategy"],
        "yearly_returns": ["return", "0", "strategy"],
    }.get(kind, [])
    for candidate in preferred:
        if candidate in frame.columns:
            return candidate
    for candidate in frame.columns:
        if candidate == date_column:
            continue
        if pd.to_numeric(frame[candidate], errors="coerce").notna().any():
            return str(candidate)
    return None


def _drawdown_from_equity(equity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not equity:
        return []
    values = pd.Series([row["value"] for row in equity], dtype=float)
    drawdown = values / values.cummax() - 1
    return [{"date": row["date"], "value": float(value)} for row, value in zip(equity, drawdown, strict=True)]

