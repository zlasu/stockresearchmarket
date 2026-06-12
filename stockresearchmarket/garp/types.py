from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FactorStatus:
    category: str
    factor: str
    status: str
    reason: str
    source: str


@dataclass
class GarpDataBundle:
    frames: dict[str, pd.DataFrame]
    close: pd.DataFrame
    fundamentals: pd.DataFrame
    estimates: pd.DataFrame
    sectors: dict[str, str]
    availability: list[FactorStatus]
    source_notes: list[str] = field(default_factory=list)


@dataclass
class UniverseResult:
    members: list[str]
    diagnostics: pd.DataFrame


@dataclass
class FactorResult:
    values: pd.DataFrame
    metadata: pd.DataFrame
    category_scores: pd.DataFrame | None = None


@dataclass
class ScoreResult:
    scores: pd.DataFrame
    category_weights: dict[str, float]
    availability: pd.DataFrame
    notes: list[str]


@dataclass
class PortfolioDecision:
    weights: pd.Series
    selected: list[str]
    turnover: float
    notes: list[str]


@dataclass
class GarpBacktestResult:
    experiment_id: str
    output_dir: Path
    config: dict[str, Any]
    equity: pd.Series
    returns: pd.Series
    benchmark_equity: pd.DataFrame
    benchmark_returns: pd.DataFrame
    weights: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    factor_values: pd.DataFrame
    scores: pd.DataFrame
    data_availability: pd.DataFrame
    metrics: dict[str, float | int | str]
    monthly_returns: pd.Series
    yearly_returns: pd.Series
    limitations: list[str]

