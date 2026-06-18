from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VariantRecord:
    id: str
    name: str
    path: Path
    kind: str
    metrics: dict[str, Any]
    original_metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    role: str | None = None
    rank: int | None = None
    source: str | None = None
    table_paths: dict[str, Path] = field(default_factory=dict)
    series_paths: dict[str, Path] = field(default_factory=dict)
    series_columns: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    computed_score: float | None = None


@dataclass
class ExperimentRecord:
    id: str
    name: str
    path: Path
    kind: str
    created_at: str | None = None
    variants: list[VariantRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    table_paths: dict[str, Path] = field(default_factory=dict)
    series_paths: dict[str, Path] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    best_variant_id: str | None = None

