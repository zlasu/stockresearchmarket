from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockresearchmarket.garp.config import get_config
from stockresearchmarket.garp.types import FactorResult, ScoreResult


def score_factors(factors: FactorResult, config: dict[str, Any]) -> ScoreResult:
    if factors.values.empty:
        return ScoreResult(pd.DataFrame(), {}, factors.metadata, ["No factor rows were available."])
    include_unsafe = bool(get_config(config, "factors.include_unsafe", False))
    lower = float(get_config(config, "factors.winsorize_lower", 0.05))
    upper = float(get_config(config, "factors.winsorize_upper", 0.95))
    base_weights = dict(get_config(config, "factors.base_weights", {}))
    usable_meta = factors.metadata[factors.metadata["status"].eq("available") | (include_unsafe & factors.metadata["status"].eq("unsafe"))]
    usable_factors = [factor for factor in usable_meta["factor"].tolist() if factor in factors.values.columns]
    notes: list[str] = []
    if not usable_factors:
        return ScoreResult(pd.DataFrame(index=factors.values.index), {}, factors.metadata, ["No safe factors were available."])

    ranked = pd.DataFrame(index=factors.values.index)
    for factor in usable_factors:
        values = pd.to_numeric(factors.values[factor], errors="coerce")
        if values.notna().sum() < 3:
            notes.append(f"Skipped {factor}: fewer than 3 non-null cross-sectional values.")
            continue
        clipped = values.clip(values.quantile(lower), values.quantile(upper))
        ranked[factor] = clipped.rank(pct=True)

    category_scores = pd.DataFrame(index=factors.values.index)
    category_availability: dict[str, list[str]] = {}
    for category in sorted(set(usable_meta["category"])):
        cols = [factor for factor in usable_meta.loc[usable_meta["category"].eq(category), "factor"] if factor in ranked.columns]
        if cols:
            category_scores[category] = ranked[cols].mean(axis=1)
            category_availability[category] = cols
    category_scores["risk_pass"] = factors.values.get("risk_pass", pd.Series(1.0, index=factors.values.index)).fillna(0.0)
    category_weights = _available_weights(base_weights, category_scores.columns, notes)
    total = pd.Series(0.0, index=factors.values.index)
    for category, weight in category_weights.items():
        total = total.add(category_scores[category].fillna(category_scores[category].median()) * weight, fill_value=0)
    scores = category_scores.copy()
    scores["total_score"] = total.where(category_scores["risk_pass"] > 0, np.nan)
    scores["rank"] = scores["total_score"].rank(ascending=False, method="first")
    factors.category_scores = category_scores
    return ScoreResult(scores=scores.sort_values("rank"), category_weights=category_weights, availability=factors.metadata, notes=notes)


def _available_weights(base_weights: dict[str, float], available_columns: pd.Index, notes: list[str]) -> dict[str, float]:
    available = {category: float(weight) for category, weight in base_weights.items() if category in available_columns}
    missing = [category for category in base_weights if category not in available_columns]
    if missing:
        notes.append(f"Redistributed missing category weights: {', '.join(missing)}.")
    total = sum(available.values())
    if total <= 0:
        return {}
    return {category: weight / total for category, weight in available.items()}

