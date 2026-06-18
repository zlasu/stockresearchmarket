from __future__ import annotations

from statistics import mean
from typing import Any

import pandas as pd

from stockresearchmarket.ui.models import VariantRecord


def score_variants(variants: list[VariantRecord]) -> None:
    if not variants:
        return
    if any(_num(variant.metrics.get("research_score")) is not None for variant in variants):
        for variant in variants:
            variant.computed_score = _num(variant.metrics.get("research_score"))
        return

    rows = []
    for variant in variants:
        rows.append(
            {
                "id": variant.id,
                "sharpe": _num(variant.metrics.get("sharpe")),
                "calmar": _num(variant.metrics.get("calmar")),
                "cagr": _num(variant.metrics.get("cagr")),
                "dd": _num(variant.metrics.get("drawdown_magnitude")),
                "turnover": _num(variant.metrics.get("avg_annual_turnover")),
            }
        )
    frame = pd.DataFrame(rows).set_index("id")
    if frame.empty:
        return

    score = pd.Series(0.0, index=frame.index)
    weights = {"sharpe": 0.35, "calmar": 0.25, "cagr": 0.20}
    for column, weight in weights.items():
        values = frame[column].astype(float)
        if values.notna().any():
            score += values.rank(pct=True).fillna(0.0) * weight

    dd = frame["dd"].astype(float)
    if dd.notna().any():
        score += (1 - dd.rank(pct=True)).fillna(0.0) * 0.10

    turnover = frame["turnover"].astype(float)
    if turnover.notna().any():
        score += (1 - turnover.rank(pct=True)).fillna(0.0) * 0.10

    for variant in variants:
        variant.computed_score = float(score.get(variant.id, 0.0))


def build_recommendations(variants: list[VariantRecord]) -> list[dict[str, Any]]:
    score_variants(variants)
    scored = [variant for variant in variants if variant.computed_score is not None]
    if not scored:
        return []

    top = max(scored, key=lambda variant: float(variant.computed_score or 0.0))
    recommendations = [
        {
            "title": "Research leader",
            "variant_id": top.id,
            "variant_name": top.name,
            "score": top.computed_score,
            "body": "Highest deterministic research score in this experiment. Treat it as a shortlist signal, not an investment recommendation.",
        }
    ]

    risk_candidates = [variant for variant in variants if _num(variant.metrics.get("drawdown_magnitude")) is not None]
    if risk_candidates:
        best_risk = min(risk_candidates, key=lambda variant: float(variant.metrics.get("drawdown_magnitude") or 0.0))
        if best_risk.id != top.id:
            recommendations.append(
                {
                    "title": "Risk-control candidate",
                    "variant_id": best_risk.id,
                    "variant_name": best_risk.name,
                    "score": best_risk.computed_score,
                    "body": "Lowest observed max drawdown magnitude among comparable variants. Review returns and turnover before preferring it.",
                }
            )

    turnover_values = [_num(variant.metrics.get("avg_annual_turnover")) for variant in variants]
    turnover_values = [float(value) for value in turnover_values if value is not None]
    if turnover_values:
        avg_turnover = mean(turnover_values)
        high_turnover = [
            variant
            for variant in variants
            if _num(variant.metrics.get("avg_annual_turnover")) is not None
            and float(variant.metrics["avg_annual_turnover"]) > avg_turnover * 1.5
        ]
        if high_turnover:
            watch = max(high_turnover, key=lambda variant: float(variant.computed_score or 0.0))
            recommendations.append(
                {
                    "title": "Turnover watch",
                    "variant_id": watch.id,
                    "variant_name": watch.name,
                    "score": watch.computed_score,
                    "body": "Promising score with elevated turnover. Validate costs, taxes, and slippage before treating it as robust.",
                }
            )

    return recommendations


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

