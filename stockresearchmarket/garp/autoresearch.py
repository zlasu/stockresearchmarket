from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stockresearchmarket.garp.backtest import run_garp_backtest
from stockresearchmarket.garp.config import deep_merge, get_config, load_garp_config


def run_autoresearch(
    base_experiment: str = "001_baseline_garp",
    provider: str | None = "synthetic",
    years: int | None = 8,
    max_experiments: int | None = None,
    output_root: Path = Path("experiments/garp_autoresearch"),
) -> pd.DataFrame:
    base_config = load_garp_config(base_experiment)
    max_runs = int(max_experiments or get_config(base_config, "autoresearch.max_experiments", 12))
    output_root.mkdir(parents=True, exist_ok=True)
    leaderboard_rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(generate_variants(base_config), start=1):
        if idx > max_runs:
            break
        variant_id = f"auto_{idx:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        variant_path = output_root / f"{variant_id}.yaml"
        experiment_payload = {
            "id": variant_id,
            "name": f"Autoresearch {idx:03d}",
            "description": "Generated GARP parameter variant.",
            "overrides": variant,
        }
        variant_path.write_text(yaml.safe_dump(experiment_payload, sort_keys=False), encoding="utf-8")
        result = run_garp_backtest(variant_path, provider=provider, years=years, output_root=output_root / "runs")
        row = {"experiment_id": result.experiment_id, "config_path": str(variant_path), "output_dir": str(result.output_dir), **result.metrics}
        leaderboard_rows.append(row)
    leaderboard = rank_experiments(pd.DataFrame(leaderboard_rows))
    leaderboard.to_csv(output_root / "leaderboard.csv", index=False)
    (output_root / "leaderboard.json").write_text(json.dumps(leaderboard.to_dict("records"), indent=2, default=str), encoding="utf-8")
    return leaderboard


def generate_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = get_config(config, "autoresearch.parameter_grid", {})
    variants: list[dict[str, Any]] = []
    base_weights = deepcopy(get_config(config, "factors.base_weights", {}))
    for top_n in grid.get("top_n", [20]):
        variants.append({"portfolio": {"top_n": top_n}})
    for frequency in grid.get("rebalance_frequency", ["ME"]):
        variants.append({"rebalance": {"frequency": frequency}})
    for sma_filter in grid.get("sma_filter", [False, True]):
        variants.append({"risk_management": {"market_filter": {"enabled": bool(sma_filter)}}})
    for target in grid.get("volatility_target", [0.0, 0.15]):
        variants.append({"risk_management": {"volatility_target": {"enabled": target > 0, "target_volatility": target or 0.15}}})
    for hold_rank in grid.get("turnover_band", [0, 30]):
        if hold_rank:
            variants.append({"portfolio": {"hold_until_rank": hold_rank, "turnover_threshold": 0.10}})
    for momentum_weight in grid.get("momentum_weight", [base_weights.get("momentum", 0.2)]):
        weights = deepcopy(base_weights)
        weights["momentum"] = momentum_weight
        variants.append({"factors": {"base_weights": _renormalize(weights)}})
    return [deep_merge({}, variant) for variant in variants]


def rank_experiments(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    ranked = results.copy()
    ranked["cagr_rank"] = ranked["cagr"].rank(pct=True)
    ranked["max_drawdown_penalty"] = ranked["max_drawdown"].abs().rank(pct=True)
    ranked["turnover_penalty"] = ranked["avg_annual_turnover"].rank(pct=True)
    ranked["research_score"] = (
        0.35 * ranked["sharpe"].fillna(0)
        + 0.25 * ranked["calmar"].fillna(0)
        + 0.20 * ranked["cagr_rank"].fillna(0)
        - 0.10 * ranked["max_drawdown_penalty"].fillna(0)
        - 0.10 * ranked["turnover_penalty"].fillna(0)
    )
    return ranked.sort_values("research_score", ascending=False).reset_index(drop=True)


def load_leaderboard(path: Path = Path("experiments/garp_autoresearch/leaderboard.csv")) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _renormalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        return weights
    return {key: max(value, 0.0) / total for key, value in weights.items()}

