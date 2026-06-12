from __future__ import annotations

import pandas as pd

from stockresearchmarket.garp.backtest import run_garp_backtest
from stockresearchmarket.garp.config import load_garp_config
from stockresearchmarket.garp.data_loader import latest_snapshot, load_garp_data
from stockresearchmarket.garp.factors import compute_factors
from stockresearchmarket.garp.portfolio import build_portfolio
from stockresearchmarket.garp.scoring import score_factors
from stockresearchmarket.garp.types import FactorResult
from stockresearchmarket.garp.universe import build_universe


def test_latest_snapshot_prevents_lookahead() -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "AAA", "as_of_date": "2020-01-15", "market_cap": 10},
            {"ticker": "AAA", "as_of_date": "2020-03-15", "market_cap": 99},
        ]
    )
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    snapshot = latest_snapshot(frame, pd.Timestamp("2020-02-01"))
    assert snapshot.loc["AAA", "market_cap"] == 10


def test_missing_revisions_weight_is_redistributed() -> None:
    values = pd.DataFrame(
        {
            "growth_revenue_growth_3y": [0.1, 0.2, 0.3],
            "quality_roe": [0.2, 0.3, 0.4],
        },
        index=["A", "B", "C"],
    )
    metadata = pd.DataFrame(
        [
            {"category": "growth", "factor": "growth_revenue_growth_3y", "status": "available"},
            {"category": "quality", "factor": "quality_roe", "status": "available"},
            {"category": "revisions", "factor": "revisions_eps_revision_90d", "status": "unavailable"},
        ]
    )
    config = load_garp_config("001_baseline_garp")
    scored = score_factors(FactorResult(values=values, metadata=metadata), config)
    assert scored.category_weights["growth"] > 0
    assert "revisions" not in scored.category_weights
    assert round(sum(scored.category_weights.values()), 10) == 1.0


def test_portfolio_weights_sum_to_one_or_less() -> None:
    config = load_garp_config("001_baseline_garp")
    bundle = load_garp_data(config, provider="synthetic", years=4, refresh=True)
    as_of = bundle.close.index[-1]
    universe = build_universe(bundle, as_of, config)
    factors = compute_factors(bundle, universe.members, as_of, config)
    scored = score_factors(factors, config)
    decision = build_portfolio(scored, bundle, as_of, config)
    assert decision.weights.sum() <= 1.0000001
    assert decision.weights.sum() > 0


def test_garp_backtest_smoke_outputs_artifacts(tmp_path) -> None:
    result = run_garp_backtest("001_baseline_garp", provider="synthetic", years=4, refresh=True, output_root=tmp_path)
    assert not result.equity.empty
    assert (result.output_dir / "report.html").exists()
    assert (result.output_dir / "data_availability.csv").exists()
    assert "benchmark_total_return" in result.metrics

