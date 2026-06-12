from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stockresearchmarket.garp.config import get_config, load_garp_config
from stockresearchmarket.garp.data_loader import load_garp_data
from stockresearchmarket.garp.factors import compute_factors
from stockresearchmarket.garp.metrics import garp_metrics, monthly_returns, yearly_returns
from stockresearchmarket.garp.plotting import write_garp_report
from stockresearchmarket.garp.portfolio import build_portfolio
from stockresearchmarket.garp.scoring import score_factors
from stockresearchmarket.garp.types import GarpBacktestResult
from stockresearchmarket.garp.universe import build_universe


def run_garp_backtest(
    experiment: str | Path | None = "001_baseline_garp",
    provider: str | None = None,
    years: int | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    output_root: Path = Path("experiments/garp"),
) -> GarpBacktestResult:
    config = load_garp_config(experiment)
    bundle = load_garp_data(config, provider=provider, years=years, start=start, end=end, refresh=refresh)
    experiment_id = _experiment_id(config)
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    close = bundle.close.sort_index().ffill()
    investable = [ticker for ticker in get_config(config, "universe.tickers", []) if ticker in close.columns]
    benchmarks = [ticker for ticker in get_config(config, "data.benchmark_tickers", ["SPY", "QQQ"]) if ticker in close.columns]
    rebalance_dates = _rebalance_dates(close.index, str(get_config(config, "rebalance.frequency", "ME")))
    weights = pd.DataFrame(index=close.index, columns=list(dict.fromkeys(investable + ["CASH"])), dtype="float64")
    previous_weights = pd.Series(dtype=float)
    holdings_rows = []
    trades_rows = []
    factor_rows = []
    score_rows = []
    availability_rows = []
    limitations = [note for note in bundle.source_notes if note]

    for rebalance_date in rebalance_dates:
        universe = build_universe(bundle, rebalance_date, config)
        if len(universe.members) < max(3, int(get_config(config, "portfolio.top_n", 20)) // 3):
            limitations.append(f"{rebalance_date.date()}: small universe ({len(universe.members)} members).")
        factors = compute_factors(bundle, universe.members, rebalance_date, config)
        scored = score_factors(factors, config)
        decision = build_portfolio(scored, bundle, rebalance_date, config, previous_weights)
        aligned_weights = decision.weights.reindex(weights.columns).fillna(0.0)
        weights.loc[rebalance_date] = aligned_weights
        trades_rows.append(
            {
                "date": rebalance_date,
                "turnover": decision.turnover,
                "selected": ",".join(decision.selected),
                "notes": " | ".join(decision.notes + scored.notes),
            }
        )
        for ticker, weight in aligned_weights[aligned_weights > 0].items():
            holdings_rows.append({"date": rebalance_date, "ticker": ticker, "weight": float(weight), "sector": bundle.sectors.get(ticker, "Cash")})
        _append_snapshot(factor_rows, factors.values, rebalance_date)
        _append_snapshot(score_rows, scored.scores, rebalance_date)
        availability = scored.availability.copy()
        availability["date"] = rebalance_date
        availability_rows.append(availability)
        previous_weights = aligned_weights

    weights = weights.ffill().fillna(0.0).astype(float)
    result = _simulate(config, close, weights, benchmarks, output_dir, bundle, holdings_rows, trades_rows, factor_rows, score_rows, availability_rows, limitations)
    write_garp_report(result)
    return result


def _simulate(
    config: dict[str, Any],
    close: pd.DataFrame,
    weights: pd.DataFrame,
    benchmarks: list[str],
    output_dir: Path,
    bundle: Any,
    holdings_rows: list[dict[str, Any]],
    trades_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    availability_rows: list[pd.DataFrame],
    limitations: list[str],
) -> GarpBacktestResult:
    annualization_days = int(get_config(config, "backtest.annualization_days", 252))
    risk_free_rate = float(get_config(config, "backtest.risk_free_rate", 0.0))
    initial_capital = float(get_config(config, "backtest.initial_capital", 10_000))
    cost_bps = float(get_config(config, "portfolio.transaction_cost_bps", 3.5))
    tax_bps = float(get_config(config, "portfolio.tax_drag_bps_per_turnover", 0.0))
    returns_panel = close.reindex(columns=weights.columns.intersection(close.columns)).pct_change().fillna(0.0)
    effective_weights = weights.shift(1).fillna(0.0)
    effective_asset_weights = effective_weights.reindex(columns=returns_panel.columns).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    gross_returns = (effective_asset_weights * returns_panel).sum(axis=1)
    cost_drag = turnover * ((cost_bps + tax_bps) / 10_000)
    returns = gross_returns - cost_drag
    equity = initial_capital * (1 + returns).cumprod()
    benchmark_returns = close[benchmarks].pct_change().fillna(0.0) if benchmarks else pd.DataFrame(index=close.index)
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()
    metrics = garp_metrics(returns, equity, benchmark_returns, turnover, annualization_days, risk_free_rate)
    metrics["transaction_cost_drag"] = float(cost_drag.sum())
    metrics["estimated_tax_drag"] = float((turnover * tax_bps / 10_000).sum())
    holdings = pd.DataFrame(holdings_rows)
    trades = pd.DataFrame(trades_rows)
    factor_values = pd.DataFrame(factor_rows)
    scores = pd.DataFrame(score_rows)
    data_availability = pd.concat(availability_rows, ignore_index=True) if availability_rows else pd.DataFrame()
    result = GarpBacktestResult(
        experiment_id=output_dir.name,
        output_dir=output_dir,
        config=config,
        equity=equity.rename("strategy"),
        returns=returns.rename("strategy_return"),
        benchmark_equity=benchmark_equity,
        benchmark_returns=benchmark_returns,
        weights=weights,
        holdings=holdings,
        trades=trades,
        factor_values=factor_values,
        scores=scores,
        data_availability=data_availability,
        metrics=metrics,
        monthly_returns=monthly_returns(returns),
        yearly_returns=yearly_returns(returns),
        limitations=list(dict.fromkeys(limitations)),
    )
    _write_outputs(result)
    return result


def _write_outputs(result: GarpBacktestResult) -> None:
    result.equity.to_csv(result.output_dir / "equity_curve.csv")
    result.returns.to_csv(result.output_dir / "daily_returns.csv")
    result.benchmark_equity.to_csv(result.output_dir / "benchmark_equity.csv")
    result.weights.to_csv(result.output_dir / "weights.csv")
    result.holdings.to_csv(result.output_dir / "holdings_history.csv", index=False)
    result.trades.to_csv(result.output_dir / "trades.csv", index=False)
    result.factor_values.to_csv(result.output_dir / "factor_values.csv", index=False)
    result.scores.to_csv(result.output_dir / "scores.csv", index=False)
    result.data_availability.to_csv(result.output_dir / "data_availability.csv", index=False)
    result.monthly_returns.to_csv(result.output_dir / "monthly_returns.csv")
    result.yearly_returns.to_csv(result.output_dir / "yearly_returns.csv")
    (result.output_dir / "summary.json").write_text(json.dumps(result.metrics, indent=2, default=str), encoding="utf-8")
    with (result.output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result.config, handle, sort_keys=False, allow_unicode=True)


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    if frequency == "quarterly":
        frequency = "QE"
    if frequency == "monthly":
        frequency = "ME"
    dates = pd.Series(index=index, data=1).resample(frequency).last().dropna().index
    return [index[index.searchsorted(date, side="right") - 1] for date in dates if index.searchsorted(date, side="right") > 0]


def _append_snapshot(rows: list[dict[str, Any]], frame: pd.DataFrame, date: pd.Timestamp) -> None:
    if frame.empty:
        return
    snapshot = frame.copy()
    snapshot["date"] = date
    snapshot["ticker"] = snapshot.index
    rows.extend(snapshot.reset_index(drop=True).to_dict("records"))


def _experiment_id(config: dict[str, Any]) -> str:
    experiment = get_config(config, "experiment.id", "garp")
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{stamp}_{experiment}"
