from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ml_hypothesis_suite import markdown_table, momentum_scores
from scripts.run_ml_ranker_walkforward import (
    BENCHMARK_TICKERS,
    _load_universe,
    _trim_run,
    add_alpha_columns,
    choose_eligible_tickers,
    data_quality,
    load_yfinance_close,
)
from stockresearchmarket.strategies.ml_ranker import (
    buy_hold_run,
    rebalance_dates,
    simulate_portfolio,
    summarize_runs,
)


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start).normalize() if args.start else end - pd.DateOffset(years=args.years)
    constituents = _load_universe(args, output_dir)
    security_map = constituents.set_index("yf_ticker")["security"].astype(str).to_dict()
    sector_map = constituents.set_index("yf_ticker")["sector"].astype(str).to_dict()
    benchmark_tickers = [ticker for ticker in BENCHMARK_TICKERS if ticker not in constituents["yf_ticker"].tolist()]
    tickers = sorted(set(constituents["yf_ticker"].tolist() + benchmark_tickers))
    close = load_yfinance_close(tickers, start=start, end=end, output_dir=output_dir, refresh=args.refresh)
    quality = data_quality(close, start=start, end=end)
    quality.to_csv(output_dir / "data_quality.csv", index=False)

    eligible = choose_eligible_tickers(
        quality,
        constituents["yf_ticker"].tolist(),
        min_years=args.min_years,
        max_missing_fraction=args.max_missing_fraction,
        max_tickers=args.max_tickers,
    )
    if len(eligible) < args.top_n:
        raise RuntimeError(f"Only {len(eligible)} eligible tickers; need at least {args.top_n}.")

    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns]].dropna(how="all").ffill(limit=5)
    scores = momentum_scores(eligible_close, lookback=args.lookback, skip=args.skip)
    weights, actions, holdings, rebalance_summary = build_hold_band_trace(
        eligible_close,
        scores,
        security_map,
        sector_map,
        top_n=args.top_n,
        hold_until_rank=args.hold_until_rank,
        rebalance=args.rebalance,
        focus_year=args.focus_year,
    )

    run = simulate_portfolio(eligible_close, weights, f"momentum_hold{args.hold_until_rank}", cost_bps=args.cost_bps)
    benchmark_runs = [buy_hold_run(benchmark_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in benchmark_close.columns]
    comparison_start = pd.Timestamp(f"{args.focus_year}-01-01")
    all_runs = [_trim_run(item, comparison_start) for item in [run, *benchmark_runs]]
    summary = summarize_runs(all_runs, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="SPY")
    daily_returns = pd.concat([item.returns for item in all_runs], axis=1).sort_index()
    equity = pd.concat([item.equity for item in all_runs], axis=1).sort_index()

    actions.to_csv(output_dir / f"momentum_hold{args.hold_until_rank}_{args.focus_year}_actions.csv", index=False)
    holdings.to_csv(output_dir / f"momentum_hold{args.hold_until_rank}_{args.focus_year}_holdings.csv", index=False)
    rebalance_summary.to_csv(output_dir / f"momentum_hold{args.hold_until_rank}_{args.focus_year}_rebalance_summary.csv", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    daily_returns.to_csv(output_dir / "daily_returns.csv")
    equity.to_csv(output_dir / "equity_curves.csv")
    weights.to_csv(output_dir / f"weights_momentum_hold{args.hold_until_rank}.csv")
    write_report(output_dir, actions, holdings, rebalance_summary, summary, args, len(constituents), len(eligible))
    write_summary(output_dir, actions, holdings, rebalance_summary, summary, args, len(constituents), len(eligible))
    write_metadata(output_dir, args, start, end, len(constituents), len(eligible))

    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))
    print(rebalance_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain momentum_hold120 holdings and transactions for a focus year.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--focus-year", type=int, default=2026)
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--min-years", type=float, default=7.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--hold-until-rank", type=int, default=120)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--skip", type=int, default=21)
    parser.add_argument("--rebalance", default="ME")
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/momentum_hold120_trade_log")
    return parser.parse_args()


def build_hold_band_trace(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    security_map: dict[str, str],
    sector_map: dict[str, str],
    *,
    top_n: int,
    hold_until_rank: int,
    rebalance: str,
    focus_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    current_holdings: list[str] = []
    action_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    focus_start = pd.Timestamp(f"{focus_year}-01-01")
    focus_end = pd.Timestamp(f"{focus_year}-12-31")
    last_pre_focus_snapshot: list[dict[str, Any]] = []

    for date in rebalance_dates(close.index, rebalance):
        date = pd.Timestamp(date)
        weights.loc[date] = 0.0
        score = scores.loc[date].dropna().sort_values(ascending=False)
        if score.empty:
            continue
        ranks = pd.Series(range(1, len(score) + 1), index=score.index, dtype="int64")
        previous = list(current_holdings)
        keep = [ticker for ticker in previous if ticker in ranks and int(ranks[ticker]) <= hold_until_rank]
        fill = [ticker for ticker in score.index if ticker not in keep][: max(0, top_n - len(keep))]
        selected = (keep + fill)[:top_n]
        current_holdings = selected
        if selected:
            weights.loc[date, selected] = 1 / len(selected)

        is_focus_rebalance = focus_start <= date <= focus_end
        snapshot = [
            holding_record(
                date,
                ticker,
                selected.index(ticker) + 1,
                ranks,
                score,
                security_map,
                sector_map,
                target_weight=1 / len(selected),
                source="rebalance",
            )
            for ticker in selected
        ]
        if date < focus_start:
            last_pre_focus_snapshot = snapshot
        if is_focus_rebalance:
            holding_rows.extend(snapshot)
            action_rows.extend(
                action_records(
                    date,
                    previous,
                    keep,
                    fill,
                    selected,
                    ranks,
                    score,
                    security_map,
                    sector_map,
                    hold_until_rank,
                )
            )
            summary_rows.append(
                {
                    "rebalance_date": date.date().isoformat(),
                    "holdings": len(selected),
                    "kept": len(set(previous).intersection(selected)),
                    "bought": len(set(selected).difference(previous)),
                    "sold": len(set(previous).difference(selected)),
                    "average_rank": float(pd.Series([ranks[ticker] for ticker in selected if ticker in ranks]).mean()),
                    "best_rank": int(min(ranks[ticker] for ticker in selected if ticker in ranks)),
                    "worst_rank": int(max(ranks[ticker] for ticker in selected if ticker in ranks)),
                    "average_momentum_score": float(pd.Series([score[ticker] for ticker in selected if ticker in score]).mean()),
                }
            )

    weights = weights.ffill().fillna(0.0)
    if last_pre_focus_snapshot:
        for row in last_pre_focus_snapshot:
            updated = row.copy()
            updated["rebalance_date"] = focus_start.date().isoformat()
            updated["source"] = "carried_from_previous_rebalance"
            holding_rows.insert(0, updated)

    return (
        weights,
        pd.DataFrame(action_rows),
        pd.DataFrame(holding_rows),
        pd.DataFrame(summary_rows),
    )


def holding_record(
    date: pd.Timestamp,
    ticker: str,
    portfolio_slot: int,
    ranks: pd.Series,
    score: pd.Series,
    security_map: dict[str, str],
    sector_map: dict[str, str],
    *,
    target_weight: float,
    source: str,
) -> dict[str, Any]:
    rank = int(ranks[ticker]) if ticker in ranks else None
    momentum_score = float(score[ticker]) if ticker in score else None
    return {
        "rebalance_date": date.date().isoformat(),
        "portfolio_slot": portfolio_slot,
        "ticker": ticker,
        "security": security_map.get(ticker, ticker),
        "sector": sector_map.get(ticker, "Unknown"),
        "momentum_rank": rank,
        "momentum_score_12_1": momentum_score,
        "target_weight": target_weight,
        "source": source,
    }


def action_records(
    date: pd.Timestamp,
    previous: list[str],
    keep: list[str],
    fill: list[str],
    selected: list[str],
    ranks: pd.Series,
    score: pd.Series,
    security_map: dict[str, str],
    sector_map: dict[str, str],
    hold_until_rank: int,
) -> list[dict[str, Any]]:
    rows = []
    previous_set = set(previous)
    selected_set = set(selected)
    sold = sorted(previous_set - selected_set)
    bought = [ticker for ticker in selected if ticker not in previous_set]
    kept = [ticker for ticker in selected if ticker in previous_set]
    previous_weight = 1 / len(previous) if previous else 0.0
    selected_weight = 1 / len(selected) if selected else 0.0

    for ticker in sold:
        rank = int(ranks[ticker]) if ticker in ranks else None
        rows.append(
            base_action_row(
                date,
                ticker,
                "SELL",
                rank,
                score,
                security_map,
                sector_map,
                previous_weight,
                0.0,
                sell_reason(rank, hold_until_rank),
            )
        )
    for ticker in bought:
        rank = int(ranks[ticker]) if ticker in ranks else None
        rows.append(
            base_action_row(
                date,
                ticker,
                "BUY",
                rank,
                score,
                security_map,
                sector_map,
                0.0,
                selected_weight,
                f"fills an open slot with one of the highest-ranked names after keeping ranks <= {hold_until_rank}",
            )
        )
    for ticker in kept:
        rank = int(ranks[ticker]) if ticker in ranks else None
        rows.append(
            base_action_row(
                date,
                ticker,
                "HOLD",
                rank,
                score,
                security_map,
                sector_map,
                previous_weight,
                selected_weight,
                f"existing holding remains inside hold band: rank {rank} <= {hold_until_rank}",
            )
        )
    rows.sort(key=lambda row: (row["rebalance_date"], action_sort_key(row["action"]), row["momentum_rank"] or 9999, row["ticker"]))
    return rows


def base_action_row(
    date: pd.Timestamp,
    ticker: str,
    action: str,
    rank: int | None,
    score: pd.Series,
    security_map: dict[str, str],
    sector_map: dict[str, str],
    previous_weight: float,
    target_weight: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "rebalance_date": date.date().isoformat(),
        "action": action,
        "ticker": ticker,
        "security": security_map.get(ticker, ticker),
        "sector": sector_map.get(ticker, "Unknown"),
        "momentum_rank": rank,
        "momentum_score_12_1": float(score[ticker]) if ticker in score else None,
        "previous_weight": previous_weight,
        "target_weight": target_weight,
        "weight_delta": target_weight - previous_weight,
        "reason": reason,
    }


def sell_reason(rank: int | None, hold_until_rank: int) -> str:
    if rank is None:
        return "sold because ticker has no valid momentum rank at rebalance"
    return f"sold because momentum rank {rank} moved outside the hold band and is now worse than {hold_until_rank}"


def action_sort_key(action: str) -> int:
    return {"SELL": 0, "BUY": 1, "HOLD": 2}.get(action, 9)


def write_report(
    output_dir: Path,
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    rebalance_summary: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>momentum_hold{args.hold_until_rank} {args.focus_year} Trade Log</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #111827; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    p {{ color: #4B5563; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #E5E7EB; padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, td:nth-child(2), td:nth-child(3), td:nth-child(4), td:last-child {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>momentum_hold{args.hold_until_rank} {args.focus_year} Trade Log</h1>
  <p>Current S&P 500 constituents {constituent_count}; eligible {eligible_count}; monthly rebalance; top {args.top_n}; hold until rank {args.hold_until_rank}; one-way costs {args.cost_bps:g} bps.</p>
  <p>Signal: 12-1 momentum, czyli 252-session return with the latest {args.skip} sessions skipped. Current-constituent survivorship bias remains.</p>
  <h2>Strategy Summary Since {args.focus_year}-01-01</h2>
  {summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Rebalance Summary</h2>
  {rebalance_summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Transactions And Holds</h2>
  {actions.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Holdings By Rebalance</h2>
  {holdings.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_summary(
    output_dir: Path,
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    rebalance_summary: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
) -> None:
    buy_sell = actions.loc[actions["action"].isin(["BUY", "SELL"])] if not actions.empty else pd.DataFrame()
    text = f"""# momentum_hold{args.hold_until_rank} {args.focus_year} Trade Log

Universe: current S&P 500 constituents {constituent_count}; eligible tickers {eligible_count}.

Rules: top {args.top_n} by 12-1 momentum, monthly rebalance, keep existing positions while rank <= {args.hold_until_rank}, equal weight.

Universe caveat: current constituents only, so delisted and removed historical members are missing.

## Strategy Summary Since {args.focus_year}-01-01

{markdown_table(summary)}

## Rebalance Summary

{markdown_table(rebalance_summary)}

## Buy/Sell Transactions

{markdown_table(buy_sell)}

## Holdings By Rebalance

{markdown_table(holdings)}

## CSV Artifacts

- `momentum_hold{args.hold_until_rank}_{args.focus_year}_actions.csv`
- `momentum_hold{args.hold_until_rank}_{args.focus_year}_holdings.csv`
- `momentum_hold{args.hold_until_rank}_{args.focus_year}_rebalance_summary.csv`
- `weights_momentum_hold{args.hold_until_rank}.csv`
- `strategy_summary.csv`
- `daily_returns.csv`
- `equity_curves.csv`
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    constituent_count: int,
    eligible_count: int,
) -> None:
    payload = {
        "input_start": start.date().isoformat(),
        "input_end": end.date().isoformat(),
        "focus_year": args.focus_year,
        "strategy": f"momentum_hold{args.hold_until_rank}",
        "lookback_sessions": args.lookback,
        "skip_sessions": args.skip,
        "top_n": args.top_n,
        "hold_until_rank": args.hold_until_rank,
        "rebalance": args.rebalance,
        "cost_bps": args.cost_bps,
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "survivorship_bias_warning": "Uses current S&P 500 constituents across the full backtest.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
