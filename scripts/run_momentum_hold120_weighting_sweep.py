from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_bot_candidate_chart_pack import add_qqq_alpha
from scripts.run_ml_hypothesis_suite import markdown_table, momentum_scores
from scripts.run_ml_ranker_walkforward import (
    BENCHMARK_TICKERS,
    _first_active_weight_date,
    _load_universe,
    _trim_run,
    add_alpha_columns,
    choose_eligible_tickers,
    data_quality,
    load_yfinance_close,
)
from stockresearchmarket.strategies.ml_ranker import buy_hold_run, rebalance_dates, simulate_portfolio, summarize_runs

SCHEME_LABELS = {
    "momentum_hold120_equal": "equal weight",
    "momentum_hold120_rank": "rank weight",
    "momentum_hold120_score": "score weight",
    "momentum_hold120_score_cap6": "score weight capped 6%",
    "momentum_hold120_inverse_vol": "inverse vol",
    "momentum_hold120_inverse_vol_cap6": "inverse vol capped 6%",
    "SPY": "SPY",
    "QQQ": "QQQ",
}
COLORS = {
    "momentum_hold120_equal": "#2563EB",
    "momentum_hold120_rank": "#0F766E",
    "momentum_hold120_score": "#DC2626",
    "momentum_hold120_score_cap6": "#7C3AED",
    "momentum_hold120_inverse_vol": "#EA580C",
    "momentum_hold120_inverse_vol_cap6": "#0891B2",
    "SPY": "#374151",
    "QQQ": "#F59E0B",
}
VARIANT_ORDER = [
    "momentum_hold120_equal",
    "momentum_hold120_rank",
    "momentum_hold120_score",
    "momentum_hold120_score_cap6",
    "momentum_hold120_inverse_vol",
    "momentum_hold120_inverse_vol_cap6",
]
CHART_ORDER = [*VARIANT_ORDER, "SPY", "QQQ"]


@dataclass(frozen=True)
class SelectionSnapshot:
    date: pd.Timestamp
    selected: list[str]
    scores: pd.Series
    ranks: pd.Series


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / stamp
    charts_dir = output_dir / "charts"
    weights_dir = output_dir / "weights"
    charts_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start).normalize() if args.start else end - pd.DateOffset(years=args.years)
    constituents = _load_universe(args, output_dir)
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
    trace = build_hold_band_selection_trace(
        eligible_close,
        scores,
        top_n=args.top_n,
        hold_until_rank=args.hold_until_rank,
        rebalance=args.rebalance,
    )
    selection_summary = summarize_selection_trace(trace)
    selection_summary.to_csv(output_dir / "selection_trace_summary.csv", index=False)

    weight_variants = build_weight_variants(
        eligible_close,
        trace,
        cap_weight=args.cap_weight,
        vol_lookback=args.vol_lookback,
    )
    for name, weights in weight_variants.items():
        weights.to_csv(weights_dir / f"{name}.csv")

    strategy_runs = [simulate_portfolio(eligible_close, weight_variants[name], name, cost_bps=args.cost_bps) for name in VARIANT_ORDER]
    benchmark_runs = [buy_hold_run(benchmark_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in ["SPY", "QQQ"]]
    common_start = max(_first_active_weight_date(weight_variants[name]) for name in VARIANT_ORDER)
    all_runs = [_trim_run(run, common_start) for run in strategy_runs + benchmark_runs]

    summary = summarize_runs(all_runs, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="SPY")
    summary = add_qqq_alpha(summary)
    concentration = summarize_concentration(weight_variants, common_start)
    concentration.to_csv(output_dir / "concentration_summary.csv", index=False)
    summary = summary.merge(concentration, on="strategy", how="left")
    summary["label"] = summary["strategy"].map(SCHEME_LABELS).fillna(summary["strategy"])
    summary["display_order"] = summary["strategy"].map({name: idx for idx, name in enumerate(CHART_ORDER)}).fillna(99)
    summary = summary.sort_values(["display_order"]).drop(columns=["display_order"]).reset_index(drop=True)

    returns = pd.concat([run.returns for run in all_runs], axis=1).sort_index()
    equity = pd.concat([run.equity for run in all_runs], axis=1).sort_index()
    drawdown = equity.div(equity.cummax()) - 1
    turnover = pd.concat([run.turnover for run in strategy_runs], axis=1).sort_index().loc[common_start:]
    rebalance_concentration = build_rebalance_concentration(weight_variants).loc[lambda frame: frame["date"] >= common_start.date().isoformat()]

    summary.to_csv(output_dir / "weighting_summary.csv", index=False)
    returns.to_csv(output_dir / "daily_returns.csv")
    equity.to_csv(output_dir / "equity_curves.csv")
    drawdown.to_csv(output_dir / "drawdowns.csv")
    turnover.to_csv(output_dir / "turnover.csv")
    rebalance_concentration.to_csv(output_dir / "rebalance_concentration.csv", index=False)

    chart_paths = make_charts(charts_dir, equity, drawdown, rebalance_concentration, args.zoom_starts)
    write_report(output_dir, summary, selection_summary, chart_paths, common_start, start, end, args, len(constituents), len(eligible))
    write_summary(output_dir, summary, selection_summary, chart_paths, common_start, start, end, args, len(constituents), len(eligible))
    write_metadata(output_dir, args, start, end, common_start, len(constituents), len(eligible))

    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare weighting schemes for momentum_hold120.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start")
    parser.add_argument("--end")
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
    parser.add_argument("--cap-weight", type=float, default=0.06)
    parser.add_argument("--vol-lookback", type=int, default=63)
    parser.add_argument("--zoom-starts", default="2025-01-01,2026-01-01")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/momentum_hold120_weighting_sweep")
    return parser.parse_args()


def build_hold_band_selection_trace(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    top_n: int,
    hold_until_rank: int,
    rebalance: str,
) -> list[SelectionSnapshot]:
    current_holdings: list[str] = []
    trace: list[SelectionSnapshot] = []
    for date in rebalance_dates(close.index, rebalance):
        score = scores.loc[date].dropna().sort_values(ascending=False)
        if score.empty:
            continue
        ranks = score.rank(ascending=False, method="first")
        keep = [ticker for ticker in current_holdings if ticker in ranks.index and ranks[ticker] <= hold_until_rank]
        fill = [ticker for ticker in score.index if ticker not in keep][: max(0, top_n - len(keep))]
        selected = (keep + fill)[:top_n]
        current_holdings = selected
        trace.append(SelectionSnapshot(pd.Timestamp(date), selected, score, ranks))
    return trace


def build_weight_variants(
    close: pd.DataFrame,
    trace: list[SelectionSnapshot],
    *,
    cap_weight: float,
    vol_lookback: int,
) -> dict[str, pd.DataFrame]:
    variants = {name: pd.DataFrame(index=close.index, columns=close.columns, dtype="float64") for name in VARIANT_ORDER}
    for snapshot in trace:
        for name in VARIANT_ORDER:
            variants[name].loc[snapshot.date] = 0.0
            weights = scheme_weights(
                close,
                snapshot,
                scheme=name,
                cap_weight=cap_weight,
                vol_lookback=vol_lookback,
            )
            if not weights.empty:
                variants[name].loc[snapshot.date, weights.index] = weights
    return {name: frame.ffill().fillna(0.0) for name, frame in variants.items()}


def scheme_weights(
    close: pd.DataFrame,
    snapshot: SelectionSnapshot,
    *,
    scheme: str,
    cap_weight: float,
    vol_lookback: int,
) -> pd.Series:
    selected = snapshot.selected
    if not selected:
        return pd.Series(dtype="float64")
    selected_scores = snapshot.scores.reindex(selected).astype(float)
    if scheme.endswith("_equal"):
        raw = pd.Series(1.0, index=selected, dtype="float64")
    elif scheme.endswith("_rank"):
        ordered = sorted(selected, key=lambda ticker: (float(snapshot.ranks[ticker]), ticker))
        raw = pd.Series(np.arange(len(ordered), 0, -1, dtype=float), index=ordered)
    elif scheme.endswith("_score") or "score_cap" in scheme:
        floor = float(selected_scores.min())
        raw = selected_scores - floor + 1e-6
    elif scheme.endswith("_inverse_vol") or "inverse_vol_cap" in scheme:
        returns = close[selected].loc[: snapshot.date].pct_change(fill_method=None).tail(vol_lookback)
        risk = returns.std(ddof=0).replace(0, np.nan)
        raw = (1 / risk).replace([np.inf, -np.inf], np.nan)
    else:
        raise ValueError(f"Unsupported weighting scheme: {scheme}")

    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty or raw.sum() <= 0:
        raw = pd.Series(1.0, index=selected, dtype="float64")
    weights = raw / raw.sum()
    if "cap" in scheme:
        weights = cap_weights(weights, cap_weight)
    return weights / weights.sum()


def cap_weights(weights: pd.Series, cap_weight: float) -> pd.Series:
    if weights.empty or cap_weight <= 0:
        return weights
    capped = weights.clip(upper=cap_weight)
    leftover = 1 - capped.sum()
    while leftover > 1e-9 and (capped < cap_weight - 1e-12).any():
        room = cap_weight - capped[capped < cap_weight]
        add = room / room.sum() * leftover
        capped.loc[add.index] = (capped.loc[add.index] + add).clip(upper=cap_weight)
        leftover = 1 - capped.sum()
    return capped / capped.sum() if capped.sum() > 0 else weights


def summarize_selection_trace(trace: list[SelectionSnapshot]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    previous: list[str] = []
    for snapshot in trace:
        selected = snapshot.selected
        rows.append(
            {
                "rebalance_date": snapshot.date.date().isoformat(),
                "holdings": len(selected),
                "kept": len(set(previous).intersection(selected)),
                "bought": len(set(selected).difference(previous)),
                "sold": len(set(previous).difference(selected)),
                "average_rank": float(pd.Series([snapshot.ranks[ticker] for ticker in selected]).mean()),
                "best_rank": int(min(snapshot.ranks[ticker] for ticker in selected)),
                "worst_rank": int(max(snapshot.ranks[ticker] for ticker in selected)),
                "average_score": float(pd.Series([snapshot.scores[ticker] for ticker in selected]).mean()),
            }
        )
        previous = selected
    return pd.DataFrame(rows)


def summarize_concentration(weight_variants: dict[str, pd.DataFrame], common_start: pd.Timestamp) -> pd.DataFrame:
    rebalance_concentration = build_rebalance_concentration(weight_variants)
    filtered = rebalance_concentration.loc[rebalance_concentration["date"] >= common_start.date().isoformat()].copy()
    grouped = (
        filtered.groupby("strategy")
        .agg(
            avg_rebalance_max_weight=("max_weight", "mean"),
            worst_rebalance_max_weight=("max_weight", "max"),
            avg_effective_names=("effective_names", "mean"),
        )
        .reset_index()
    )
    return grouped


def build_rebalance_concentration(weight_variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for strategy, weights in weight_variants.items():
        rebalance_mask = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1)).gt(0)
        active = weights.loc[rebalance_mask & weights.sum(axis=1).gt(0)]
        for date, row in active.iterrows():
            positive = row[row > 0]
            if positive.empty:
                continue
            effective_names = 1 / positive.pow(2).sum()
            rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "strategy": strategy,
                    "max_weight": float(positive.max()),
                    "effective_names": float(effective_names),
                }
            )
    return pd.DataFrame(rows)


def make_charts(
    charts_dir: Path,
    equity: pd.DataFrame,
    drawdown: pd.DataFrame,
    rebalance_concentration: pd.DataFrame,
    zoom_starts_raw: str,
) -> list[Path]:
    zoom_starts = [pd.Timestamp(item.strip()) for item in zoom_starts_raw.split(",") if item.strip()]
    chart_paths = [
        make_equity_drawdown_chart(charts_dir / "weighting_equity_drawdown.png", equity, drawdown, title_suffix="full period"),
        make_concentration_chart(charts_dir / "weighting_concentration.png", rebalance_concentration),
    ]
    for zoom_start in zoom_starts:
        zoom_equity = equity.loc[equity.index >= zoom_start].copy()
        if zoom_equity.empty:
            continue
        rebased = zoom_equity.div(zoom_equity.iloc[0])
        zoom_drawdown = rebased.div(rebased.cummax()) - 1
        chart_paths.append(
            make_equity_drawdown_chart(
                charts_dir / f"weighting_equity_drawdown_{zoom_start.strftime('%Y%m%d')}.png",
                rebased,
                zoom_drawdown,
                title_suffix=f"from {zoom_equity.index[0].date().isoformat()}",
            )
        )
    return chart_paths


def make_equity_drawdown_chart(path: Path, equity: pd.DataFrame, drawdown: pd.DataFrame, *, title_suffix: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, height_ratios=[3, 2])
    strategies = [column for column in CHART_ORDER if column in equity.columns]
    for strategy in strategies:
        axes[0].plot(equity.index, equity[strategy], label=SCHEME_LABELS.get(strategy, strategy), color=COLORS.get(strategy), linewidth=2.1 if strategy not in {"SPY", "QQQ"} else 1.8)
        axes[1].plot(drawdown.index, drawdown[strategy], label=SCHEME_LABELS.get(strategy, strategy), color=COLORS.get(strategy), linewidth=2.1 if strategy not in {"SPY", "QQQ"} else 1.8)
    axes[0].set_title(f"momentum_hold120 weighting sweep: equity ({title_suffix})")
    axes[1].set_title("Drawdown")
    axes[0].set_ylabel("Equity")
    axes[1].set_ylabel("Drawdown")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_concentration_chart(path: Path, rebalance_concentration: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, height_ratios=[2, 2])
    frame = rebalance_concentration.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for strategy in VARIANT_ORDER:
        subset = frame.loc[frame["strategy"].eq(strategy)]
        if subset.empty:
            continue
        axes[0].plot(subset["date"], subset["max_weight"], label=SCHEME_LABELS[strategy], color=COLORS[strategy], linewidth=2.0)
        axes[1].plot(subset["date"], subset["effective_names"], label=SCHEME_LABELS[strategy], color=COLORS[strategy], linewidth=2.0)
    axes[0].set_title("Concentration by rebalance")
    axes[0].set_ylabel("Max position weight")
    axes[1].set_ylabel("Effective number of names")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    chart_paths: list[Path],
    common_start: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
) -> None:
    images = "\n".join(
        f'<figure><img src="{path.relative_to(output_dir).as_posix()}" alt="{path.name}" style="width: 100%; border: 1px solid #E5E7EB;"><figcaption>{path.name}</figcaption></figure>'
        for path in chart_paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>momentum_hold120 weighting sweep</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #111827; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    p, li {{ color: #4B5563; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #E5E7EB; padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ display: block; margin: 12px 0 28px; }}
  </style>
</head>
<body>
<main>
  <h1>momentum_hold120 weighting sweep</h1>
  <p>Measured period: {common_start.date().isoformat()} to {end.date().isoformat()}. Warmup data: {start.date().isoformat()} to {(common_start - pd.Timedelta(days=1)).date().isoformat()}.</p>
  <p>Universe: current S&P 500 constituents {constituent_count}; eligible {eligible_count}; top {args.top_n}; hold until rank {args.hold_until_rank}; one-way cost {args.cost_bps:g} bps; current-constituent survivorship bias remains.</p>
  <p>Weighting assumptions: capped variants use max position weight {args.cap_weight:.0%}; inverse-vol variants use trailing {args.vol_lookback} trading days of realized volatility.</p>
  <h2>Strategy Summary</h2>
  {summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Selection Trace</h2>
  {selection_summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Charts</h2>
  {images}
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_summary(
    output_dir: Path,
    summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    chart_paths: list[Path],
    common_start: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
) -> None:
    chart_lines = "\n".join(f"- `{path.relative_to(output_dir)}`" for path in chart_paths)
    text = f"""# momentum_hold120 weighting sweep

Measured period: {common_start.date().isoformat()} to {end.date().isoformat()}.

Warmup period: {start.date().isoformat()} to {(common_start - pd.Timedelta(days=1)).date().isoformat()}.

Universe: current S&P 500 constituents {constituent_count}; eligible tickers {eligible_count}; top {args.top_n}; hold until rank {args.hold_until_rank}; one-way costs {args.cost_bps:g} bps.

Weighting assumptions:

- capped variants use max single-name weight `{args.cap_weight:.0%}`
- inverse-vol variants use trailing `{args.vol_lookback}` sessions
- selection logic stays identical across variants; only weights change
- universe remains survivorship-biased because it uses current constituents only

## Strategy Summary

{markdown_table(summary)}

## Selection Trace

{markdown_table(selection_summary)}

## Charts

{chart_lines}
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    common_start: pd.Timestamp,
    constituent_count: int,
    eligible_count: int,
) -> None:
    payload = {
        "input_start": start.date().isoformat(),
        "input_end": end.date().isoformat(),
        "measured_start": common_start.date().isoformat(),
        "strategy_family": "momentum_hold120_weighting_sweep",
        "lookback_sessions": args.lookback,
        "skip_sessions": args.skip,
        "top_n": args.top_n,
        "hold_until_rank": args.hold_until_rank,
        "rebalance": args.rebalance,
        "cost_bps": args.cost_bps,
        "cap_weight": args.cap_weight,
        "vol_lookback": args.vol_lookback,
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "survivorship_bias_warning": "Uses current S&P 500 constituents across the full backtest.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
