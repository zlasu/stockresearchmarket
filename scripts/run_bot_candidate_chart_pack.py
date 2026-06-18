from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ml_hypothesis_suite import markdown_table, momentum_scores, rank_weights_from_scores
from scripts.run_ml_ranker_sweep import Variant, apply_risk_overlay, effective_min_train_rows, extra_trees
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
from scripts.run_momentum_hold_risk_sweep import apply_momentum_overlay
from stockresearchmarket.strategies.ml_ranker import (
    FEATURE_COLUMNS,
    build_ml_ranker_weights,
    buy_hold_run,
    make_price_feature_panel,
    momentum_weights,
    rebalance_dates,
    simulate_portfolio,
    summarize_runs,
)

BOT_ORDER = [
    "momentum_hold75",
    "momentum_hold120",
    "momentum_12_1_top30",
    "momentum_hold75_sma200_50",
    "et_t30_5y_market_sma200",
]
CHART_ORDER = [*BOT_ORDER, "SPY", "QQQ"]
BOT_ROLES = {
    "momentum_hold75": "return_first_finalist",
    "momentum_hold120": "stability_first_finalist",
    "momentum_12_1_top30": "high_turnover_control",
    "momentum_hold75_sma200_50": "defensive_overlay",
    "et_t30_5y_market_sma200": "ml_research_slot",
    "SPY": "benchmark",
    "QQQ": "benchmark",
}
COLORS = {
    "momentum_hold75": "#0F766E",
    "momentum_hold120": "#2563EB",
    "momentum_12_1_top30": "#DC2626",
    "momentum_hold75_sma200_50": "#7C3AED",
    "et_t30_5y_market_sma200": "#9333EA",
    "SPY": "#374151",
    "QQQ": "#F97316",
}


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / stamp
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

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
    if len(eligible) < 50:
        raise RuntimeError(f"Only {len(eligible)} eligible tickers; bot chart pack needs a broad universe.")

    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns]].dropna(how="all").ffill(limit=5)
    weights = build_bot_weights(eligible_close, benchmark_close, args)
    strategy_runs = [simulate_portfolio(eligible_close, weights[name], name, cost_bps=args.cost_bps) for name in BOT_ORDER]
    benchmark_runs = [buy_hold_run(benchmark_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in ["SPY", "QQQ"]]
    common_start = max(_first_active_weight_date(weights[name]) for name in BOT_ORDER)
    all_runs = [_trim_run(run, common_start) for run in strategy_runs + benchmark_runs]

    summary = summarize_runs(all_runs, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="SPY")
    summary = add_qqq_alpha(summary)
    summary["bot_role"] = summary["strategy"].map(BOT_ROLES).fillna("candidate")
    summary["display_order"] = summary["strategy"].map({name: idx for idx, name in enumerate(CHART_ORDER)}).fillna(99)
    summary = summary.sort_values(["display_order"]).drop(columns=["display_order"]).reset_index(drop=True)
    returns = pd.concat([run.returns for run in all_runs], axis=1).sort_index()
    equity = pd.concat([run.equity for run in all_runs], axis=1).sort_index()
    drawdown = equity.div(equity.cummax()) - 1
    turnover = pd.concat([run.turnover for run in strategy_runs], axis=1).sort_index().loc[common_start:]
    monthly_returns = compound_returns(returns, "ME")
    yearly_returns = compound_returns(returns, "YE")
    rolling = rolling_daily_metrics(returns, args.rolling_days, args.risk_free_rate)
    zoom_starts = parse_zoom_starts(args.zoom_starts)

    summary.to_csv(output_dir / "bot_candidate_summary.csv", index=False)
    returns.to_csv(output_dir / "daily_returns.csv")
    equity.to_csv(output_dir / "equity_curves.csv")
    drawdown.to_csv(output_dir / "drawdowns.csv")
    turnover.to_csv(output_dir / "turnover.csv")
    monthly_returns.to_csv(output_dir / "monthly_returns.csv")
    yearly_returns.to_csv(output_dir / "yearly_returns.csv")
    rolling.to_csv(output_dir / f"rolling_{args.rolling_days}d_metrics.csv", index=False)

    chart_paths = make_charts(charts_dir, summary, equity, drawdown, returns, yearly_returns, rolling, zoom_starts)
    write_report(output_dir, chart_paths, summary, common_start, start, end, args, len(constituents), len(eligible))
    write_summary(output_dir, chart_paths, summary, common_start, start, end, args, len(constituents), len(eligible))
    write_metadata(output_dir, args, start, end, common_start, len(constituents), len(eligible))

    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chart pack for the five selected bot test candidates.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--min-years", type=float, default=7.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--skip", type=int, default=21)
    parser.add_argument("--horizon-days", type=int, default=21)
    parser.add_argument("--rebalance", default="ME")
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--rolling-days", type=int, default=252)
    parser.add_argument("--zoom-starts", default="2025-01-01,2026-01-01")
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/bot_candidate_chart_pack")
    return parser.parse_args()


def parse_zoom_starts(value: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(item.strip()) for item in value.split(",") if item.strip()]


def build_bot_weights(
    eligible_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    scores = momentum_scores(eligible_close, lookback=args.lookback, skip=args.skip)
    hold75 = rank_weights_from_scores(
        eligible_close,
        scores,
        top_n=args.top_n,
        rebalance=args.rebalance,
        hold_until_rank=75,
    )
    weights = {
        "momentum_hold75": hold75,
        "momentum_hold120": rank_weights_from_scores(
            eligible_close,
            scores,
            top_n=args.top_n,
            rebalance=args.rebalance,
            hold_until_rank=120,
        ),
        "momentum_12_1_top30": momentum_weights(
            eligible_close,
            lookback=args.lookback,
            skip=args.skip,
            top_n=args.top_n,
            rebalance=args.rebalance,
        ),
        "momentum_hold75_sma200_50": apply_momentum_overlay(hold75, eligible_close, benchmark_close, "sma200_50"),
    }

    dates = rebalance_dates(eligible_close.index, args.rebalance)
    feature_panel = make_price_feature_panel(eligible_close, dates, args.horizon_days)
    variant = Variant(
        "et_t30_5y_market_sma200",
        "extra_trees",
        30,
        5,
        10_000,
        FEATURE_COLUMNS,
        "equal",
        extra_trees(25, 40, args.random_state),
        "market_sma200",
    )
    ml_result = build_ml_ranker_weights(
        eligible_close,
        top_n=variant.top_n,
        train_years=variant.train_years,
        rebalance=args.rebalance,
        horizon_days=args.horizon_days,
        min_train_rows=effective_min_train_rows(variant, eligible_close),
        random_state=args.random_state,
        model_factory=variant.model_factory,
        feature_columns=variant.feature_columns,
        feature_panel=feature_panel,
        weighting=variant.weighting,
    )
    weights[variant.variant_id] = apply_risk_overlay(ml_result.weights, eligible_close, benchmark_close, variant.risk_overlay)
    return weights


def add_qqq_alpha(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    if "QQQ" not in set(frame["strategy"]):
        return frame
    qqq_return = float(frame.loc[frame["strategy"].eq("QQQ"), "total_return"].iloc[0])
    qqq_cagr = float(frame.loc[frame["strategy"].eq("QQQ"), "cagr"].iloc[0])
    frame["alpha_total_return_vs_qqq"] = frame["total_return"] - qqq_return
    frame["alpha_cagr_vs_qqq"] = frame["cagr"] - qqq_cagr
    return frame


def compound_returns(returns: pd.DataFrame, frequency: str) -> pd.DataFrame:
    return returns.resample(frequency).apply(lambda values: (1 + values).prod() - 1).dropna(how="all")


def rolling_daily_metrics(returns: pd.DataFrame, window: int, risk_free_rate: float) -> pd.DataFrame:
    records = []
    for strategy in returns.columns:
        series = returns[strategy].fillna(0.0)
        rolling_return = (1 + series).rolling(window).apply(np.prod, raw=True) - 1
        rolling_std = series.rolling(window).std(ddof=0)
        rolling_sharpe = (series.rolling(window).mean() - risk_free_rate / 252) / rolling_std.replace(0, np.nan) * np.sqrt(252)
        for date in rolling_return.dropna().index:
            records.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "strategy": strategy,
                    "rolling_return": float(rolling_return.loc[date]),
                    "rolling_sharpe": float(rolling_sharpe.loc[date]) if pd.notna(rolling_sharpe.loc[date]) else np.nan,
                }
            )
    return pd.DataFrame(records)


def make_charts(
    charts_dir: Path,
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    drawdown: pd.DataFrame,
    returns: pd.DataFrame,
    yearly_returns: pd.DataFrame,
    rolling: pd.DataFrame,
    zoom_starts: list[pd.Timestamp],
) -> list[Path]:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    zoom_paths = [plot_recent_equity_drawdown(charts_dir, equity, zoom_start) for zoom_start in zoom_starts]
    paths = [
        plot_equity_drawdown(charts_dir, equity, drawdown, "bot_candidates_equity_drawdown.png", "Bot Candidates vs SPY/QQQ"),
        *zoom_paths,
        plot_rolling(charts_dir, rolling),
        plot_annual_returns(charts_dir, yearly_returns),
        plot_return_drawdown(charts_dir, summary),
        plot_underwater(charts_dir, drawdown),
    ]
    for path in paths:
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Empty chart file: {path}")
    return paths


def plot_equity_drawdown(charts_dir: Path, equity: pd.DataFrame, drawdown: pd.DataFrame, filename: str, title: str) -> Path:
    columns = [column for column in CHART_ORDER if column in equity]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for column in columns:
        color = COLORS.get(column)
        lw = 2.8 if column in BOT_ORDER[:2] else 1.7
        style = "--" if column in {"SPY", "QQQ"} else "-"
        axes[0].plot(equity.index, equity[column], label=column, color=color, linewidth=lw, linestyle=style)
        axes[1].plot(drawdown.index, drawdown[column], label=column, color=color, linewidth=lw, linestyle=style)
    axes[0].set_title(title)
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_title("Drawdowns")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = charts_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recent_equity_drawdown(charts_dir: Path, equity: pd.DataFrame, requested_start: pd.Timestamp) -> Path:
    recent_equity = equity.loc[requested_start:].copy()
    if recent_equity.empty:
        raise RuntimeError(f"No equity data available from {requested_start.date().isoformat()}.")
    actual_start = pd.Timestamp(recent_equity.index[0])
    recent_equity = recent_equity.div(recent_equity.iloc[0])
    recent_drawdown = recent_equity.div(recent_equity.cummax()) - 1
    start_slug = requested_start.strftime("%Y%m%d")
    return plot_equity_drawdown(
        charts_dir,
        recent_equity,
        recent_drawdown,
        f"bot_candidates_zoom_{start_slug}_equity_drawdown.png",
        (
            f"Bot Candidates vs SPY/QQQ Since {requested_start.date().isoformat()} "
            f"(Rebased on {actual_start.date().isoformat()})"
        ),
    )


def plot_rolling(charts_dir: Path, rolling: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    if not rolling.empty:
        frame = rolling.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        for strategy in CHART_ORDER:
            group = frame.loc[frame["strategy"].eq(strategy)].sort_values("date")
            if group.empty:
                continue
            style = "--" if strategy in {"SPY", "QQQ"} else "-"
            lw = 2.7 if strategy in BOT_ORDER[:2] else 1.6
            axes[0].plot(group["date"], group["rolling_return"], color=COLORS.get(strategy), label=strategy, linewidth=lw, linestyle=style)
            axes[1].plot(group["date"], group["rolling_sharpe"], color=COLORS.get(strategy), label=strategy, linewidth=lw, linestyle=style)
    axes[0].set_title("Rolling 252D Return")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    axes[1].set_title("Rolling 252D Sharpe")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = charts_dir / "bot_candidates_rolling_252d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_annual_returns(charts_dir: Path, yearly_returns: pd.DataFrame) -> Path:
    columns = [column for column in CHART_ORDER if column in yearly_returns]
    frame = yearly_returns[columns].copy()
    frame.index = pd.DatetimeIndex(frame.index).year
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(frame.index))
    width = 0.84 / len(columns)
    for idx, column in enumerate(columns):
        ax.bar(x + idx * width - 0.42 + width / 2, frame[column], width=width, label=column, color=COLORS.get(column))
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(frame.index.astype(str), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.set_title("Calendar-Year Returns")
    ax.set_ylabel("Return")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = charts_dir / "bot_candidates_annual_returns.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_return_drawdown(charts_dir: Path, summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for _, row in summary.iterrows():
        strategy = str(row["strategy"])
        ax.scatter(row["max_drawdown"], row["cagr"], color=COLORS.get(strategy, "#6B7280"), s=80)
        ax.annotate(strategy, (row["max_drawdown"], row["cagr"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("CAGR vs Max Drawdown")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("CAGR")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    fig.tight_layout()
    path = charts_dir / "bot_candidates_return_drawdown.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_underwater(charts_dir: Path, drawdown: pd.DataFrame) -> Path:
    columns = [column for column in CHART_ORDER if column in drawdown]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for column in columns:
        style = "--" if column in {"SPY", "QQQ"} else "-"
        lw = 2.7 if column in BOT_ORDER[:2] else 1.5
        ax.plot(drawdown.index, drawdown[column], color=COLORS.get(column), label=column, linewidth=lw, linestyle=style)
    ax.set_title("Underwater History")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    path = charts_dir / "bot_candidates_underwater_history.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    output_dir: Path,
    chart_paths: list[Path],
    summary: pd.DataFrame,
    common_start: pd.Timestamp,
    input_start: pd.Timestamp,
    input_end: pd.Timestamp,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
) -> None:
    image_tags = "\n".join(
        f"<figure><img src='{path.relative_to(output_dir)}' alt='{path.stem}'><figcaption>{path.stem}</figcaption></figure>"
        for path in chart_paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bot Candidate Chart Pack</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #111827; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    p {{ color: #4B5563; }}
    img {{ width: 100%; height: auto; border: 1px solid #E5E7EB; }}
    figure {{ margin: 24px 0; }}
    figcaption {{ color: #6B7280; font-size: 13px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #E5E7EB; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>Bot Candidate Chart Pack</h1>
  <p>Input history {input_start.date().isoformat()} to {input_end.date().isoformat()}; common comparison start {common_start.date().isoformat()}.</p>
  <p>Universe: current S&P 500 constituents {constituent_count}; eligible {eligible_count}; one-way costs {args.cost_bps:g} bps. Current-constituent survivorship bias remains.</p>
  <h2>Charts</h2>
  {image_tags}
  <h2>Summary</h2>
  {summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_summary(
    output_dir: Path,
    chart_paths: list[Path],
    summary: pd.DataFrame,
    common_start: pd.Timestamp,
    input_start: pd.Timestamp,
    input_end: pd.Timestamp,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
) -> None:
    chart_list = "\n".join(f"- `{path.relative_to(output_dir)}`" for path in chart_paths)
    text = f"""# Bot Candidate Chart Pack

Input history: {input_start.date().isoformat()} to {input_end.date().isoformat()}

Common comparison start: {common_start.date().isoformat()}

Universe: current S&P 500 constituents {constituent_count}; eligible tickers {eligible_count}.

Costs: {args.cost_bps:g} bps one-way. Rebalance: {args.rebalance}.

Universe caveat: current constituents only, so delisted and removed historical members are missing.

## Summary

{markdown_table(summary)}

## Charts

{chart_list}

## CSV Artifacts

- `bot_candidate_summary.csv`
- `daily_returns.csv`
- `equity_curves.csv`
- `drawdowns.csv`
- `turnover.csv`
- `monthly_returns.csv`
- `yearly_returns.csv`
- `rolling_{args.rolling_days}d_metrics.csv`
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
        "common_comparison_start": common_start.date().isoformat(),
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "bot_candidates": BOT_ORDER,
        "benchmarks": ["SPY", "QQQ"],
        "cost_bps": args.cost_bps,
        "rebalance": args.rebalance,
        "zoom_starts": [date.date().isoformat() for date in parse_zoom_starts(args.zoom_starts)],
        "survivorship_bias_warning": "Uses current S&P 500 constituents across the full backtest.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
