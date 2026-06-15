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

from scripts.run_ml_hypothesis_suite import (
    cscv_pbo,
    deflated_sharpe_table,
    markdown_table,
    momentum_scores,
    rank_weights_from_scores,
)
from scripts.run_ml_ranker_walkforward import (
    BENCHMARK_TICKERS,
    _load_universe,
    _trim_run,
    add_alpha_columns,
    choose_eligible_tickers,
    data_quality,
    load_yfinance_close,
)
from scripts.run_momentum_hold_risk_sweep import apply_momentum_overlay
from stockresearchmarket.engine.metrics import performance_metrics
from stockresearchmarket.strategies.ml_ranker import (
    StrategyRun,
    buy_hold_run,
    equal_weight_weights,
    momentum_weights,
    simulate_portfolio,
    summarize_runs,
)

FOCUS_STRATEGIES = [
    "momentum_hold75",
    "momentum_hold120",
    "momentum_hold90",
    "momentum_12_1_top30",
    "SPY",
    "QQQ",
    "RSP",
    "eligible_equal_weight",
]

PLOT_COLORS = {
    "momentum_hold75": "#0F766E",
    "momentum_hold120": "#2563EB",
    "momentum_hold90": "#7C3AED",
    "momentum_12_1_top30": "#DC2626",
    "SPY": "#374151",
    "QQQ": "#F97316",
    "RSP": "#64748B",
    "eligible_equal_weight": "#16A34A",
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
    if len(eligible) < 50:
        raise RuntimeError(f"Only {len(eligible)} eligible tickers; finalist validation needs a broad universe.")

    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns]].dropna(how="all").ffill(limit=5)
    scores = momentum_scores(eligible_close, lookback=args.lookback, skip=args.skip)

    weights = build_finalist_weights(eligible_close, benchmark_close, scores, args)
    runs = build_runs(eligible_close, benchmark_close, weights, args)
    common_start = pd.Timestamp(args.comparison_start) if args.comparison_start else common_active_start(runs)
    trimmed = [_trim_run(run, common_start) for run in runs]

    summary = summarize_runs(trimmed, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="SPY")
    summary = add_validation_notes(summary)
    returns = pd.concat([run.returns for run in trimmed], axis=1).sort_index()
    equity = pd.concat([run.equity for run in trimmed], axis=1).sort_index()
    turnover = pd.concat([run.turnover for run in trimmed if run.name in weights], axis=1).sort_index()

    monthly_returns = compound_period_returns(returns, "ME")
    yearly_returns = compound_period_returns(returns, "YE")
    yearly_metrics = period_metrics(returns, turnover, "YE", args.risk_free_rate)
    yearly_pass = pass_rate_summary(yearly_metrics)
    rolling = rolling_window_metrics(returns, turnover, args.rolling_months, args.risk_free_rate)
    rolling_pass = rolling_pass_summary(rolling)
    regime = market_regime_performance(returns, benchmark_close["SPY"].reindex(returns.index).ffill(), args.risk_free_rate)
    sector_exposure = sector_exposure_table(weights, sector_map)
    turnover_yearly = turnover.resample("YE").sum()
    validation = validation_tables(monthly_returns, args.pbo_slices)

    summary.to_csv(output_dir / "finalist_summary.csv", index=False)
    returns.to_csv(output_dir / "daily_returns.csv")
    equity.to_csv(output_dir / "equity_curves.csv")
    turnover.to_csv(output_dir / "turnover.csv")
    monthly_returns.to_csv(output_dir / "monthly_returns.csv")
    yearly_returns.to_csv(output_dir / "yearly_returns.csv")
    yearly_metrics.to_csv(output_dir / "yearly_metrics.csv", index=False)
    yearly_pass.to_csv(output_dir / "yearly_pass_rates.csv", index=False)
    rolling_slug = rolling_window_slug(args.rolling_months)
    rolling.to_csv(output_dir / f"{rolling_slug}_windows.csv", index=False)
    rolling_pass.to_csv(output_dir / f"{rolling_slug}_pass_rates.csv", index=False)
    regime.to_csv(output_dir / "regime_performance.csv", index=False)
    sector_exposure.to_csv(output_dir / "average_sector_exposure.csv", index=False)
    turnover_yearly.to_csv(output_dir / "yearly_turnover.csv")
    validation["deflated_sharpe"].to_csv(output_dir / "deflated_sharpe.csv", index=False)
    validation["pbo_splits"].to_csv(output_dir / "pbo_splits.csv", index=False)
    validation["pbo_summary"].to_csv(output_dir / "pbo_summary.csv", index=False)

    chart_paths = make_charts(
        charts_dir,
        summary,
        equity,
        returns,
        monthly_returns,
        yearly_returns,
        rolling,
        regime,
        sector_exposure,
        turnover_yearly,
        validation["pbo_splits"],
        args.rolling_months,
    )
    write_report(
        output_dir,
        chart_paths,
        summary,
        yearly_pass,
        rolling_pass,
        regime,
        validation,
        common_start,
        args,
        len(constituents),
        len(eligible),
    )
    write_summary(
        output_dir,
        chart_paths,
        summary,
        yearly_pass,
        rolling_pass,
        regime,
        validation,
        common_start,
        args,
        len(constituents),
        len(eligible),
    )
    write_metadata(output_dir, args, start, end, common_start, len(constituents), len(eligible), len(summary))

    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))
    print(rolling_pass.to_string(index=False))
    print(validation["pbo_summary"].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate momentum hold-band finalists across periods and regimes.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--comparison-start", default="2017-07-31")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--min-years", type=float, default=7.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--skip", type=int, default=21)
    parser.add_argument("--rebalance", default="ME")
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--rolling-months", type=int, default=36)
    parser.add_argument("--pbo-slices", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/momentum_finalist_validation")
    return parser.parse_args()


def build_finalist_weights(
    eligible_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    scores: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    base = {
        "momentum_12_1_top30": momentum_weights(
            eligible_close,
            lookback=args.lookback,
            skip=args.skip,
            top_n=args.top_n,
            rebalance=args.rebalance,
        ),
        "eligible_equal_weight": equal_weight_weights(eligible_close, args.rebalance),
    }
    for hold_rank in [45, 60, 75, 90, 120]:
        base[f"momentum_hold{hold_rank}"] = rank_weights_from_scores(
            eligible_close,
            scores,
            top_n=args.top_n,
            rebalance=args.rebalance,
            hold_until_rank=hold_rank,
        )
    base["momentum_hold75_sma200_50"] = apply_momentum_overlay(
        base["momentum_hold75"],
        eligible_close,
        benchmark_close,
        "sma200_50",
    )
    return base


def build_runs(
    eligible_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    weights: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> list[StrategyRun]:
    strategy_order = [
        "momentum_hold75",
        "momentum_hold120",
        "momentum_hold90",
        "momentum_12_1_top30",
        "momentum_hold75_sma200_50",
        "momentum_hold60",
        "momentum_hold45",
        "eligible_equal_weight",
    ]
    runs = [
        simulate_portfolio(eligible_close, weights[name], name, cost_bps=args.cost_bps)
        for name in strategy_order
        if name in weights
    ]
    runs.extend(buy_hold_run(benchmark_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in benchmark_close.columns)
    return runs


def add_validation_notes(summary: pd.DataFrame) -> pd.DataFrame:
    note_map = {
        "momentum_hold75": "finalist",
        "momentum_hold120": "low_turnover_finalist",
        "momentum_hold90": "prior_finalist",
        "momentum_12_1_top30": "baseline_momentum",
        "momentum_hold75_sma200_50": "drawdown_overlay",
        "eligible_equal_weight": "current_universe_equal_weight",
        "SPY": "benchmark",
        "QQQ": "benchmark",
        "RSP": "benchmark",
    }
    frame = summary.copy()
    frame["validation_role"] = frame["strategy"].map(note_map).fillna("hold_band_candidate")
    return frame


def common_active_start(runs: list[StrategyRun]) -> pd.Timestamp:
    starts = []
    for run in runs:
        active = run.weights.sum(axis=1).gt(0)
        if active.any():
            starts.append(pd.Timestamp(active.idxmax()))
    if not starts:
        raise RuntimeError("No active strategy runs.")
    return max(starts)


def compound_period_returns(returns: pd.DataFrame, frequency: str) -> pd.DataFrame:
    period = returns.resample(frequency).apply(lambda values: (1 + values).prod() - 1)
    return period.dropna(how="all")


def period_metrics(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    frequency: str,
    risk_free_rate: float,
) -> pd.DataFrame:
    records = []
    for period_end, period_returns in returns.resample(frequency):
        period_returns = period_returns.dropna(how="all")
        if len(period_returns) < 60:
            continue
        period_turnover = turnover.reindex(period_returns.index).fillna(0.0)
        for strategy in period_returns.columns:
            series = period_returns[strategy].dropna()
            if len(series) < 60:
                continue
            equity = (1 + series).cumprod()
            tseries = period_turnover[strategy] if strategy in period_turnover else pd.Series(0.0, index=series.index)
            metrics = performance_metrics(series, equity, int((tseries > 0).sum()), risk_free_rate=risk_free_rate)
            records.append(
                {
                    "period_end": pd.Timestamp(period_end).date().isoformat(),
                    "year": int(pd.Timestamp(period_end).year),
                    "strategy": strategy,
                    **metrics,
                    "turnover": float(tseries.sum()),
                    "observations": int(len(series)),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    benchmark_returns = frame.loc[frame["strategy"].eq("SPY"), ["year", "total_return"]].rename(
        columns={"total_return": "spy_total_return"}
    )
    baseline_returns = frame.loc[frame["strategy"].eq("momentum_12_1_top30"), ["year", "total_return"]].rename(
        columns={"total_return": "baseline_total_return"}
    )
    frame = frame.merge(benchmark_returns, on="year", how="left").merge(baseline_returns, on="year", how="left")
    frame["beat_spy"] = frame["total_return"].gt(frame["spy_total_return"])
    frame["beat_baseline_momentum"] = frame["total_return"].gt(frame["baseline_total_return"])
    return frame


def pass_rate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    focus = metrics.loc[metrics["strategy"].isin(FOCUS_STRATEGIES)].copy()
    return (
        focus.groupby("strategy")
        .agg(
            windows=("year", "count"),
            positive_rate=("total_return", lambda values: float((values > 0).mean())),
            beat_spy_rate=("beat_spy", "mean"),
            beat_baseline_rate=("beat_baseline_momentum", "mean"),
            median_return=("total_return", "median"),
            worst_return=("total_return", "min"),
            median_sharpe=("sharpe", "median"),
            worst_drawdown=("max_drawdown", "min"),
        )
        .reset_index()
        .sort_values(["beat_baseline_rate", "median_sharpe"], ascending=False)
    )


def rolling_window_metrics(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    window_months: int,
    risk_free_rate: float,
) -> pd.DataFrame:
    monthly = compound_period_returns(returns, "ME").dropna(how="any")
    if len(monthly) < window_months:
        return pd.DataFrame()
    records = []
    for end_pos in range(window_months - 1, len(monthly)):
        start_date = pd.Timestamp(monthly.index[end_pos - window_months + 1])
        end_date = pd.Timestamp(monthly.index[end_pos])
        daily_slice = returns.loc[start_date:end_date].dropna(how="all")
        turnover_slice = turnover.reindex(daily_slice.index).fillna(0.0)
        if len(daily_slice) < int(window_months * 15):
            continue
        row_by_strategy: dict[str, dict[str, float | str | int | bool]] = {}
        for strategy in daily_slice.columns:
            series = daily_slice[strategy].dropna()
            if len(series) < int(window_months * 15):
                continue
            equity = (1 + series).cumprod()
            tseries = turnover_slice[strategy] if strategy in turnover_slice else pd.Series(0.0, index=series.index)
            metrics = performance_metrics(series, equity, int((tseries > 0).sum()), risk_free_rate=risk_free_rate)
            row_by_strategy[strategy] = {
                "window_start": start_date.date().isoformat(),
                "window_end": end_date.date().isoformat(),
                "strategy": strategy,
                **metrics,
                "turnover": float(tseries.sum()),
                "observations": int(len(series)),
            }
        spy_return = row_by_strategy.get("SPY", {}).get("total_return", np.nan)
        baseline_return = row_by_strategy.get("momentum_12_1_top30", {}).get("total_return", np.nan)
        for row in row_by_strategy.values():
            row["spy_total_return"] = float(spy_return) if pd.notna(spy_return) else np.nan
            row["baseline_total_return"] = float(baseline_return) if pd.notna(baseline_return) else np.nan
            row["beat_spy"] = bool(row["total_return"] > spy_return) if pd.notna(spy_return) else False
            row["beat_baseline_momentum"] = (
                bool(row["total_return"] > baseline_return) if pd.notna(baseline_return) else False
            )
            records.append(row)
    return pd.DataFrame(records)


def rolling_pass_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    if rolling.empty:
        return pd.DataFrame()
    focus = rolling.loc[rolling["strategy"].isin(FOCUS_STRATEGIES)].copy()
    return (
        focus.groupby("strategy")
        .agg(
            windows=("window_end", "count"),
            positive_rate=("total_return", lambda values: float((values > 0).mean())),
            beat_spy_rate=("beat_spy", "mean"),
            beat_baseline_rate=("beat_baseline_momentum", "mean"),
            median_cagr=("cagr", "median"),
            p10_cagr=("cagr", lambda values: float(values.quantile(0.10))),
            median_sharpe=("sharpe", "median"),
            p10_sharpe=("sharpe", lambda values: float(values.quantile(0.10))),
            worst_drawdown=("max_drawdown", "min"),
        )
        .reset_index()
        .sort_values(["beat_baseline_rate", "median_sharpe"], ascending=False)
    )


def rolling_window_slug(months: int) -> str:
    return f"rolling_{months}m"


def rolling_window_label(months: int) -> str:
    if months % 12 == 0:
        years = months // 12
        return f"Rolling {years}-Year" if years == 1 else f"Rolling {years}-Year"
    return f"Rolling {months}-Month"


def market_regime_performance(
    returns: pd.DataFrame,
    spy_close: pd.Series,
    risk_free_rate: float,
) -> pd.DataFrame:
    spy = spy_close.reindex(returns.index).ffill()
    sma200 = spy.rolling(200).mean()
    mom63 = spy.pct_change(63, fill_method=None)
    regimes = pd.Series("unclassified", index=returns.index)
    regimes.loc[spy.gt(sma200) & mom63.ge(0)] = "risk_on_above_sma200_pos3m"
    regimes.loc[spy.gt(sma200) & mom63.lt(0)] = "pullback_above_sma200_neg3m"
    regimes.loc[spy.le(sma200) & mom63.ge(0)] = "rebound_below_sma200_pos3m"
    regimes.loc[spy.le(sma200) & mom63.lt(0)] = "risk_off_below_sma200_neg3m"
    records = []
    for regime_name, index in regimes.groupby(regimes).groups.items():
        if regime_name == "unclassified":
            continue
        subset = returns.loc[index].dropna(how="all")
        if len(subset) < 20:
            continue
        for strategy in subset.columns:
            series = subset[strategy].dropna()
            if len(series) < 20:
                continue
            equity = (1 + series).cumprod()
            metrics = performance_metrics(series, equity, trades=0, risk_free_rate=risk_free_rate)
            records.append(
                {
                    "regime": regime_name,
                    "strategy": strategy,
                    "observations": int(len(series)),
                    "total_return": metrics["total_return"],
                    "cagr_like": metrics["cagr"],
                    "volatility": metrics["volatility"],
                    "sharpe": metrics["sharpe"],
                    "hit_rate": float(series.gt(0).mean()),
                    "mean_daily_return": float(series.mean()),
                }
            )
    return pd.DataFrame(records)


def sector_exposure_table(weights: dict[str, pd.DataFrame], sector_map: dict[str, str]) -> pd.DataFrame:
    records = []
    for strategy in ["momentum_hold75", "momentum_hold120", "momentum_hold90", "momentum_12_1_top30"]:
        if strategy not in weights:
            continue
        frame = weights[strategy].copy()
        grouped = frame.T.groupby(lambda ticker: sector_map.get(str(ticker), "Unknown")).sum().T
        avg = grouped.mean().sort_values(ascending=False)
        for sector, exposure in avg.items():
            records.append({"strategy": strategy, "sector": sector, "avg_weight": float(exposure)})
    return pd.DataFrame(records)


def validation_tables(monthly_returns: pd.DataFrame, pbo_slices: int) -> dict[str, pd.DataFrame]:
    candidates = [
        column
        for column in [
            "momentum_12_1_top30",
            "momentum_hold45",
            "momentum_hold60",
            "momentum_hold75",
            "momentum_hold90",
            "momentum_hold120",
            "momentum_hold75_sma200_50",
        ]
        if column in monthly_returns.columns
    ]
    candidate_monthly = monthly_returns[candidates].dropna(how="any")
    pbo_splits = cscv_pbo(candidate_monthly, pbo_slices)
    pbo_summary = pd.DataFrame(
        [
            {
                "candidate_count": len(candidates),
                "monthly_observations": len(candidate_monthly),
                "pbo": float(pbo_splits["is_overfit"].mean()) if not pbo_splits.empty else np.nan,
                "median_selected_oos_rank_pct": float(pbo_splits["selected_oos_rank_pct"].median())
                if not pbo_splits.empty
                else np.nan,
                "splits": int(len(pbo_splits)),
            }
        ]
    )
    return {
        "deflated_sharpe": deflated_sharpe_table(candidate_monthly, n_trials=len(candidates)),
        "pbo_splits": pbo_splits,
        "pbo_summary": pbo_summary,
    }


def make_charts(
    charts_dir: Path,
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    returns: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    yearly_returns: pd.DataFrame,
    rolling: pd.DataFrame,
    regime: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    turnover_yearly: pd.DataFrame,
    pbo_splits: pd.DataFrame,
    rolling_months: int,
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
    paths = [
        plot_equity_drawdown(charts_dir, equity),
        plot_annual_returns(charts_dir, yearly_returns),
        plot_monthly_heatmap(charts_dir, monthly_returns, "momentum_hold75"),
        plot_rolling_metrics(charts_dir, rolling, rolling_months),
        plot_rolling_scatter(charts_dir, rolling, rolling_months),
        plot_regime_performance(charts_dir, regime),
        plot_turnover(charts_dir, turnover_yearly),
        plot_sector_exposure(charts_dir, sector_exposure),
        plot_pbo(charts_dir, pbo_splits),
        plot_leaderboard(charts_dir, summary),
    ]
    for path in paths:
        if path.exists() and path.stat().st_size <= 0:
            raise RuntimeError(f"Chart was created but is empty: {path}")
    return paths


def plot_equity_drawdown(charts_dir: Path, equity: pd.DataFrame) -> Path:
    columns = [column for column in FOCUS_STRATEGIES if column in equity.columns]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for column in columns:
        color = PLOT_COLORS.get(column)
        lw = 2.6 if column == "momentum_hold75" else 1.5
        axes[0].plot(equity.index, equity[column], label=column, color=color, linewidth=lw)
        drawdown = equity[column] / equity[column].cummax() - 1
        axes[1].plot(drawdown.index, drawdown, label=column, color=color, linewidth=lw)
    axes[0].set_title("Finalist Equity Curves")
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_title("Drawdowns")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = charts_dir / "finalist_equity_drawdown.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_annual_returns(charts_dir: Path, yearly_returns: pd.DataFrame) -> Path:
    columns = [column for column in ["momentum_hold75", "momentum_hold120", "momentum_12_1_top30", "SPY", "QQQ"] if column in yearly_returns]
    frame = yearly_returns[columns].copy()
    frame.index = pd.DatetimeIndex(frame.index).year
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(frame.index))
    width = 0.82 / max(len(columns), 1)
    for idx, column in enumerate(columns):
        ax.bar(x + idx * width - 0.41 + width / 2, frame[column], width=width, label=column, color=PLOT_COLORS.get(column))
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(frame.index.astype(str), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.set_title("Calendar-Year Returns")
    ax.set_ylabel("Return")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = charts_dir / "annual_returns.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_monthly_heatmap(charts_dir: Path, monthly_returns: pd.DataFrame, strategy: str) -> Path:
    series = monthly_returns[strategy].dropna()
    table = series.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="return").sort_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanquantile(np.abs(data), 0.90)
    image = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Monthly Return Heatmap: {strategy}")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=7, color="#111827")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    fig.tight_layout()
    path = charts_dir / "monthly_heatmap_momentum_hold75.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rolling_metrics(charts_dir: Path, rolling: pd.DataFrame, rolling_months: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    if not rolling.empty:
        focus = rolling.loc[rolling["strategy"].isin(["momentum_hold75", "momentum_hold120", "momentum_12_1_top30", "SPY", "QQQ"])].copy()
        focus["window_end_dt"] = pd.to_datetime(focus["window_end"])
        for strategy, group in focus.groupby("strategy"):
            group = group.sort_values("window_end_dt")
            color = PLOT_COLORS.get(strategy)
            lw = 2.6 if strategy == "momentum_hold75" else 1.5
            axes[0].plot(group["window_end_dt"], group["cagr"], label=strategy, color=color, linewidth=lw)
            axes[1].plot(group["window_end_dt"], group["sharpe"], label=strategy, color=color, linewidth=lw)
    label = rolling_window_label(rolling_months)
    axes[0].set_title(f"{label} CAGR")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    axes[1].set_title(f"{label} Sharpe")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = charts_dir / f"{rolling_window_slug(rolling_months)}_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rolling_scatter(charts_dir: Path, rolling: pd.DataFrame, rolling_months: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    if not rolling.empty:
        focus = rolling.loc[rolling["strategy"].isin(["momentum_hold75", "momentum_hold120", "momentum_12_1_top30", "SPY", "QQQ"])].copy()
        for strategy, group in focus.groupby("strategy"):
            ax.scatter(
                group["max_drawdown"],
                group["cagr"],
                label=strategy,
                color=PLOT_COLORS.get(strategy),
                alpha=0.65,
                s=28,
            )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title(f"{rolling_window_label(rolling_months)} Windows: CAGR vs Max Drawdown")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("CAGR")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = charts_dir / f"{rolling_window_slug(rolling_months)}_return_drawdown_scatter.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regime_performance(charts_dir: Path, regime: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    if not regime.empty:
        focus = regime.loc[regime["strategy"].isin(["momentum_hold75", "momentum_hold120", "momentum_12_1_top30", "SPY", "QQQ"])].copy()
        regimes = focus["regime"].drop_duplicates().tolist()
        strategies = ["momentum_hold75", "momentum_hold120", "momentum_12_1_top30", "SPY", "QQQ"]
        x = np.arange(len(regimes))
        width = 0.82 / len(strategies)
        for idx, strategy in enumerate(strategies):
            values = (
                focus.loc[focus["strategy"].eq(strategy)]
                .set_index("regime")
                .reindex(regimes)["cagr_like"]
                .fillna(0.0)
            )
            ax.bar(x + idx * width - 0.41 + width / 2, values, width=width, label=strategy, color=PLOT_COLORS.get(strategy))
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace("_", "\n") for label in regimes], fontsize=8)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.set_title("Conditional Performance by SPY Regime")
    ax.set_ylabel("Annualized return-like metric")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = charts_dir / "regime_performance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_turnover(charts_dir: Path, turnover_yearly: pd.DataFrame) -> Path:
    columns = [column for column in ["momentum_hold75", "momentum_hold120", "momentum_hold90", "momentum_12_1_top30"] if column in turnover_yearly]
    frame = turnover_yearly[columns].copy()
    frame.index = pd.DatetimeIndex(frame.index).year
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for column in columns:
        ax.plot(frame.index, frame[column], marker="o", label=column, color=PLOT_COLORS.get(column), linewidth=2)
    ax.set_title("Annual Turnover")
    ax.set_ylabel("One-way turnover sum")
    ax.set_xticks(frame.index)
    ax.set_xticklabels(frame.index.astype(str), rotation=45, ha="right")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = charts_dir / "annual_turnover.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sector_exposure(charts_dir: Path, sector_exposure: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    if not sector_exposure.empty:
        pivot = sector_exposure.pivot(index="strategy", columns="sector", values="avg_weight").fillna(0.0)
        pivot = pivot.reindex(["momentum_hold75", "momentum_hold120", "momentum_hold90", "momentum_12_1_top30"]).dropna(how="all")
        order = pivot.mean().sort_values(ascending=False).index
        pivot = pivot[order]
        bottom = np.zeros(len(pivot))
        palette = plt.get_cmap("tab20").colors
        for idx, sector in enumerate(pivot.columns):
            values = pivot[sector].to_numpy()
            ax.bar(pivot.index, values, bottom=bottom, label=sector, color=palette[idx % len(palette)])
            bottom += values
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        ax.legend(ncol=3, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.set_title("Average Sector Exposure")
    ax.set_ylabel("Average portfolio weight")
    fig.tight_layout()
    path = charts_dir / "average_sector_exposure.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pbo(charts_dir: Path, pbo_splits: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if not pbo_splits.empty:
        axes[0].hist(pbo_splits["selected_oos_rank_pct"], bins=np.linspace(0, 1, 11), color="#2563EB", edgecolor="white")
        selected_counts = pbo_splits["selected_strategy"].value_counts().sort_values(ascending=True)
        axes[1].barh(selected_counts.index, selected_counts.values, color="#0F766E")
    axes[0].axvline(0.5, color="#DC2626", linestyle="--", linewidth=1.2)
    axes[0].set_title("CSCV OOS Rank Distribution")
    axes[0].set_xlabel("Selected strategy OOS rank percentile")
    axes[0].set_ylabel("Split count")
    axes[1].set_title("Strategy Selected In-Sample")
    axes[1].set_xlabel("Split count")
    fig.tight_layout()
    path = charts_dir / "cscv_pbo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_leaderboard(charts_dir: Path, summary: pd.DataFrame) -> Path:
    focus = summary.loc[summary["strategy"].isin(FOCUS_STRATEGIES + ["momentum_hold75_sma200_50"])].copy()
    focus = focus.sort_values("sharpe", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(focus["strategy"], focus["sharpe"], color=[PLOT_COLORS.get(value, "#6B7280") for value in focus["strategy"]])
    ax.set_title("Full-Period Sharpe Leaderboard")
    ax.set_xlabel("Sharpe")
    fig.tight_layout()
    path = charts_dir / "full_period_sharpe_leaderboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    output_dir: Path,
    chart_paths: list[Path],
    summary: pd.DataFrame,
    yearly_pass: pd.DataFrame,
    rolling_pass: pd.DataFrame,
    regime: pd.DataFrame,
    validation: dict[str, pd.DataFrame],
    common_start: pd.Timestamp,
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
  <title>Momentum Finalist Validation</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #111827; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    h1, h2 {{ margin-bottom: 8px; }}
    p {{ color: #4B5563; }}
    img {{ width: 100%; height: auto; border: 1px solid #E5E7EB; }}
    figure {{ margin: 24px 0; }}
    figcaption {{ color: #6B7280; font-size: 13px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #E5E7EB; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>Momentum Finalist Validation</h1>
  <p>Comparison start {common_start.date().isoformat()}; current S&P 500 constituents {constituent_count}; eligible {eligible_count}; one-way costs {args.cost_bps:g} bps.</p>
  <p>Universe caveat: current constituents only; removed and delisted historical members are missing.</p>
  <h2>Charts</h2>
  {image_tags}
  <h2>Full-Period Summary</h2>
  {summary.head(20).to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Calendar-Year Pass Rates</h2>
  {yearly_pass.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>{rolling_window_label(args.rolling_months)} Pass Rates</h2>
  {rolling_pass.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Regime Performance</h2>
  {regime.head(80).to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>Deflated Sharpe</h2>
  {validation["deflated_sharpe"].to_html(index=False, float_format=lambda value: f"{value:.4f}")}
  <h2>PBO Summary</h2>
  {validation["pbo_summary"].to_html(index=False, float_format=lambda value: f"{value:.4f}")}
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_summary(
    output_dir: Path,
    chart_paths: list[Path],
    summary: pd.DataFrame,
    yearly_pass: pd.DataFrame,
    rolling_pass: pd.DataFrame,
    regime: pd.DataFrame,
    validation: dict[str, pd.DataFrame],
    common_start: pd.Timestamp,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
) -> None:
    chart_list = "\n".join(f"- `{path.relative_to(output_dir)}`" for path in chart_paths)
    text = f"""# Momentum Finalist Validation

Comparison start: {common_start.date().isoformat()}

Universe: current S&P 500 constituents {constituent_count}; eligible tickers {eligible_count}.

Costs: {args.cost_bps:g} bps one-way. Rebalance: {args.rebalance}. Rolling window: {args.rolling_months} months.

Universe caveat: current constituents only, so delisted and removed historical members are missing.

## Full-Period Summary

{markdown_table(summary.head(20))}

## Calendar-Year Pass Rates

{markdown_table(yearly_pass)}

## {rolling_window_label(args.rolling_months)} Pass Rates

{markdown_table(rolling_pass)}

## Regime Performance

{markdown_table(regime.head(80))}

## Deflated Sharpe

{markdown_table(validation["deflated_sharpe"])}

## PBO Summary

{markdown_table(validation["pbo_summary"])}

## Charts

{chart_list}

## CSV Artifacts

- `finalist_summary.csv`
- `daily_returns.csv`
- `equity_curves.csv`
- `turnover.csv`
- `monthly_returns.csv`
- `yearly_returns.csv`
- `yearly_metrics.csv`
- `yearly_pass_rates.csv`
- `{rolling_window_slug(args.rolling_months)}_windows.csv`
- `{rolling_window_slug(args.rolling_months)}_pass_rates.csv`
- `regime_performance.csv`
- `average_sector_exposure.csv`
- `yearly_turnover.csv`
- `deflated_sharpe.csv`
- `pbo_splits.csv`
- `pbo_summary.csv`
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
    run_count: int,
) -> None:
    payload = {
        "input_start": start.date().isoformat(),
        "input_end": end.date().isoformat(),
        "comparison_start": common_start.date().isoformat(),
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "run_count": run_count,
        "cost_bps": args.cost_bps,
        "top_n": args.top_n,
        "lookback": args.lookback,
        "skip": args.skip,
        "rebalance": args.rebalance,
        "rolling_months": args.rolling_months,
        "survivorship_bias_warning": "Uses current S&P 500 constituents across the full backtest.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
