from __future__ import annotations

import argparse
import itertools
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

from scripts.run_bot_candidate_chart_pack import add_qqq_alpha
from scripts.run_ml_hypothesis_suite import markdown_table, momentum_scores, rank_weights_from_scores
from scripts.run_ml_ranker_walkforward import (
    _load_universe,
    _trim_run,
    add_alpha_columns,
    choose_eligible_tickers,
    data_quality,
    load_yfinance_close,
)
from stockresearchmarket.strategies.ml_ranker import buy_hold_run, simulate_portfolio, summarize_runs


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / stamp
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()
    input_start = pd.Timestamp(args.start).normalize() if args.start else end - pd.DateOffset(years=args.years)
    measured_start = pd.Timestamp(args.measured_start).normalize() if args.measured_start else input_start
    warmup_start = min(input_start, measured_start - pd.DateOffset(years=args.warmup_years))

    strategy_run, strategy_weights, constituent_count, eligible_count = build_target_strategy(args, warmup_start, end, output_dir)
    candidate_tickers = [item.strip().upper() for item in args.etfs.split(",") if item.strip()]
    etf_close = load_yfinance_close(candidate_tickers, start=warmup_start, end=end, output_dir=output_dir, refresh=args.refresh)
    etf_quality = data_quality(etf_close, start=warmup_start, end=end)
    etf_quality.to_csv(output_dir / "etf_data_quality.csv", index=False)

    available = [ticker for ticker in candidate_tickers if ticker in etf_close.columns and not etf_close[ticker].dropna().empty]
    if len(available) < 3:
        raise RuntimeError(f"Only {len(available)} ETF candidates have usable data; need at least 3.")

    common_start = max(
        measured_start,
        pd.Timestamp(strategy_run.returns.dropna().index.min()),
        max(pd.Timestamp(etf_close[ticker].dropna().index.min()) for ticker in available),
    )
    strategy_trimmed = _trim_run(strategy_run, common_start)
    etf_close = etf_close[available].loc[etf_close.index >= common_start].dropna(how="all").ffill(limit=5)

    all_results = search_proxy_mixes(
        etf_close,
        strategy_trimmed.returns,
        args=args,
    )
    all_results.to_csv(output_dir / "all_proxy_results.csv", index=False)

    best_two = all_results.loc[all_results["etf_count"].eq(2)].sort_values("tracking_error_ann").iloc[0]
    best_three = all_results.loc[all_results["etf_count"].eq(3)].sort_values("tracking_error_ann").iloc[0]
    best_overall = all_results.sort_values("tracking_error_ann").iloc[0]
    selected = pd.DataFrame([best_two, best_three, best_overall]).drop_duplicates(subset=["combo_key"]).reset_index(drop=True)

    proxy_runs = {}
    proxy_weights = {}
    for _, row in selected.iterrows():
        mix = json.loads(row["weights_json"])
        weights = constant_mix_weights(etf_close.index, mix)
        proxy_weights[row["label"]] = weights
        proxy_runs[row["label"]] = simulate_portfolio(etf_close[list(mix)], weights, row["label"], cost_bps=args.cost_bps)

    benchmark_runs = [buy_hold_run(etf_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in ["QQQ"] if ticker in etf_close.columns]
    all_runs = [_trim_run(strategy_trimmed, common_start), *[_trim_run(run, common_start) for run in proxy_runs.values()], *[_trim_run(run, common_start) for run in benchmark_runs]]

    summary = summarize_runs(all_runs, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="QQQ" if "QQQ" in summary["strategy"].tolist() else summary.iloc[-1]["strategy"])
    if "QQQ" in summary["strategy"].tolist():
        summary = add_qqq_alpha(summary)
    summary["display_order"] = summary["strategy"].map(
        {
            strategy_trimmed.name: 0,
            best_two["label"]: 1,
            best_three["label"]: 2,
            best_overall["label"]: 3,
            "QQQ": 4,
        }
    ).fillna(99)
    summary = summary.sort_values(["display_order"]).drop(columns=["display_order"]).reset_index(drop=True)

    returns = pd.concat([run.returns for run in all_runs], axis=1).sort_index()
    equity = pd.concat([run.equity for run in all_runs], axis=1).sort_index()
    drawdown = equity.div(equity.cummax()) - 1
    rolling_gap = build_tracking_gap_frame(strategy_trimmed.returns, proxy_runs)

    selected.to_csv(output_dir / "best_proxy_candidates.csv", index=False)
    summary.to_csv(output_dir / "proxy_summary.csv", index=False)
    returns.to_csv(output_dir / "daily_returns.csv")
    equity.to_csv(output_dir / "equity_curves.csv")
    drawdown.to_csv(output_dir / "drawdowns.csv")
    rolling_gap.to_csv(output_dir / "rolling_tracking_gap.csv", index=False)
    strategy_weights.to_csv(output_dir / "target_strategy_weights.csv")
    for label, weights in proxy_weights.items():
        weights.to_csv(output_dir / f"weights_{label}.csv")

    chart_paths = make_charts(charts_dir, equity, drawdown, rolling_gap, args.zoom_starts)
    write_report(
        output_dir,
        summary,
        selected,
        chart_paths,
        all_results,
        common_start,
        input_start,
        warmup_start,
        measured_start,
        end,
        args,
        constituent_count,
        eligible_count,
        available,
    )
    write_summary(
        output_dir,
        summary,
        selected,
        chart_paths,
        all_results,
        common_start,
        input_start,
        warmup_start,
        measured_start,
        end,
        args,
        constituent_count,
        eligible_count,
        available,
    )
    write_metadata(output_dir, args, input_start, warmup_start, measured_start, end, common_start, constituent_count, eligible_count, available)

    print(f"Output: {output_dir}")
    print(selected.to_string(index=False))
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the closest 2-3 ETF proxy mix for momentum_hold120.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start", default="2011-06-13")
    parser.add_argument("--measured-start")
    parser.add_argument("--warmup-years", type=int, default=10)
    parser.add_argument("--end")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--min-years", type=float, default=7.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--hold-until-rank", type=int, default=120)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--skip", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--etfs", default="SPMO,MTUM,SOXX,SMH,QQQ,VGT")
    parser.add_argument("--zoom-starts", default="2025-01-01,2026-01-01")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/momentum_hold120_etf_proxy_mix")
    return parser.parse_args()


def build_target_strategy(
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
):
    constituents = _load_universe(args, output_dir)
    constituent_count = len(constituents)
    close = load_yfinance_close(constituents["yf_ticker"].tolist(), start=start, end=end, output_dir=output_dir, refresh=args.refresh)
    quality = data_quality(close, start=start, end=end)
    quality.to_csv(output_dir / "strategy_data_quality.csv", index=False)
    eligible = choose_eligible_tickers(
        quality,
        constituents["yf_ticker"].tolist(),
        min_years=args.min_years,
        max_missing_fraction=args.max_missing_fraction,
        max_tickers=args.max_tickers,
    )
    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    scores = momentum_scores(eligible_close, lookback=args.lookback, skip=args.skip)
    weights = rank_weights_from_scores(
        eligible_close,
        scores,
        top_n=args.top_n,
        rebalance="ME",
        hold_until_rank=args.hold_until_rank,
    )
    run = simulate_portfolio(eligible_close, weights, "momentum_hold120_target", cost_bps=args.cost_bps)
    return run, weights, constituent_count, len(eligible)


def generate_simplex_weights(count: int, step: float) -> list[tuple[float, ...]]:
    units = int(round(1 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("weight_step must divide 1.0 exactly, e.g. 0.01, 0.02, 0.05.")
    combos: list[tuple[float, ...]] = []
    for parts in itertools.product(range(1, units + 1), repeat=count - 1):
        total = sum(parts)
        if total >= units:
            continue
        last = units - total
        weights = tuple([part * step for part in parts] + [last * step])
        combos.append(weights)
    return combos


def constant_mix_weights(index: pd.DatetimeIndex, mix: dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame(index=index, columns=list(mix), dtype="float64")
    for ticker, weight in mix.items():
        frame.loc[:, ticker] = weight
    return frame.fillna(0.0)


def search_proxy_mixes(
    etf_close: pd.DataFrame,
    target_returns: pd.Series,
    *,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    step = args.weight_step
    target_returns = target_returns.loc[target_returns.index.intersection(etf_close.index)].dropna()
    etf_returns = etf_close.pct_change(fill_method=None).fillna(0.0).loc[target_returns.index]
    for size in [2, 3]:
        for combo in itertools.combinations(etf_close.columns.tolist(), size):
            combo_returns = etf_returns[list(combo)]
            for weight_tuple in generate_simplex_weights(size, step):
                mix = dict(zip(combo, weight_tuple, strict=True))
                proxy_returns = combo_returns.mul(weight_tuple, axis=1).sum(axis=1)
                diff = proxy_returns - target_returns
                tracking_error_ann = float(diff.std(ddof=0) * np.sqrt(252))
                tracking_rmse_daily = float(np.sqrt(np.mean(np.square(diff))))
                corr = float(proxy_returns.corr(target_returns))
                beta = float(proxy_returns.cov(target_returns) / target_returns.var()) if float(target_returns.var()) > 0 else np.nan
                proxy_equity = (1 + proxy_returns.fillna(0.0)).cumprod()
                target_equity = (1 + target_returns.fillna(0.0)).cumprod()
                proxy_total_return = float(proxy_equity.iloc[-1] - 1)
                target_total_return = float(target_equity.iloc[-1] - 1)
                proxy_cagr = float(proxy_equity.iloc[-1] ** (252 / max(len(proxy_returns), 1)) - 1)
                target_cagr = float(target_equity.iloc[-1] ** (252 / max(len(target_returns), 1)) - 1)
                proxy_dd = float((proxy_equity.div(proxy_equity.cummax()) - 1).min())
                target_dd = float((target_equity.div(target_equity.cummax()) - 1).min())
                rows.append(
                    {
                        "etf_count": size,
                        "combo_key": "|".join(combo),
                        "combo": " + ".join(combo),
                        "label": f"proxy_{size}etf_" + "_".join(combo),
                        "weights_json": json.dumps(mix, sort_keys=True),
                        "tracking_error_ann": tracking_error_ann,
                        "tracking_rmse_daily": tracking_rmse_daily,
                        "correlation": corr,
                        "beta_to_target": beta,
                        "total_return": proxy_total_return,
                        "target_total_return": target_total_return,
                        "total_return_gap": proxy_total_return - target_total_return,
                        "cagr": proxy_cagr,
                        "target_cagr": target_cagr,
                        "cagr_gap": proxy_cagr - target_cagr,
                        "max_drawdown": proxy_dd,
                        "target_max_drawdown": target_dd,
                        "max_drawdown_gap": proxy_dd - target_dd,
                    }
                )
    return pd.DataFrame(rows).sort_values(["tracking_error_ann", "tracking_rmse_daily", "etf_count"]).reset_index(drop=True)


def build_tracking_gap_frame(target_returns: pd.Series, proxy_runs: dict[str, object]) -> pd.DataFrame:
    rows = []
    for label, run in proxy_runs.items():
        diff = run.returns.loc[target_returns.index] - target_returns
        rolling = diff.rolling(63).std(ddof=0) * np.sqrt(252)
        for date, value in rolling.dropna().items():
            rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "strategy": label,
                    "rolling_63d_tracking_error": float(value),
                }
            )
    return pd.DataFrame(rows)


def make_charts(
    charts_dir: Path,
    equity: pd.DataFrame,
    drawdown: pd.DataFrame,
    rolling_gap: pd.DataFrame,
    zoom_starts_raw: str,
) -> list[Path]:
    chart_paths = [
        make_equity_drawdown_chart(charts_dir / "proxy_equity_drawdown.png", equity, drawdown, title_suffix="full period"),
        make_tracking_error_chart(charts_dir / "proxy_tracking_error.png", rolling_gap),
    ]
    zoom_starts = [pd.Timestamp(item.strip()) for item in zoom_starts_raw.split(",") if item.strip()]
    for zoom_start in zoom_starts:
        zoom_equity = equity.loc[equity.index >= zoom_start].copy()
        if zoom_equity.empty:
            continue
        rebased = zoom_equity.div(zoom_equity.iloc[0])
        zoom_drawdown = rebased.div(rebased.cummax()) - 1
        chart_paths.append(
            make_equity_drawdown_chart(
                charts_dir / f"proxy_equity_drawdown_{zoom_start.strftime('%Y%m%d')}.png",
                rebased,
                zoom_drawdown,
                title_suffix=f"from {zoom_equity.index[0].date().isoformat()}",
            )
        )
    return chart_paths


def make_equity_drawdown_chart(path: Path, equity: pd.DataFrame, drawdown: pd.DataFrame, *, title_suffix: str) -> Path:
    colors = {
        "momentum_hold120_target": "#111827",
        "QQQ": "#F59E0B",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, height_ratios=[3, 2])
    for column in equity.columns:
        color = colors.get(column)
        linewidth = 2.4 if column == "momentum_hold120_target" else 1.9
        axes[0].plot(equity.index, equity[column], label=column, color=color, linewidth=linewidth)
        axes[1].plot(drawdown.index, drawdown[column], label=column, color=color, linewidth=linewidth)
    axes[0].set_title(f"ETF proxy mix vs momentum_hold120 ({title_suffix})")
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


def make_tracking_error_chart(path: Path, rolling_gap: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    if not rolling_gap.empty:
        frame = rolling_gap.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        for strategy, group in frame.groupby("strategy"):
            ax.plot(group["date"], group["rolling_63d_tracking_error"], label=strategy, linewidth=1.9)
    ax.set_title("Rolling 63D annualized tracking error vs momentum_hold120")
    ax.set_ylabel("Tracking error")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    chart_paths: list[Path],
    all_results: pd.DataFrame,
    common_start: pd.Timestamp,
    input_start: pd.Timestamp,
    warmup_start: pd.Timestamp,
    measured_start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
    available: list[str],
) -> None:
    top10 = all_results.head(10).copy()
    images = "\n".join(
        f'<figure><img src="{path.relative_to(output_dir).as_posix()}" alt="{path.name}" style="width: 100%; border: 1px solid #E5E7EB;"><figcaption>{path.name}</figcaption></figure>'
        for path in chart_paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>momentum_hold120 ETF proxy mix</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #111827; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    p, li {{ color: #4B5563; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #E5E7EB; padding: 6px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>momentum_hold120 ETF proxy mix</h1>
  <p>Measured period: {common_start.date().isoformat()} to {end.date().isoformat()}. Requested measured start: {measured_start.date().isoformat()}. Warmup data: {warmup_start.date().isoformat()} to {(common_start - pd.Timedelta(days=1)).date().isoformat()}.</p>
  <p>Target strategy: current-S&amp;P-500 `momentum_hold120`, monthly rebalance, top {args.top_n}, equal weight, hold while rank &lt;= {args.hold_until_rank}, one-way cost {args.cost_bps:g} bps. Universe is survivorship-biased.</p>
  <p>ETF candidates: {", ".join(available)}. Proxy search: static 2-ETF and 3-ETF target weights, non-negative weights, {args.weight_step:.0%} grid step. Ranking objective: lowest annualized tracking error on daily returns.</p>
  <h2>Best Candidates</h2>
  {selected.to_html(index=False, float_format=lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))}
  <h2>Run Summary</h2>
  {summary.to_html(index=False, float_format=lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))}
  <h2>Top 10 Proxy Mixes</h2>
  {top10.to_html(index=False, float_format=lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))}
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
    selected: pd.DataFrame,
    chart_paths: list[Path],
    all_results: pd.DataFrame,
    common_start: pd.Timestamp,
    input_start: pd.Timestamp,
    warmup_start: pd.Timestamp,
    measured_start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
    constituent_count: int,
    eligible_count: int,
    available: list[str],
) -> None:
    chart_lines = "\n".join(f"- `{path.relative_to(output_dir)}`" for path in chart_paths)
    text = f"""# momentum_hold120 ETF proxy mix

Measured period: {common_start.date().isoformat()} to {end.date().isoformat()}.

Requested measured start: {measured_start.date().isoformat()}.

Warmup period: {warmup_start.date().isoformat()} to {(common_start - pd.Timedelta(days=1)).date().isoformat()}.

Target strategy: current-S&P-500 `momentum_hold120`, eligible {eligible_count} out of {constituent_count}, monthly rebalance, top {args.top_n}, equal weight, hold while rank <= {args.hold_until_rank}, one-way costs {args.cost_bps:g} bps.

ETF candidates tested: {", ".join(available)}.

Search assumptions:

- only 2-ETF and 3-ETF mixes
- long-only weights
- static long-only weights
- weight grid step `{args.weight_step:.0%}`
- ranking objective is lowest annualized tracking error on daily returns
- target strategy universe remains survivorship-biased

## Best Candidates

{markdown_table(selected)}

## Run Summary

{markdown_table(summary)}

## Top 10 Proxy Mixes

{markdown_table(all_results.head(10))}

## Charts

{chart_lines}
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    input_start: pd.Timestamp,
    warmup_start: pd.Timestamp,
    measured_start: pd.Timestamp,
    end: pd.Timestamp,
    common_start: pd.Timestamp,
    constituent_count: int,
    eligible_count: int,
    available: list[str],
) -> None:
    payload = {
        "input_start": input_start.date().isoformat(),
        "warmup_start": warmup_start.date().isoformat(),
        "requested_measured_start": measured_start.date().isoformat(),
        "input_end": end.date().isoformat(),
        "measured_start": common_start.date().isoformat(),
        "strategy": "momentum_hold120_target",
        "target_top_n": args.top_n,
        "target_hold_until_rank": args.hold_until_rank,
        "cost_bps": args.cost_bps,
        "candidate_etfs": available,
        "weight_step": args.weight_step,
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "survivorship_bias_warning": "Target strategy uses current S&P 500 constituents across the full backtest.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
