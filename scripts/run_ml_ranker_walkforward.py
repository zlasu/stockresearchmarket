from __future__ import annotations

import argparse
import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots

from stockresearchmarket.features.indicators import rolling_sharpe
from stockresearchmarket.strategies.ml_ranker import (
    StrategyRun,
    build_ml_ranker_weights,
    buy_hold_run,
    equal_weight_weights,
    momentum_weights,
    simulate_portfolio,
    summarize_runs,
)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BENCHMARK_TICKERS = ["SPY", "QQQ", "RSP"]


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if len(eligible) < max(args.top_n, 10):
        raise RuntimeError(f"Only {len(eligible)} eligible tickers; need at least {max(args.top_n, 10)} for this test.")

    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns]].dropna(how="all").ffill(limit=5)

    ml_result = build_ml_ranker_weights(
        eligible_close,
        top_n=args.top_n,
        train_years=args.train_years,
        rebalance=args.rebalance,
        horizon_days=args.horizon_days,
        min_train_rows=args.min_train_rows,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    ml_name = f"ml_extra_trees_top{args.top_n}"
    runs = [
        simulate_portfolio(eligible_close, ml_result.weights, ml_name, cost_bps=args.cost_bps),
        simulate_portfolio(
            eligible_close,
            equal_weight_weights(eligible_close, args.rebalance),
            "eligible_equal_weight",
            cost_bps=args.cost_bps,
        ),
        simulate_portfolio(
            eligible_close,
            momentum_weights(
                eligible_close,
                lookback=args.momentum_lookback,
                skip=args.momentum_skip,
                top_n=args.top_n,
                rebalance=args.rebalance,
            ),
            f"momentum_12_1_top{args.top_n}",
            cost_bps=args.cost_bps,
        ),
    ]
    benchmark_runs = [buy_hold_run(benchmark_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in benchmark_close.columns]
    first_ml_weight_date = _first_active_weight_date(ml_result.weights)
    all_runs = [_trim_run(run, first_ml_weight_date) for run in runs + benchmark_runs]

    summary = summarize_runs(all_runs, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="SPY")
    equity = pd.concat([run.equity for run in all_runs], axis=1).sort_index()
    returns = pd.concat([run.returns for run in all_runs], axis=1).sort_index()
    turnover = pd.concat([run.turnover for run in runs], axis=1).sort_index()

    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    equity.to_csv(output_dir / "equity_curves.csv")
    returns.to_csv(output_dir / "daily_returns.csv")
    turnover.to_csv(output_dir / "turnover.csv")
    ml_result.weights.to_csv(output_dir / f"weights_{ml_name}.csv")
    ml_result.predictions.to_csv(output_dir / "ml_predictions.csv", index=False)
    ml_result.feature_panel.to_csv(output_dir / "ml_feature_panel.csv", index=False)
    ml_result.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    ml_result.walk_forward_windows.to_csv(output_dir / "walk_forward_windows.csv", index=False)

    metadata = build_metadata(args, start, end, first_ml_weight_date, constituents, eligible, quality, ml_name)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_html_report(output_dir, summary, equity, returns, ml_result.feature_importance, metadata)
    write_markdown_summary(output_dir, summary, equity, ml_result.walk_forward_windows, metadata)

    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a monthly walk-forward ML stock-ranker test.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--max-tickers", type=int, default=0, help="0 means all eligible current S&P 500 constituents.")
    parser.add_argument("--tickers", default="", help="Optional comma-separated custom universe; bypasses Wikipedia universe.")
    parser.add_argument("--min-years", type=float, default=7.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.15)
    parser.add_argument("--horizon-days", type=int, default=21)
    parser.add_argument("--rebalance", default="ME")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--min-train-rows", type=int, default=2_000)
    parser.add_argument("--momentum-lookback", type=int, default=252)
    parser.add_argument("--momentum-skip", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/ml_ranker_walkforward")
    return parser.parse_args()


def _load_universe(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    if args.tickers:
        tickers = [ticker.strip().upper().replace(".", "-") for ticker in args.tickers.split(",") if ticker.strip()]
        table = pd.DataFrame(
            {
                "ticker": tickers,
                "yf_ticker": tickers,
                "security": tickers,
                "sector": "custom",
                "industry": "custom",
                "source_url": "user_supplied",
            }
        )
    else:
        table = fetch_current_sp500_constituents()
    table.to_csv(output_dir / "universe.csv", index=False)
    return table


def fetch_current_sp500_constituents() -> pd.DataFrame:
    response = requests.get(
        WIKIPEDIA_SP500_URL,
        headers={"User-Agent": "StockResearchMarket/0.1 ML ranker research script"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    table = tables[0].copy()
    table = table.rename(
        columns={"Symbol": "ticker", "Security": "security", "GICS Sector": "sector", "GICS Sub-Industry": "industry"}
    )
    table["yf_ticker"] = table["ticker"].astype(str).str.replace(".", "-", regex=False)
    table["source_url"] = WIKIPEDIA_SP500_URL
    return table[["ticker", "yf_ticker", "security", "sector", "industry", "source_url"]]


def load_yfinance_close(
    tickers: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    Path("data/historical").mkdir(parents=True, exist_ok=True)
    cache_path = Path("data/historical") / f"ml_ranker_yfinance_close_{start:%Y%m%d}_{end:%Y%m%d}_{len(tickers)}.parquet"
    if cache_path.exists() and not refresh:
        close = pd.read_parquet(cache_path)
    else:
        chunks = []
        for chunk in _chunks(tickers, 80):
            raw = yf.download(
                tickers=chunk,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                field = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
                chunk_close = raw[field].copy()
            else:
                chunk_close = raw[["Close"]].rename(columns={"Close": chunk[0]})
            chunks.append(chunk_close)
        if not chunks:
            raise RuntimeError("yfinance returned no close-price data.")
        close = pd.concat(chunks, axis=1).sort_index()
        close = close.loc[:, ~close.columns.duplicated()]
        close.to_parquet(cache_path)
    close.to_csv(output_dir / "close_panel.csv")
    return close


def data_quality(close: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    expected = max(len(pd.bdate_range(start, end)), 1)
    rows = []
    for ticker in close.columns:
        series = close[ticker].dropna()
        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(series)),
                "start": series.index.min().date().isoformat() if not series.empty else None,
                "end": series.index.max().date().isoformat() if not series.empty else None,
                "missing_fraction": float(1 - len(series) / expected),
                "first_close": float(series.iloc[0]) if not series.empty else np.nan,
                "last_close": float(series.iloc[-1]) if not series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_fraction", "ticker"]).reset_index(drop=True)


def choose_eligible_tickers(
    quality: pd.DataFrame,
    universe: list[str],
    *,
    min_years: float,
    max_missing_fraction: float,
    max_tickers: int,
) -> list[str]:
    min_rows = int(252 * min_years)
    frame = quality.loc[
        quality["ticker"].isin(universe)
        & quality["rows"].ge(min_rows)
        & quality["missing_fraction"].le(max_missing_fraction)
    ].sort_values(["missing_fraction", "ticker"])
    values = frame["ticker"].astype(str).tolist()
    return values[:max_tickers] if max_tickers > 0 else values


def add_alpha_columns(summary: pd.DataFrame, *, benchmark: str) -> pd.DataFrame:
    frame = summary.copy()
    if benchmark not in set(frame["strategy"]):
        return frame
    benchmark_return = float(frame.loc[frame["strategy"].eq(benchmark), "total_return"].iloc[0])
    benchmark_cagr = float(frame.loc[frame["strategy"].eq(benchmark), "cagr"].iloc[0])
    frame["alpha_total_return_vs_spy"] = frame["total_return"] - benchmark_return
    frame["alpha_cagr_vs_spy"] = frame["cagr"] - benchmark_cagr
    return frame


def _first_active_weight_date(weights: pd.DataFrame) -> pd.Timestamp:
    active = weights.sum(axis=1).gt(0)
    if not active.any():
        raise RuntimeError("ML strategy never produced an active portfolio.")
    return pd.Timestamp(active.idxmax())


def _trim_run(run: StrategyRun, start: pd.Timestamp) -> StrategyRun:
    return StrategyRun(
        name=run.name,
        returns=run.returns.loc[start:].copy(),
        weights=run.weights.loc[start:].copy(),
        turnover=run.turnover.loc[start:].copy(),
    )


def build_metadata(
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    first_ml_weight_date: pd.Timestamp,
    constituents: pd.DataFrame,
    eligible: list[str],
    quality: pd.DataFrame,
    ml_name: str,
) -> dict[str, Any]:
    return {
        "strategy": ml_name,
        "source": "current S&P 500 constituents from Wikipedia; adjusted daily close prices from yfinance",
        "source_url": WIKIPEDIA_SP500_URL if not args.tickers else "user_supplied",
        "adjustment_basis": "yfinance auto_adjust=True adjusted daily close; dividends are reflected in adjusted prices when provided by yfinance",
        "survivorship_bias_warning": "Uses the current universe over the full history; removed/delisted past constituents are not included.",
        "measured_start_input": start.date().isoformat(),
        "measured_end_input": end.date().isoformat(),
        "strategy_comparison_start": first_ml_weight_date.date().isoformat(),
        "train_years": args.train_years,
        "horizon_days": args.horizon_days,
        "rebalance": args.rebalance,
        "top_n": args.top_n,
        "cost_bps_one_way": args.cost_bps,
        "risk_free_rate": args.risk_free_rate,
        "constituent_count": int(len(constituents)),
        "eligible_count": int(len(eligible)),
        "downloaded_columns": int(len(quality)),
        "failed_or_ineligible_count": int(len(constituents) - len(eligible)),
    }


def write_html_report(
    output_dir: Path,
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    returns: pd.DataFrame,
    feature_importance: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=(
            "Normalized Equity Curves",
            "Drawdowns",
            "Rolling 63D Sharpe",
            "CAGR By Strategy",
            "ML Feature Importance",
        ),
        row_heights=[0.34, 0.18, 0.18, 0.15, 0.15],
    )
    for column in equity.columns:
        width = 2.7 if column.startswith("ml_") else 1.7
        dash = "dash" if column in BENCHMARK_TICKERS else None
        fig.add_trace(go.Scatter(x=equity.index, y=equity[column], name=column, line={"width": width, "dash": dash}), row=1, col=1)
        drawdown = equity[column] / equity[column].cummax() - 1
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name=f"{column} DD", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=returns.index, y=rolling_sharpe(returns[column]), name=f"{column} 63D Sharpe", showlegend=False), row=3, col=1)
    fig.add_trace(go.Bar(x=summary["strategy"], y=summary["cagr"], name="CAGR"), row=4, col=1)
    if not feature_importance.empty:
        fig.add_trace(
            go.Bar(
                x=feature_importance["mean_importance"],
                y=feature_importance["feature"],
                name="Mean importance",
                orientation="h",
            ),
            row=5,
            col=1,
        )
    fig.update_layout(template="plotly_white", height=1320, title="ML Stock Ranker Walk-Forward Test", hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    cards = "".join(
        _metric_card(row["strategy"], row["total_return"], row["sharpe"], row["max_drawdown"])
        for _, row in summary.head(4).iterrows()
    )
    table_html = summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    header = f"""
    <section style="font-family:Inter,Arial,sans-serif;max-width:1220px;margin:24px auto 8px;">
      <h1 style="margin:0 0 8px;">ML Stock Ranker Walk-Forward Test</h1>
      <p style="margin:0;color:#4b5563;">Comparison start {metadata["strategy_comparison_start"]}; input data {metadata["measured_start_input"]} to {metadata["measured_end_input"]}; {metadata["eligible_count"]} eligible tickers; one-way costs {metadata["cost_bps_one_way"]} bps.</p>
      <p style="color:#4b5563;">{metadata["survivorship_bias_warning"]}</p>
      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:16px;">{cards}</div>
    </section>
    """
    html = header + fig.to_html(full_html=False, include_plotlyjs="cdn") + "<section style='max-width:1220px;margin:20px auto;font-family:Inter,Arial,sans-serif;'><h2>Summary</h2>" + table_html + "</section>"
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_markdown_summary(
    output_dir: Path,
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    windows: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    period_start = equity.index.min().date().isoformat()
    period_end = equity.index.max().date().isoformat()
    tested_windows = windows.loc[windows["status"].eq("tested")] if not windows.empty else pd.DataFrame()
    pass_rate = (
        float(tested_windows["forward_spread"].gt(0).mean())
        if not tested_windows.empty and "forward_spread" in tested_windows
        else np.nan
    )
    median_spread = (
        float(tested_windows["forward_spread"].median())
        if not tested_windows.empty and "forward_spread" in tested_windows
        else np.nan
    )
    key_columns = [
        "strategy",
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "trades",
        "avg_annual_turnover",
        "alpha_total_return_vs_spy",
    ]
    text = f"""# ML Stock Ranker Walk-Forward Test

Measured daily backtest period in files: {period_start} to {period_end}

Input data period requested: {metadata["measured_start_input"]} to {metadata["measured_end_input"]}

Strategy comparison start: {metadata["strategy_comparison_start"]}

Data: {metadata["source"]}. Adjustment basis: {metadata["adjustment_basis"]}.

Universe caveat: {metadata["survivorship_bias_warning"]}

Costs: {metadata["cost_bps_one_way"]:.2f} bps one-way. Rebalance: {metadata["rebalance"]}. ML training window: {metadata["train_years"]} years. Label horizon: {metadata["horizon_days"]} trading days.

Eligible tickers: {metadata["eligible_count"]} / {metadata["constituent_count"]}.

Walk-forward window pass rate vs equal universe forward return: {pass_rate:.1%}

Median selected-minus-universe forward spread: {median_spread:.2%}

## Strategy Summary

{_markdown_table(summary[[column for column in key_columns if column in summary.columns]])}

## Artifacts

- `report.html`
- `strategy_summary.csv`
- `equity_curves.csv`
- `daily_returns.csv`
- `ml_predictions.csv`
- `feature_importance.csv`
- `walk_forward_windows.csv`
- `weights_{metadata["strategy"]}.csv`
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def _metric_card(strategy: str, total_return: float, sharpe: float, max_drawdown: float) -> str:
    return (
        "<div style='border:1px solid #dbe1ea;border-radius:8px;padding:10px;background:#fbfcfe;'>"
        f"<div style='font-size:11px;color:#586272;text-transform:uppercase;'>{strategy}</div>"
        f"<div style='font-size:22px;font-weight:700;color:#111827;margin-top:4px;'>{total_return:.1%}</div>"
        f"<div style='font-size:12px;color:#4b5563;margin-top:4px;'>Sharpe {sharpe:.2f} | DD {max_drawdown:.1%}</div>"
        "</div>"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                values.append(value.date().isoformat())
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


if __name__ == "__main__":
    main()
