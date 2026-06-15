from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ml_hypothesis_suite import (
    bootstrap_monthly_returns,
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
from stockresearchmarket.strategies.ml_ranker import (
    StrategyRun,
    buy_hold_run,
    equal_weight_weights,
    momentum_weights,
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
        raise RuntimeError(f"Only {len(eligible)} eligible tickers; hold/risk sweep needs a broader universe.")

    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns]].dropna(how="all").ffill(limit=5)
    scores = momentum_scores(eligible_close, lookback=args.lookback, skip=args.skip)
    hold_ranks = parse_int_list(args.hold_ranks)
    cost_bps_values = parse_float_list(args.cost_bps_values)
    overlay_names = parse_str_list(args.overlays)

    base_weights = {
        "momentum_12_1_top30": momentum_weights(
            eligible_close,
            lookback=args.lookback,
            skip=args.skip,
            top_n=args.top_n,
            rebalance=args.rebalance,
        ),
        "eligible_equal_weight": equal_weight_weights(eligible_close, args.rebalance),
    }
    for hold_rank in hold_ranks:
        base_weights[f"momentum_hold{hold_rank}"] = rank_weights_from_scores(
            eligible_close,
            scores,
            top_n=args.top_n,
            rebalance=args.rebalance,
            hold_until_rank=hold_rank,
        )

    all_summary_rows = []
    all_daily_returns = []
    all_equity = []
    for cost_bps in cost_bps_values:
        print(f"Running cost {cost_bps:g} bps", flush=True)
        runs: list[StrategyRun] = []
        metadata: list[dict[str, Any]] = []
        for strategy_name, weights in base_weights.items():
            overlays = ["none"] if strategy_name == "eligible_equal_weight" else overlay_names
            for overlay in overlays:
                adjusted = apply_momentum_overlay(weights, eligible_close, benchmark_close, overlay)
                name = strategy_name if overlay == "none" else f"{strategy_name}_{overlay}"
                run = simulate_portfolio(eligible_close, adjusted, f"{name}_cost{cost_bps:g}bps", cost_bps=cost_bps)
                runs.append(run)
                metadata.append(
                    {
                        "strategy": run.name,
                        "base_strategy": strategy_name,
                        "overlay": overlay,
                        "cost_bps": cost_bps,
                        "hold_until_rank": extract_hold_rank(strategy_name),
                    }
                )
        for ticker in benchmark_close.columns:
            run = buy_hold_run(benchmark_close[ticker], f"{ticker}_cost{cost_bps:g}bps", cost_bps=cost_bps)
            runs.append(run)
            metadata.append(
                {
                    "strategy": run.name,
                    "base_strategy": ticker,
                    "overlay": "buy_hold",
                    "cost_bps": cost_bps,
                    "hold_until_rank": np.nan,
                }
            )

        common_start = pd.Timestamp(args.comparison_start) if args.comparison_start else common_active_start(runs)
        trimmed_runs = [_trim_run(run, common_start) for run in runs]
        summary = summarize_runs(trimmed_runs, risk_free_rate=args.risk_free_rate)
        summary = add_alpha_columns(summary, benchmark=f"SPY_cost{cost_bps:g}bps")
        summary = summary.merge(pd.DataFrame(metadata), on="strategy", how="left")
        summary["comparison_start"] = common_start.date().isoformat()
        all_summary_rows.append(summary)
        all_daily_returns.append(pd.concat([run.returns for run in trimmed_runs], axis=1).sort_index())
        all_equity.append(pd.concat([run.equity for run in trimmed_runs], axis=1).sort_index())

    summary_all = pd.concat(all_summary_rows, ignore_index=True)
    returns_all = pd.concat(all_daily_returns, axis=1).sort_index()
    equity_all = pd.concat(all_equity, axis=1).sort_index()
    bootstrap = bootstrap_monthly_returns(returns_all[summary_all["strategy"].head(args.bootstrap_top_n).tolist()], args.bootstrap_runs, args.random_state)
    dsr = deflated_sharpe_table(
        returns_all[summary_all["strategy"].head(args.dsr_top_n).tolist()]
        .resample("ME")
        .apply(lambda values: (1 + values).prod() - 1)
        .dropna(how="any"),
        n_trials=args.dsr_top_n,
    )
    cost_stress = summarize_cost_stress(summary_all)
    overlay_pivot = summarize_overlay_effects(summary_all)

    summary_all.to_csv(output_dir / "hold_risk_cost_summary.csv", index=False)
    returns_all.to_csv(output_dir / "daily_returns.csv")
    equity_all.to_csv(output_dir / "equity_curves.csv")
    bootstrap.to_csv(output_dir / "monte_carlo_monthly_bootstrap.csv", index=False)
    dsr.to_csv(output_dir / "deflated_sharpe.csv", index=False)
    cost_stress.to_csv(output_dir / "cost_stress.csv", index=False)
    overlay_pivot.to_csv(output_dir / "overlay_effects.csv", index=False)
    write_report(output_dir, summary_all, equity_all, returns_all, bootstrap, dsr, cost_stress, overlay_pivot, args, len(eligible))
    write_summary(output_dir, summary_all, bootstrap, dsr, cost_stress, overlay_pivot, args, len(eligible))
    write_metadata(output_dir, args, start, end, len(constituents), len(eligible), len(summary_all))

    print(f"Output: {output_dir}")
    print(summary_all.head(35).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test momentum hold-band finalists with costs and risk overlays.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--comparison-start", default="2017-07-31")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--min-years", type=float, default=7.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--hold-ranks", default="45,60,75,90,120")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--skip", type=int, default=21)
    parser.add_argument("--rebalance", default="ME")
    parser.add_argument("--cost-bps-values", default="3.5,7,15,25")
    parser.add_argument("--overlays", default="none,sma200_50,sma200_50_mom25,voltarget20")
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--bootstrap-runs", type=int, default=1_000)
    parser.add_argument("--bootstrap-top-n", type=int, default=24)
    parser.add_argument("--dsr-top-n", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=31)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/momentum_hold_risk_sweep")
    return parser.parse_args()


def apply_momentum_overlay(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    overlay: str,
) -> pd.DataFrame:
    if overlay == "none":
        return weights
    if "SPY" not in benchmark_close:
        return weights
    spy = benchmark_close["SPY"].reindex(weights.index).ffill()
    sma200 = spy.rolling(200).mean()
    spy_3m_momentum = spy.pct_change(63, fill_method=None)
    if overlay == "sma200_50":
        scale = pd.Series(1.0, index=weights.index)
        scale.loc[spy.lt(sma200)] = 0.5
        return weights.mul(scale.ffill().fillna(1.0), axis=0)
    if overlay == "sma200_50_mom25":
        scale = pd.Series(1.0, index=weights.index)
        below = spy.lt(sma200)
        scale.loc[below] = 0.5
        scale.loc[below & spy_3m_momentum.lt(0)] = 0.25
        return weights.mul(scale.ffill().fillna(1.0), axis=0)
    if overlay == "sma200_cash":
        scale = pd.Series(1.0, index=weights.index)
        scale.loc[spy.lt(sma200)] = 0.0
        return weights.mul(scale.ffill().fillna(1.0), axis=0)
    if overlay.startswith("voltarget"):
        target = float(overlay.replace("voltarget", "")) / 100
        asset_returns = close.pct_change(fill_method=None).fillna(0.0)
        effective = weights.shift(1).fillna(0.0)
        unscaled_returns = (effective * asset_returns).sum(axis=1)
        realized_vol = unscaled_returns.rolling(63).std(ddof=0) * np.sqrt(252)
        scale = (target / realized_vol.replace(0, np.nan)).clip(upper=1.0).reindex(weights.index).ffill().fillna(1.0)
        return weights.mul(scale, axis=0)
    raise ValueError(f"Unsupported overlay: {overlay}")


def summarize_cost_stress(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["overlay"].eq("none")].copy()
    base["base_strategy_clean"] = base["base_strategy"].astype(str)
    frame = base.pivot_table(
        index="base_strategy_clean",
        columns="cost_bps",
        values=["total_return", "cagr", "sharpe", "max_drawdown", "avg_annual_turnover"],
        aggfunc="first",
    )
    frame.columns = [f"{metric}_cost{cost:g}bps" for metric, cost in frame.columns]
    return frame.reset_index().rename(columns={"base_strategy_clean": "base_strategy"})


def summarize_overlay_effects(summary: pd.DataFrame) -> pd.DataFrame:
    focus = summary.loc[summary["cost_bps"].eq(summary["cost_bps"].min())].copy()
    columns = ["base_strategy", "overlay", "total_return", "cagr", "sharpe", "max_drawdown", "avg_gross_exposure"]
    return focus[columns].sort_values(["base_strategy", "sharpe"], ascending=[True, False]).reset_index(drop=True)


def common_active_start(runs: list[StrategyRun]) -> pd.Timestamp:
    starts = []
    for run in runs:
        active = run.weights.sum(axis=1).gt(0)
        if active.any():
            starts.append(pd.Timestamp(active.idxmax()))
    if not starts:
        raise RuntimeError("No active strategy runs.")
    return max(starts)


def extract_hold_rank(strategy_name: str) -> int | float:
    if "hold" not in strategy_name:
        return np.nan
    part = strategy_name.split("hold", maxsplit=1)[1]
    digits = "".join(char for char in part if char.isdigit())
    return int(digits) if digits else np.nan


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    returns: pd.DataFrame,
    bootstrap: pd.DataFrame,
    dsr: pd.DataFrame,
    cost_stress: pd.DataFrame,
    overlay_pivot: pd.DataFrame,
    args: argparse.Namespace,
    eligible_count: int,
) -> None:
    top = summary["strategy"].head(12).tolist()
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=("Top Equity Curves", "Drawdowns", "Sharpe By Top Variants", "Cost Stress: CAGR"),
        row_heights=[0.38, 0.22, 0.20, 0.20],
    )
    for column in top:
        if column not in equity:
            continue
        width = 2.8 if column == top[0] else 1.5
        fig.add_trace(go.Scatter(x=equity.index, y=equity[column], name=column, line={"width": width}), row=1, col=1)
        dd = equity[column] / equity[column].cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"{column} DD", showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=summary["strategy"].head(24), y=summary["sharpe"].head(24), name="Sharpe"), row=3, col=1)
    cost_cols = [column for column in cost_stress.columns if column.startswith("cagr_cost")]
    for column in cost_cols:
        fig.add_trace(go.Bar(x=cost_stress["base_strategy"], y=cost_stress[column], name=column), row=4, col=1)
    fig.update_layout(template="plotly_white", height=1250, title="Momentum Hold/Risk/Cost Sweep", hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    html = f"""
    <section style="font-family:Inter,Arial,sans-serif;max-width:1280px;margin:24px auto 8px;">
      <h1 style="margin:0 0 8px;">Momentum Hold/Risk/Cost Sweep</h1>
      <p style="margin:0;color:#4b5563;">Eligible tickers {eligible_count}; costs {args.cost_bps_values} bps; overlays {args.overlays}.</p>
      <p style="color:#4b5563;">Current-constituent universe; exploratory and survivorship-biased.</p>
    </section>
    """
    html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "<section style='font-family:Inter,Arial,sans-serif;max-width:1280px;margin:20px auto;'>"
    html += "<h2>Leaderboard</h2>" + summary.head(60).to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "<h2>Cost Stress</h2>" + cost_stress.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "<h2>Overlay Effects</h2>" + overlay_pivot.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "<h2>Bootstrap</h2>" + bootstrap.head(60).to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "<h2>Deflated Sharpe</h2>" + dsr.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "</section>"
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_summary(
    output_dir: Path,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    dsr: pd.DataFrame,
    cost_stress: pd.DataFrame,
    overlay_pivot: pd.DataFrame,
    args: argparse.Namespace,
    eligible_count: int,
) -> None:
    text = f"""# Momentum Hold/Risk/Cost Sweep

Eligible current S&P 500 tickers: {eligible_count}

Costs tested: {args.cost_bps_values} bps one-way.

Hold ranks tested: {args.hold_ranks}.

Risk overlays tested: {args.overlays}.

Universe caveat: current constituents only, so delisted and removed historical members are missing.

## Leaderboard

{markdown_table(summary.head(40))}

## Cost Stress

{markdown_table(cost_stress)}

## Overlay Effects

{markdown_table(overlay_pivot)}

## Monthly Bootstrap

{markdown_table(bootstrap.head(40))}

## Deflated Sharpe

{markdown_table(dsr)}

## Artifacts

- `report.html`
- `hold_risk_cost_summary.csv`
- `cost_stress.csv`
- `overlay_effects.csv`
- `monte_carlo_monthly_bootstrap.csv`
- `deflated_sharpe.csv`
- `equity_curves.csv`
- `daily_returns.csv`
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    constituent_count: int,
    eligible_count: int,
    run_count: int,
) -> None:
    payload = {
        "input_start": start.date().isoformat(),
        "input_end": end.date().isoformat(),
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "run_count": run_count,
        "cost_bps_values": parse_float_list(args.cost_bps_values),
        "hold_ranks": parse_int_list(args.hold_ranks),
        "overlays": parse_str_list(args.overlays),
        "survivorship_bias_warning": "Uses current S&P 500 constituents across the full backtest.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
