from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from stockresearchmarket.strategies.ml_ranker import (
    FEATURE_COLUMNS,
    MLRankerResult,
    StrategyRun,
    build_ml_ranker_weights,
    buy_hold_run,
    equal_weight_weights,
    make_price_feature_panel,
    momentum_weights,
    rebalance_dates,
    simulate_portfolio,
    summarize_runs,
)

MOMENTUM_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "mom_12_1", "sma50_gap", "sma200_gap", "rsi14"]
RISK_MOMENTUM_FEATURES = [
    "ret_3m",
    "ret_6m",
    "mom_12_1",
    "vol_1m",
    "vol_3m",
    "downside_vol_3m",
    "drawdown_6m",
    "sma50_gap",
]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    family: str
    top_n: int
    train_years: int
    min_train_rows: int
    feature_columns: list[str]
    weighting: str
    model_factory: Any
    risk_overlay: str = "none"


@dataclass(frozen=True)
class VariantRun:
    variant: Variant
    ml_result: MLRankerResult
    run: StrategyRun
    first_active_date: pd.Timestamp


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
        raise RuntimeError(f"Only {len(eligible)} eligible tickers; sweep needs a broader universe.")

    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns]].dropna(how="all").ffill(limit=5)
    dates = rebalance_dates(eligible_close.index, args.rebalance)
    feature_panel = make_price_feature_panel(eligible_close, dates, args.horizon_days)
    feature_panel.to_parquet(output_dir / "feature_panel.parquet")

    variants = build_variants(args.random_state)
    if not args.include_slow_models:
        variants = [variant for variant in variants if variant.family not in {"random_forest", "mlp_adam"}]
    selected_variants = variants[: args.max_variants] if args.max_variants else variants
    variant_runs = [
        run_variant(
            variant,
            eligible_close,
            benchmark_close,
            feature_panel,
            args,
        )
        for variant in selected_variants
    ]
    common_start = max(run.first_active_date for run in variant_runs)

    strategy_runs = [_trim_run(run.run, common_start) for run in variant_runs]
    benchmark_runs = build_benchmark_runs(eligible_close, benchmark_close, args, common_start)
    all_runs = strategy_runs + benchmark_runs
    summary = summarize_runs(all_runs, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="SPY")
    summary = add_variant_metadata(summary, variant_runs, common_start)

    equity = pd.concat([run.equity for run in all_runs], axis=1).sort_index()
    returns = pd.concat([run.returns for run in all_runs], axis=1).sort_index()
    windows = collect_windows(variant_runs)
    importances = collect_feature_importances(variant_runs)

    bootstrap = bootstrap_monthly_returns(returns, args.bootstrap_runs, args.random_state)
    universe_mc = run_universe_monte_carlo(
        eligible,
        eligible_close,
        benchmark_close,
        feature_panel,
        variant_runs,
        common_start,
        args,
    )

    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    equity.to_csv(output_dir / "equity_curves.csv")
    returns.to_csv(output_dir / "daily_returns.csv")
    windows.to_csv(output_dir / "variant_windows.csv", index=False)
    importances.to_csv(output_dir / "feature_importance_by_variant.csv", index=False)
    bootstrap.to_csv(output_dir / "monte_carlo_monthly_bootstrap.csv", index=False)
    universe_mc.to_csv(output_dir / "monte_carlo_universe_subsets.csv", index=False)
    write_report(output_dir, summary, equity, returns, bootstrap, universe_mc, common_start, args, len(eligible))
    write_summary(output_dir, summary, bootstrap, universe_mc, common_start, args, len(eligible))
    write_metadata(output_dir, args, start, end, common_start, len(constituents), len(eligible), len(selected_variants))

    print(f"Output: {output_dir}")
    print(summary.head(20).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ML stock-ranker variants and Monte Carlo robustness checks.")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--min-years", type=float, default=7.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.15)
    parser.add_argument("--horizon-days", type=int, default=21)
    parser.add_argument("--rebalance", default="ME")
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--bootstrap-runs", type=int, default=1_000)
    parser.add_argument("--universe-mc-runs", type=int, default=6)
    parser.add_argument("--universe-mc-fraction", type=float, default=0.70)
    parser.add_argument("--universe-mc-variants", type=int, default=3)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--include-slow-models", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/ml_ranker_sweep")
    return parser.parse_args()


def build_variants(seed: int) -> list[Variant]:
    return [
        Variant("et_t20_5y_leaf40_eq", "extra_trees", 20, 5, 10_000, FEATURE_COLUMNS, "equal", extra_trees(25, 40, seed)),
        Variant("et_t30_5y_leaf40_eq", "extra_trees", 30, 5, 10_000, FEATURE_COLUMNS, "equal", extra_trees(25, 40, seed)),
        Variant("et_t50_5y_leaf40_eq", "extra_trees", 50, 5, 10_000, FEATURE_COLUMNS, "equal", extra_trees(25, 40, seed)),
        Variant("et_t30_3y_leaf40_eq", "extra_trees", 30, 3, 6_000, FEATURE_COLUMNS, "equal", extra_trees(25, 40, seed)),
        Variant("et_t30_7y_leaf40_eq", "extra_trees", 30, 7, 14_000, FEATURE_COLUMNS, "equal", extra_trees(25, 40, seed)),
        Variant("et_t30_5y_leaf80_eq", "extra_trees", 30, 5, 10_000, FEATURE_COLUMNS, "equal", extra_trees(25, 80, seed)),
        Variant("et_t30_5y_leaf40_invvol", "extra_trees", 30, 5, 10_000, FEATURE_COLUMNS, "inverse_vol", extra_trees(25, 40, seed)),
        Variant(
            "et_t30_5y_leaf40_rankinv",
            "extra_trees",
            30,
            5,
            10_000,
            FEATURE_COLUMNS,
            "rank_inverse_vol",
            extra_trees(25, 40, seed),
        ),
        Variant(
            "et_t30_5y_riskmom_rankinv",
            "extra_trees",
            30,
            5,
            10_000,
            RISK_MOMENTUM_FEATURES,
            "rank_inverse_vol",
            extra_trees(25, 40, seed),
        ),
        Variant("rf_t30_5y_leaf60_eq", "random_forest", 30, 5, 10_000, FEATURE_COLUMNS, "equal", random_forest(50, 60, seed)),
        Variant("hgb_t30_lr03_leaf15", "hist_gradient_boosting", 30, 5, 10_000, FEATURE_COLUMNS, "equal", hgb(0.03, 15, seed)),
        Variant("hgb_t30_lr06_leaf31", "hist_gradient_boosting", 30, 5, 10_000, FEATURE_COLUMNS, "equal", hgb(0.06, 31, seed)),
        Variant("hgb_t30_riskmom_rankinv", "hist_gradient_boosting", 30, 5, 10_000, RISK_MOMENTUM_FEATURES, "rank_inverse_vol", hgb(0.05, 15, seed)),
        Variant("sgd_huber_t30_all", "sgd_gradient_descent", 30, 5, 10_000, FEATURE_COLUMNS, "equal", sgd("huber", seed)),
        Variant(
            "sgd_sq_t30_momentum",
            "sgd_gradient_descent",
            30,
            5,
            10_000,
            MOMENTUM_FEATURES,
            "rank",
            sgd("squared_error", seed),
        ),
        Variant("mlp_adam_t30_all", "mlp_adam", 30, 5, 10_000, FEATURE_COLUMNS, "equal", mlp(seed)),
        Variant(
            "et_t30_5y_market_sma200",
            "extra_trees",
            30,
            5,
            10_000,
            FEATURE_COLUMNS,
            "equal",
            extra_trees(25, 40, seed),
            "market_sma200",
        ),
        Variant(
            "et_t30_5y_voltarget20",
            "extra_trees",
            30,
            5,
            10_000,
            FEATURE_COLUMNS,
            "equal",
            extra_trees(25, 40, seed),
            "voltarget20",
        ),
        Variant(
            "et_t30_5y_voltarget15",
            "extra_trees",
            30,
            5,
            10_000,
            FEATURE_COLUMNS,
            "equal",
            extra_trees(25, 40, seed),
            "voltarget15",
        ),
        Variant(
            "hgb_t30_voltarget20",
            "hist_gradient_boosting",
            30,
            5,
            10_000,
            FEATURE_COLUMNS,
            "equal",
            hgb(0.05, 15, seed),
            "voltarget20",
        ),
    ]


def extra_trees(n_estimators: int, min_samples_leaf: int, seed: int) -> Any:
    def factory(random_state: int) -> Any:
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=0.8,
            random_state=seed + random_state,
            n_jobs=-1,
        )

    return factory


def random_forest(n_estimators: int, min_samples_leaf: int, seed: int) -> Any:
    def factory(random_state: int) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=0.8,
            random_state=seed + random_state,
            n_jobs=-1,
        )

    return factory


def hgb(learning_rate: float, max_leaf_nodes: int, seed: int) -> Any:
    def factory(random_state: int) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=60,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=seed + random_state,
        )

    return factory


def sgd(loss: str, seed: int) -> Any:
    def factory(random_state: int) -> Any:
        from sklearn.linear_model import SGDRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SGDRegressor(
                        loss=loss,
                        penalty="elasticnet",
                        alpha=0.0003,
                        l1_ratio=0.15,
                        max_iter=2_000,
                        tol=1e-4,
                        random_state=seed + random_state,
                    ),
                ),
            ]
        )

    return factory


def mlp(seed: int) -> Any:
    def factory(random_state: int) -> Any:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(24,),
                        activation="relu",
                        alpha=0.01,
                        learning_rate_init=0.001,
                        max_iter=140,
                        early_stopping=True,
                        random_state=seed + random_state,
                    ),
                ),
            ]
        )

    return factory


def run_variant(
    variant: Variant,
    eligible_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    feature_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> VariantRun:
    print(f"Running {variant.variant_id}", flush=True)
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
    weights = apply_risk_overlay(ml_result.weights, eligible_close, benchmark_close, variant.risk_overlay)
    first_active = _first_active_weight_date(weights)
    run = simulate_portfolio(eligible_close, weights, variant.variant_id, cost_bps=args.cost_bps)
    return VariantRun(variant=variant, ml_result=ml_result, run=run, first_active_date=first_active)


def apply_risk_overlay(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    overlay: str,
) -> pd.DataFrame:
    if overlay == "none":
        return weights
    if overlay == "market_sma200":
        if "SPY" not in benchmark_close:
            return weights
        spy = benchmark_close["SPY"].reindex(weights.index).ffill()
        risk_on = spy.gt(spy.rolling(200).mean()).astype(float).reindex(weights.index).ffill().fillna(0.0)
        return weights.mul(risk_on, axis=0)
    if overlay.startswith("voltarget"):
        target = float(overlay.replace("voltarget", "")) / 100
        asset_returns = close.pct_change(fill_method=None).fillna(0.0)
        effective = weights.shift(1).fillna(0.0)
        unscaled_returns = (effective * asset_returns).sum(axis=1)
        realized_vol = unscaled_returns.rolling(63).std(ddof=0) * np.sqrt(252)
        scale = (target / realized_vol.replace(0, np.nan)).clip(upper=1.0).reindex(weights.index).ffill().fillna(1.0)
        return weights.mul(scale, axis=0)
    raise ValueError(f"Unsupported risk overlay: {overlay}")


def build_benchmark_runs(
    eligible_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    args: argparse.Namespace,
    common_start: pd.Timestamp,
) -> list[StrategyRun]:
    benchmark_runs = [
        simulate_portfolio(
            eligible_close,
            equal_weight_weights(eligible_close, args.rebalance),
            "eligible_equal_weight",
            cost_bps=args.cost_bps,
        ),
        simulate_portfolio(
            eligible_close,
            momentum_weights(eligible_close, lookback=252, skip=21, top_n=30, rebalance=args.rebalance),
            "momentum_12_1_top30",
            cost_bps=args.cost_bps,
        ),
    ]
    benchmark_runs.extend(
        buy_hold_run(benchmark_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in benchmark_close.columns
    )
    return [_trim_run(run, common_start) for run in benchmark_runs]


def add_variant_metadata(summary: pd.DataFrame, variant_runs: list[VariantRun], common_start: pd.Timestamp) -> pd.DataFrame:
    metadata = pd.DataFrame(
        [
            {
                "strategy": run.variant.variant_id,
                "family": run.variant.family,
                "top_n": run.variant.top_n,
                "train_years": run.variant.train_years,
                "weighting": run.variant.weighting,
                "risk_overlay": run.variant.risk_overlay,
                "feature_count": len(run.variant.feature_columns),
                "first_active_date": run.first_active_date.date().isoformat(),
                "comparison_start": common_start.date().isoformat(),
            }
            for run in variant_runs
        ]
    )
    return summary.merge(metadata, how="left", on="strategy")


def collect_windows(variant_runs: list[VariantRun]) -> pd.DataFrame:
    frames = []
    for run in variant_runs:
        frame = run.ml_result.walk_forward_windows.copy()
        frame["variant_id"] = run.variant.variant_id
        frame["family"] = run.variant.family
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_feature_importances(variant_runs: list[VariantRun]) -> pd.DataFrame:
    frames = []
    for run in variant_runs:
        frame = run.ml_result.feature_importance.copy()
        frame["variant_id"] = run.variant.variant_id
        frame["family"] = run.variant.family
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bootstrap_monthly_returns(returns: pd.DataFrame, runs: int, seed: int) -> pd.DataFrame:
    if runs <= 0 or returns.empty:
        return pd.DataFrame()
    monthly = returns.resample("ME").apply(lambda values: (1 + values).prod() - 1).dropna(how="any")
    if monthly.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = []
    index = np.arange(len(monthly))
    for run_id in range(runs):
        sample = monthly.iloc[rng.choice(index, size=len(index), replace=True)]
        equity = (1 + sample).cumprod()
        total = equity.iloc[-1] - 1
        drawdown = equity.div(equity.cummax()) - 1
        cagr = (1 + total).pow(12 / len(sample)) - 1
        sharpe = sample.mean().div(sample.std(ddof=0).replace(0, np.nan)) * np.sqrt(12)
        for column in monthly.columns:
            spy_col = benchmark_column(column, monthly.columns, "SPY")
            momentum_col = benchmark_column(column, monthly.columns, "momentum_12_1_top30")
            records.append(
                {
                    "run_id": run_id,
                    "strategy": column,
                    "total_return": float(total[column]),
                    "cagr": float(cagr[column]),
                    "sharpe": float(sharpe[column]) if pd.notna(sharpe[column]) else np.nan,
                    "max_drawdown": float(drawdown[column].min()),
                    "beats_spy": bool(total[column] > total.get(spy_col, np.nan)) if spy_col else False,
                    "beats_momentum": bool(total[column] > total.get(momentum_col, np.nan)) if momentum_col else False,
                }
            )
    raw = pd.DataFrame(records)
    summary = (
        raw.groupby("strategy")
        .agg(
            mc_median_total_return=("total_return", "median"),
            mc_p05_total_return=("total_return", lambda values: values.quantile(0.05)),
            mc_p95_total_return=("total_return", lambda values: values.quantile(0.95)),
            mc_median_cagr=("cagr", "median"),
            mc_p05_max_drawdown=("max_drawdown", lambda values: values.quantile(0.05)),
            mc_prob_beats_spy=("beats_spy", "mean"),
            mc_prob_beats_momentum=("beats_momentum", "mean"),
        )
        .reset_index()
    )
    return summary


def benchmark_column(strategy: str, columns: pd.Index, base_name: str) -> str | None:
    if base_name in columns:
        return base_name
    cost_pos = strategy.rfind("_cost")
    if cost_pos >= 0:
        cost_suffix = strategy[cost_pos:]
        cost_matched = f"{base_name}{cost_suffix}"
        if cost_matched in columns:
            return cost_matched
    prefix = f"{base_name}_cost"
    for column in columns:
        if str(column).startswith(prefix):
            return str(column)
    return None


def run_universe_monte_carlo(
    eligible: list[str],
    eligible_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    feature_panel: pd.DataFrame,
    variant_runs: list[VariantRun],
    common_start: pd.Timestamp,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.universe_mc_runs <= 0:
        return pd.DataFrame()
    leaderboard = summarize_runs([_trim_run(run.run, common_start) for run in variant_runs], risk_free_rate=args.risk_free_rate)
    top_ids = leaderboard["strategy"].head(args.universe_mc_variants).tolist()
    variants = [run.variant for run in variant_runs if run.variant.variant_id in set(top_ids)]
    rng = np.random.default_rng(args.random_state + 101)
    subset_size = max(50, int(len(eligible) * args.universe_mc_fraction))
    records = []
    for mc_id in range(args.universe_mc_runs):
        subset = sorted(rng.choice(eligible, size=subset_size, replace=False).tolist())
        subset_close = eligible_close[subset]
        subset_panel = feature_panel.loc[feature_panel["ticker"].isin(subset)].copy()
        for variant in variants:
            result = build_ml_ranker_weights(
                subset_close,
                top_n=min(variant.top_n, max(5, len(subset) // 5)),
                train_years=variant.train_years,
                rebalance=args.rebalance,
                horizon_days=args.horizon_days,
                min_train_rows=effective_min_train_rows(variant, subset_close),
                random_state=args.random_state + mc_id,
                model_factory=variant.model_factory,
                feature_columns=variant.feature_columns,
                feature_panel=subset_panel,
                weighting=variant.weighting,
            )
            weights = apply_risk_overlay(result.weights, subset_close, benchmark_close, variant.risk_overlay)
            run = _trim_run(simulate_portfolio(subset_close, weights, variant.variant_id, cost_bps=args.cost_bps), common_start)
            row = summarize_runs([run], risk_free_rate=args.risk_free_rate).iloc[0].to_dict()
            records.append({"mc_id": mc_id, "subset_size": subset_size, **row})
    return pd.DataFrame(records)


def effective_min_train_rows(variant: Variant, close: pd.DataFrame) -> int:
    scaled_floor = max(500, int(len(close.columns) * max(18, variant.train_years * 8)))
    return min(variant.min_train_rows, scaled_floor)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    returns: pd.DataFrame,
    bootstrap: pd.DataFrame,
    universe_mc: pd.DataFrame,
    common_start: pd.Timestamp,
    args: argparse.Namespace,
    eligible_count: int,
) -> None:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=("Top Equity Curves", "Drawdowns", "CAGR Leaderboard", "Bootstrap Median Total Return"),
        row_heights=[0.38, 0.22, 0.20, 0.20],
    )
    top = summary["strategy"].head(10).tolist()
    for column in [col for col in top + ["SPY", "QQQ", "momentum_12_1_top30"] if col in equity.columns]:
        width = 2.8 if column == top[0] else 1.6
        dash = "dash" if column in {"SPY", "QQQ", "momentum_12_1_top30"} else None
        fig.add_trace(go.Scatter(x=equity.index, y=equity[column], name=column, line={"width": width, "dash": dash}), row=1, col=1)
        dd = equity[column] / equity[column].cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"{column} DD", showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=summary["strategy"].head(20), y=summary["cagr"].head(20), name="CAGR"), row=3, col=1)
    if not bootstrap.empty:
        boot = bootstrap.set_index("strategy").reindex(summary["strategy"].head(20)).dropna(how="all").reset_index()
        fig.add_trace(go.Bar(x=boot["strategy"], y=boot["mc_median_total_return"], name="MC median total"), row=4, col=1)
    fig.update_layout(template="plotly_white", height=1250, title="ML Ranker Sweep", hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    top_table = summary.head(25).to_html(index=False, float_format=lambda value: f"{value:.4f}")
    mc_table = bootstrap.head(25).to_html(index=False, float_format=lambda value: f"{value:.4f}") if not bootstrap.empty else ""
    universe_table = universe_mc.head(30).to_html(index=False, float_format=lambda value: f"{value:.4f}") if not universe_mc.empty else ""
    header = f"""
    <section style="font-family:Inter,Arial,sans-serif;max-width:1280px;margin:24px auto 8px;">
      <h1 style="margin:0 0 8px;">ML Ranker Sweep</h1>
      <p style="margin:0;color:#4b5563;">Comparison start {common_start.date()}; eligible tickers {eligible_count}; one-way costs {args.cost_bps} bps.</p>
      <p style="color:#4b5563;">Current-constituent universe, so this is exploration with survivorship bias.</p>
    </section>
    """
    html = header + fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "<section style='font-family:Inter,Arial,sans-serif;max-width:1280px;margin:20px auto;'>"
    html += "<h2>Leaderboard</h2>" + top_table
    html += "<h2>Monthly Bootstrap</h2>" + mc_table
    html += "<h2>Universe Subset Monte Carlo</h2>" + universe_table
    html += "</section>"
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_summary(
    output_dir: Path,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    universe_mc: pd.DataFrame,
    common_start: pd.Timestamp,
    args: argparse.Namespace,
    eligible_count: int,
) -> None:
    top = summary.head(12)
    boot = bootstrap.set_index("strategy").reindex(top["strategy"]).reset_index() if not bootstrap.empty else pd.DataFrame()
    text = f"""# ML Ranker Sweep

Comparison start: {common_start.date().isoformat()}

Eligible current S&P 500 tickers: {eligible_count}

Costs: {args.cost_bps:.2f} bps one-way. Bootstrap runs: {args.bootstrap_runs}. Universe MC runs: {args.universe_mc_runs}.

Universe caveat: current constituents only, so delisted and removed historical members are missing.

## Leaderboard

{markdown_table(top)}

## Monthly Bootstrap Summary

{markdown_table(boot)}

## Universe Subset Monte Carlo

{markdown_table(universe_mc)}

## Artifacts

- `report.html`
- `variant_summary.csv`
- `equity_curves.csv`
- `daily_returns.csv`
- `variant_windows.csv`
- `feature_importance_by_variant.csv`
- `monte_carlo_monthly_bootstrap.csv`
- `monte_carlo_universe_subsets.csv`
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
    variant_count: int,
) -> None:
    payload = {
        "input_start": start.date().isoformat(),
        "input_end": end.date().isoformat(),
        "comparison_start": common_start.date().isoformat(),
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "variant_count": variant_count,
        "cost_bps_one_way": args.cost_bps,
        "bootstrap_runs": args.bootstrap_runs,
        "universe_mc_runs": args.universe_mc_runs,
        "universe_mc_fraction": args.universe_mc_fraction,
        "survivorship_bias_warning": "Uses current S&P 500 constituents across the full backtest.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
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


if __name__ == "__main__":
    main()
