from __future__ import annotations

import argparse
import itertools
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

from scripts.run_ml_ranker_sweep import (
    apply_risk_overlay,
    bootstrap_monthly_returns,
    effective_min_train_rows,
    extra_trees,
)
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
    StrategyRun,
    build_ml_ranker_weights,
    buy_hold_run,
    equal_weight_weights,
    make_price_feature_panel,
    momentum_weights,
    rebalance_dates,
    sector_neutral_selection,
    simulate_portfolio,
    summarize_runs,
)


@dataclass(frozen=True)
class HypothesisResult:
    hypothesis: str
    run: StrategyRun
    first_active_date: pd.Timestamp
    notes: dict[str, Any]


@dataclass(frozen=True)
class SimpleVariant:
    train_years: int
    min_train_rows: int


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError(f"Only {len(eligible)} eligible tickers; hypothesis suite needs a broader universe.")

    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns]].dropna(how="all").ffill(limit=5)
    dates = rebalance_dates(eligible_close.index, args.rebalance)
    feature_panel = make_price_feature_panel(eligible_close, dates, args.horizon_days)
    feature_panel.to_parquet(output_dir / "feature_panel.parquet")

    results: list[HypothesisResult] = []
    results.extend(run_baselines(eligible_close, benchmark_close, args))
    results.extend(run_momentum_filter_hypotheses(eligible_close, benchmark_close, args))
    results.extend(run_turnover_band_hypotheses(eligible_close, benchmark_close, args))
    results.extend(run_sector_neutral_hypotheses(eligible_close, benchmark_close, sector_map, feature_panel, args))
    results.extend(run_meta_label_hypotheses(eligible_close, feature_panel, args))
    results.extend(run_risk_adjusted_target_hypotheses(eligible_close, feature_panel, args))

    common_start = max(result.first_active_date for result in results)
    trimmed = [HypothesisResult(result.hypothesis, _trim_run(result.run, common_start), common_start, result.notes) for result in results]
    runs = [result.run for result in trimmed]
    summary = summarize_runs(runs, risk_free_rate=args.risk_free_rate)
    summary = add_alpha_columns(summary, benchmark="SPY")
    summary = add_hypothesis_notes(summary, trimmed)
    equity = pd.concat([run.equity for run in runs], axis=1).sort_index()
    returns = pd.concat([run.returns for run in runs], axis=1).sort_index()
    bootstrap = bootstrap_monthly_returns(returns, args.bootstrap_runs, args.random_state)
    validation = validate_overfitting(returns, summary, args)

    summary.to_csv(output_dir / "hypothesis_summary.csv", index=False)
    equity.to_csv(output_dir / "equity_curves.csv")
    returns.to_csv(output_dir / "daily_returns.csv")
    bootstrap.to_csv(output_dir / "monte_carlo_monthly_bootstrap.csv", index=False)
    validation["deflated_sharpe"].to_csv(output_dir / "deflated_sharpe.csv", index=False)
    validation["pbo_splits"].to_csv(output_dir / "pbo_splits.csv", index=False)
    validation["pbo_summary"].to_csv(output_dir / "pbo_summary.csv", index=False)
    write_report(output_dir, summary, equity, returns, bootstrap, validation, common_start, args, len(eligible))
    write_summary(output_dir, summary, bootstrap, validation, common_start, args, len(eligible))
    write_metadata(output_dir, args, start, end, common_start, len(constituents), len(eligible), len(results))

    print(f"Output: {output_dir}")
    print(summary.head(30).to_string(index=False))
    print(validation["pbo_summary"].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential ML/momentum improvement hypotheses.")
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
    parser.add_argument("--random-state", type=int, default=23)
    parser.add_argument("--bootstrap-runs", type=int, default=1_000)
    parser.add_argument("--pbo-slices", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/ml_hypothesis_suite")
    return parser.parse_args()


def run_baselines(
    eligible_close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    args: argparse.Namespace,
) -> list[HypothesisResult]:
    print("Running baselines", flush=True)
    runs = [
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
    runs.extend(buy_hold_run(benchmark_close[ticker], ticker, cost_bps=args.cost_bps) for ticker in benchmark_close.columns)
    return [
        HypothesisResult("baseline", run, _first_active_weight_date(run.weights), {"hypothesis": "baseline"})
        for run in runs
    ]


def run_momentum_filter_hypotheses(
    close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    args: argparse.Namespace,
) -> list[HypothesisResult]:
    print("Running H1 market-filtered momentum", flush=True)
    base = momentum_weights(close, lookback=252, skip=21, top_n=30, rebalance=args.rebalance)
    variants = {
        "momentum_sma200_filter": apply_risk_overlay(base, close, benchmark_close, "market_sma200"),
        "momentum_voltarget20": apply_risk_overlay(base, close, benchmark_close, "voltarget20"),
    }
    return [
        HypothesisResult(
            "H1_market_filter",
            simulate_portfolio(close, weights, name, cost_bps=args.cost_bps),
            _first_active_weight_date(weights),
            {"hypothesis": "momentum market/risk filter"},
        )
        for name, weights in variants.items()
    ]


def run_turnover_band_hypotheses(
    close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    args: argparse.Namespace,
) -> list[HypothesisResult]:
    print("Running H2 hold-until-rank turnover bands", flush=True)
    scores = momentum_scores(close)
    variants = {
        "momentum_hold60": rank_weights_from_scores(close, scores, top_n=30, rebalance=args.rebalance, hold_until_rank=60),
        "momentum_hold90": rank_weights_from_scores(close, scores, top_n=30, rebalance=args.rebalance, hold_until_rank=90),
    }
    variants["momentum_hold60_sma200"] = apply_risk_overlay(variants["momentum_hold60"], close, benchmark_close, "market_sma200")
    return [
        HypothesisResult(
            "H2_turnover_band",
            simulate_portfolio(close, weights, name, cost_bps=args.cost_bps),
            _first_active_weight_date(weights),
            {"hypothesis": "hold until rank", "hold_until_rank": name.split("hold")[-1].split("_")[0]},
        )
        for name, weights in variants.items()
    ]


def run_sector_neutral_hypotheses(
    close: pd.DataFrame,
    benchmark_close: pd.DataFrame,
    sector_map: dict[str, str],
    feature_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> list[HypothesisResult]:
    print("Running H4 sector-neutral variants", flush=True)
    scores = momentum_scores(close)
    momentum_sector = sector_neutral_weights_from_scores(close, scores, sector_map, top_n=33, rebalance=args.rebalance)
    ml = build_extra_trees_ranker(close, feature_panel, args, train_years=5, top_n=30)
    ml_sector = sector_neutral_weights_from_predictions(close, ml.predictions, sector_map, top_n=33)
    variants = {
        "momentum_sector_neutral": momentum_sector,
        "ml_et5y_sector_neutral": ml_sector,
    }
    return [
        HypothesisResult(
            "H4_sector_neutral",
            simulate_portfolio(close, weights, name, cost_bps=args.cost_bps),
            _first_active_weight_date(weights),
            {"hypothesis": "sector neutral"},
        )
        for name, weights in variants.items()
    ]


def run_meta_label_hypotheses(
    close: pd.DataFrame,
    feature_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> list[HypothesisResult]:
    print("Running H3 meta-labeling momentum candidates", flush=True)
    variants = {
        "meta_et_mom90_top30": build_meta_label_weights(
            close,
            feature_panel,
            candidate_pool=90,
            top_n=30,
            threshold=None,
            args=args,
        ),
        "meta_et_mom90_prob55": build_meta_label_weights(
            close,
            feature_panel,
            candidate_pool=90,
            top_n=30,
            threshold=0.55,
            args=args,
        ),
    }
    return [
        HypothesisResult(
            "H3_meta_label",
            simulate_portfolio(close, weights, name, cost_bps=args.cost_bps),
            _first_active_weight_date(weights),
            {"hypothesis": "meta label momentum", "candidate_pool": 90},
        )
        for name, weights in variants.items()
    ]


def run_risk_adjusted_target_hypotheses(
    close: pd.DataFrame,
    feature_panel: pd.DataFrame,
    args: argparse.Namespace,
) -> list[HypothesisResult]:
    print("Running H5 risk-adjusted ML labels", flush=True)
    risk_panel = feature_panel.copy()
    risk_target = risk_panel["future_return"].div(risk_panel["vol_3m"].clip(lower=0.05))
    risk_panel["target_excess"] = risk_target - risk_target.groupby(risk_panel["date"]).transform("median")
    ml_risk = build_extra_trees_ranker(close, risk_panel, args, train_years=5, top_n=30)
    weights = ml_risk.weights
    return [
        HypothesisResult(
            "H5_risk_adjusted_label",
            simulate_portfolio(close, weights, "ml_et5y_risk_adjusted_target", cost_bps=args.cost_bps),
            _first_active_weight_date(weights),
            {"hypothesis": "risk-adjusted target"},
        )
    ]


def build_extra_trees_ranker(
    close: pd.DataFrame,
    feature_panel: pd.DataFrame,
    args: argparse.Namespace,
    *,
    train_years: int,
    top_n: int,
) -> Any:
    variant = SimpleVariant(train_years=train_years, min_train_rows=10_000)
    return build_ml_ranker_weights(
        close,
        top_n=top_n,
        train_years=train_years,
        rebalance=args.rebalance,
        horizon_days=args.horizon_days,
        min_train_rows=effective_min_train_rows(variant, close),
        random_state=args.random_state,
        model_factory=extra_trees(25, 40, args.random_state),
        feature_columns=FEATURE_COLUMNS,
        feature_panel=feature_panel,
        weighting="equal",
    )


def momentum_scores(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    close = close.sort_index().ffill(limit=5)
    return close.shift(skip).div(close.shift(skip + lookback)) - 1


def rank_weights_from_scores(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    top_n: int,
    rebalance: str,
    hold_until_rank: int | None = None,
) -> pd.DataFrame:
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    current_holdings: list[str] = []
    for date in rebalance_dates(close.index, rebalance):
        weights.loc[date] = 0.0
        score = scores.loc[date].dropna().sort_values(ascending=False)
        if score.empty:
            continue
        if hold_until_rank and current_holdings:
            ranks = score.rank(ascending=False, method="first")
            keep = [ticker for ticker in current_holdings if ticker in ranks and ranks[ticker] <= hold_until_rank]
        else:
            keep = []
        fill = [ticker for ticker in score.index if ticker not in keep][: max(0, top_n - len(keep))]
        selected = (keep + fill)[:top_n]
        current_holdings = selected
        if selected:
            weights.loc[date, selected] = 1 / len(selected)
    return weights.ffill().fillna(0.0)


def sector_neutral_weights_from_scores(
    close: pd.DataFrame,
    scores: pd.DataFrame,
    sector_map: dict[str, str],
    *,
    top_n: int,
    rebalance: str,
) -> pd.DataFrame:
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for date in rebalance_dates(close.index, rebalance):
        score = scores.loc[date].dropna().sort_values(ascending=False)
        weights.loc[date] = sector_neutral_selection(score, sector_map, top_n)
    return weights.ffill().fillna(0.0)


def sector_neutral_weights_from_predictions(
    close: pd.DataFrame,
    predictions: pd.DataFrame,
    sector_map: dict[str, str],
    *,
    top_n: int,
) -> pd.DataFrame:
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    if predictions.empty:
        return weights.fillna(0.0)
    for date, group in predictions.groupby("date"):
        score = group.set_index("ticker")["prediction"].dropna().sort_values(ascending=False)
        weights.loc[pd.Timestamp(date)] = sector_neutral_selection(score, sector_map, top_n)
    return weights.ffill().fillna(0.0)


def build_meta_label_weights(
    close: pd.DataFrame,
    feature_panel: pd.DataFrame,
    *,
    candidate_pool: int,
    top_n: int,
    threshold: float | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesClassifier

    scores = momentum_scores(close)
    score_panel = scores.reindex(pd.DatetimeIndex(feature_panel["date"].unique())).stack(future_stack=True).rename("momentum_score")
    score_panel.index = score_panel.index.set_names(["date", "ticker"])
    panel = feature_panel.merge(score_panel.reset_index(), on=["date", "ticker"], how="left")
    panel["momentum_rank"] = panel.groupby("date")["momentum_score"].rank(ascending=False, method="first")
    panel["momentum_rank_pct"] = panel.groupby("date")["momentum_rank"].rank(pct=True)
    panel["meta_label"] = panel["target_excess"].gt(0).astype(int)
    features = FEATURE_COLUMNS + ["momentum_rank_pct"]
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    first_valid_feature_date = panel.dropna(subset=features)["date"].min()

    for date in rebalance_dates(close.index, args.rebalance):
        weights.loc[date] = 0.0
        if pd.isna(first_valid_feature_date) or date < first_valid_feature_date + pd.DateOffset(years=5):
            continue
        current = panel.loc[
            panel["date"].eq(date) & panel["momentum_rank"].le(candidate_pool)
        ].dropna(subset=features)
        train_start = date - pd.DateOffset(years=5)
        train = panel.loc[
            panel["date"].ge(train_start)
            & panel["date"].lt(date)
            & panel["label_end"].notna()
            & panel["label_end"].le(date)
            & panel["momentum_rank"].le(candidate_pool)
        ].dropna(subset=features + ["target_excess", "meta_label"])
        min_train_rows = max(500, int(candidate_pool * 35))
        if len(current) < top_n or len(train) < min_train_rows:
            continue
        model = ExtraTreesClassifier(
            n_estimators=25,
            min_samples_leaf=40,
            max_features=0.8,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(train[features], train["meta_label"])
        current = current.copy()
        current["probability"] = model.predict_proba(current[features])[:, 1]
        ranked = current.sort_values(["probability", "momentum_score"], ascending=False)
        if threshold is not None:
            ranked = ranked.loc[ranked["probability"].ge(threshold)]
        selected = ranked.head(top_n)["ticker"].astype(str).tolist()
        if selected:
            weights.loc[date, selected] = 1 / len(selected)
    return weights.ffill().fillna(0.0)


def add_hypothesis_notes(summary: pd.DataFrame, results: list[HypothesisResult]) -> pd.DataFrame:
    notes = pd.DataFrame(
        [
            {
                "strategy": result.run.name,
                "hypothesis": result.hypothesis,
                **{f"note_{key}": value for key, value in result.notes.items()},
            }
            for result in results
        ]
    )
    return summary.merge(notes, on="strategy", how="left")


def validate_overfitting(returns: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    monthly = returns.resample("ME").apply(lambda values: (1 + values).prod() - 1).dropna(how="any")
    candidates = [strategy for strategy in summary["strategy"].tolist() if strategy not in {"SPY", "QQQ", "RSP"}]
    candidates = [strategy for strategy in candidates if strategy in monthly.columns]
    deflated = deflated_sharpe_table(monthly[candidates], n_trials=len(candidates))
    pbo_splits = cscv_pbo(monthly[candidates], n_slices=args.pbo_slices)
    pbo_summary = pd.DataFrame(
        [
            {
                "candidate_count": len(candidates),
                "monthly_observations": len(monthly),
                "pbo": float(pbo_splits["is_overfit"].mean()) if not pbo_splits.empty else np.nan,
                "median_selected_oos_rank_pct": float(pbo_splits["selected_oos_rank_pct"].median())
                if not pbo_splits.empty
                else np.nan,
                "splits": int(len(pbo_splits)),
            }
        ]
    )
    return {"deflated_sharpe": deflated, "pbo_splits": pbo_splits, "pbo_summary": pbo_summary}


def deflated_sharpe_table(monthly: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    from scipy.stats import kurtosis, norm, skew

    rows = []
    gamma = 0.5772156649
    n_trials = max(n_trials, 2)
    for column in monthly.columns:
        values = monthly[column].dropna()
        if len(values) < 12 or values.std(ddof=0) <= 1e-12:
            continue
        sr_monthly = float(values.mean() / values.std(ddof=0))
        sr_annual = sr_monthly * np.sqrt(12)
        skewness = float(skew(values, bias=False))
        kurt = float(kurtosis(values, fisher=False, bias=False))
        denominator = np.sqrt(max(1e-12, (1 - skewness * sr_monthly + ((kurt - 1) / 4) * sr_monthly**2) / (len(values) - 1)))
        expected_max_sr = denominator * (
            (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e))
        )
        dsr_probability = float(norm.cdf((sr_monthly - expected_max_sr) / denominator))
        rows.append(
            {
                "strategy": column,
                "annual_sharpe": sr_annual,
                "monthly_sharpe": sr_monthly,
                "expected_max_monthly_sharpe_under_null": expected_max_sr,
                "deflated_sharpe_probability": dsr_probability,
                "observations": int(len(values)),
                "trials": int(n_trials),
            }
        )
    return pd.DataFrame(rows).sort_values("deflated_sharpe_probability", ascending=False)


def cscv_pbo(monthly: pd.DataFrame, n_slices: int) -> pd.DataFrame:
    if monthly.empty or n_slices < 4:
        return pd.DataFrame()
    n_slices = min(n_slices, len(monthly))
    if n_slices % 2:
        n_slices -= 1
    slices = np.array_split(np.arange(len(monthly)), n_slices)
    records = []
    for train_ids in itertools.combinations(range(n_slices), n_slices // 2):
        test_ids = [idx for idx in range(n_slices) if idx not in train_ids]
        train_index = np.concatenate([slices[idx] for idx in train_ids])
        test_index = np.concatenate([slices[idx] for idx in test_ids])
        train = monthly.iloc[train_index]
        test = monthly.iloc[test_index]
        train_sharpes = train.mean().div(train.std(ddof=0).replace(0, np.nan))
        test_sharpes = test.mean().div(test.std(ddof=0).replace(0, np.nan))
        if train_sharpes.dropna().empty or test_sharpes.dropna().empty:
            continue
        selected = str(train_sharpes.idxmax())
        sorted_test = test_sharpes.rank(ascending=True, pct=True)
        oos_rank_pct = float(sorted_test[selected])
        records.append(
            {
                "selected_strategy": selected,
                "selected_train_sharpe_monthly": float(train_sharpes[selected]),
                "selected_test_sharpe_monthly": float(test_sharpes[selected]),
                "selected_oos_rank_pct": oos_rank_pct,
                "is_overfit": bool(oos_rank_pct < 0.5),
                "train_slices": ",".join(map(str, train_ids)),
                "test_slices": ",".join(map(str, test_ids)),
            }
        )
    return pd.DataFrame(records)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    returns: pd.DataFrame,
    bootstrap: pd.DataFrame,
    validation: dict[str, pd.DataFrame],
    common_start: pd.Timestamp,
    args: argparse.Namespace,
    eligible_count: int,
) -> None:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=("Top Equity Curves", "Drawdowns", "CAGR Leaderboard", "Deflated Sharpe Probability"),
        row_heights=[0.38, 0.22, 0.20, 0.20],
    )
    top = summary["strategy"].head(10).tolist()
    columns = [column for column in top + ["SPY", "QQQ", "momentum_12_1_top30"] if column in equity.columns]
    for column in columns:
        width = 2.8 if column == top[0] else 1.6
        dash = "dash" if column in {"SPY", "QQQ", "momentum_12_1_top30"} else None
        fig.add_trace(go.Scatter(x=equity.index, y=equity[column], name=column, line={"width": width, "dash": dash}), row=1, col=1)
        dd = equity[column] / equity[column].cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"{column} DD", showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=summary["strategy"].head(20), y=summary["cagr"].head(20), name="CAGR"), row=3, col=1)
    dsr = validation["deflated_sharpe"].set_index("strategy").reindex(summary["strategy"].head(20)).dropna(how="all").reset_index()
    if not dsr.empty:
        fig.add_trace(go.Bar(x=dsr["strategy"], y=dsr["deflated_sharpe_probability"], name="DSR probability"), row=4, col=1)
    fig.update_layout(template="plotly_white", height=1250, title="ML Hypothesis Suite", hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    html = f"""
    <section style="font-family:Inter,Arial,sans-serif;max-width:1280px;margin:24px auto 8px;">
      <h1 style="margin:0 0 8px;">ML Hypothesis Suite</h1>
      <p style="margin:0;color:#4b5563;">Comparison start {common_start.date()}; eligible tickers {eligible_count}; one-way costs {args.cost_bps} bps.</p>
      <p style="color:#4b5563;">Current-constituent universe; exploratory and survivorship-biased.</p>
    </section>
    """
    html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "<section style='font-family:Inter,Arial,sans-serif;max-width:1280px;margin:20px auto;'>"
    html += "<h2>Leaderboard</h2>" + summary.head(40).to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "<h2>Bootstrap</h2>" + bootstrap.head(40).to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "<h2>Deflated Sharpe</h2>" + validation["deflated_sharpe"].to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "<h2>PBO Summary</h2>" + validation["pbo_summary"].to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html += "</section>"
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def write_summary(
    output_dir: Path,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    validation: dict[str, pd.DataFrame],
    common_start: pd.Timestamp,
    args: argparse.Namespace,
    eligible_count: int,
) -> None:
    top = summary.head(20)
    boot = bootstrap.set_index("strategy").reindex(top["strategy"]).reset_index() if not bootstrap.empty else pd.DataFrame()
    text = f"""# ML Hypothesis Suite

Comparison start: {common_start.date().isoformat()}

Eligible current S&P 500 tickers: {eligible_count}

Costs: {args.cost_bps:.2f} bps one-way. Bootstrap runs: {args.bootstrap_runs}. PBO slices: {args.pbo_slices}.

Universe caveat: current constituents only, so delisted and removed historical members are missing.

## Leaderboard

{markdown_table(top)}

## Monthly Bootstrap Summary

{markdown_table(boot)}

## Deflated Sharpe

{markdown_table(validation["deflated_sharpe"])}

## PBO Summary

{markdown_table(validation["pbo_summary"])}

## Artifacts

- `report.html`
- `hypothesis_summary.csv`
- `equity_curves.csv`
- `daily_returns.csv`
- `monte_carlo_monthly_bootstrap.csv`
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
    result_count: int,
) -> None:
    payload = {
        "input_start": start.date().isoformat(),
        "input_end": end.date().isoformat(),
        "comparison_start": common_start.date().isoformat(),
        "constituent_count": constituent_count,
        "eligible_count": eligible_count,
        "result_count": result_count,
        "cost_bps_one_way": args.cost_bps,
        "bootstrap_runs": args.bootstrap_runs,
        "pbo_slices": args.pbo_slices,
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
