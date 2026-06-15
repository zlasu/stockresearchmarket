from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stockresearchmarket.engine.metrics import performance_metrics
from stockresearchmarket.features.indicators import rolling_sharpe

FEATURE_COLUMNS = [
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",
    "mom_12_1",
    "vol_1m",
    "vol_3m",
    "downside_vol_3m",
    "drawdown_6m",
    "sma50_gap",
    "sma200_gap",
    "rsi14",
]


@dataclass(frozen=True)
class StrategyRun:
    name: str
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns.fillna(0.0)).cumprod().rename(self.name)


@dataclass(frozen=True)
class MLRankerResult:
    weights: pd.DataFrame
    feature_panel: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    walk_forward_windows: pd.DataFrame


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> list[pd.Timestamp]:
    if index.empty:
        return []
    dates = pd.Series(1, index=pd.DatetimeIndex(index).sort_values()).resample(frequency).last().dropna().index
    mapped = []
    for date in dates:
        location = index.searchsorted(date, side="right") - 1
        if location >= 0:
            mapped.append(pd.Timestamp(index[location]))
    return sorted(set(mapped))


def make_price_feature_panel(
    close: pd.DataFrame,
    feature_dates: list[pd.Timestamp],
    horizon_days: int = 21,
) -> pd.DataFrame:
    close = close.sort_index().ffill(limit=5)
    returns = close.pct_change(fill_method=None)
    features = {
        "ret_1m": close.pct_change(21, fill_method=None),
        "ret_3m": close.pct_change(63, fill_method=None),
        "ret_6m": close.pct_change(126, fill_method=None),
        "ret_12m": close.pct_change(252, fill_method=None),
        "mom_12_1": close.shift(21).div(close.shift(252)) - 1,
        "vol_1m": returns.rolling(21).std(ddof=0) * np.sqrt(252),
        "vol_3m": returns.rolling(63).std(ddof=0) * np.sqrt(252),
        "downside_vol_3m": returns.clip(upper=0).rolling(63).std(ddof=0) * np.sqrt(252),
        "drawdown_6m": close.div(close.rolling(126).max()) - 1,
        "sma50_gap": close.div(close.rolling(50).mean()) - 1,
        "sma200_gap": close.div(close.rolling(200).mean()) - 1,
        "rsi14": _rsi_frame(close, 14),
    }
    future_return = close.shift(-horizon_days).div(close) - 1
    median_future_return = future_return.median(axis=1, skipna=True)
    features["future_return"] = future_return
    features["target_excess"] = future_return.sub(median_future_return, axis=0)

    valid_dates = pd.DatetimeIndex([date for date in feature_dates if date in close.index])
    pieces = []
    for name, frame in features.items():
        piece = frame.reindex(valid_dates).stack(future_stack=True).rename(name)
        pieces.append(piece)
    panel = pd.concat(pieces, axis=1)
    panel.index = panel.index.set_names(["date", "ticker"])
    panel = panel.reset_index()

    label_end_by_date: dict[pd.Timestamp, pd.Timestamp | pd.NaT] = {}
    for date in valid_dates:
        location = close.index.searchsorted(date, side="left")
        label_location = location + horizon_days
        label_end_by_date[pd.Timestamp(date)] = (
            pd.Timestamp(close.index[label_location]) if label_location < len(close.index) else pd.NaT
        )
    panel["label_end"] = panel["date"].map(label_end_by_date)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_ml_ranker_weights(
    close: pd.DataFrame,
    *,
    top_n: int = 30,
    train_years: int = 5,
    rebalance: str = "ME",
    horizon_days: int = 21,
    min_train_rows: int = 2_000,
    n_estimators: int = 200,
    min_samples_leaf: int = 40,
    random_state: int = 7,
    model_factory: Callable[[int], Any] | None = None,
    feature_columns: list[str] | None = None,
    feature_panel: pd.DataFrame | None = None,
    weighting: str = "equal",
) -> MLRankerResult:
    close = close.sort_index().ffill(limit=5)
    dates = rebalance_dates(close.index, rebalance)
    panel = feature_panel.copy() if feature_panel is not None else make_price_feature_panel(close, dates, horizon_days)
    active_features = feature_columns or FEATURE_COLUMNS
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    prediction_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    valid_feature_dates = panel.dropna(subset=active_features)["date"]
    first_valid_feature_date = valid_feature_dates.min() if not valid_feature_dates.empty else pd.NaT

    for date in dates:
        weights.loc[date] = 0.0
        current = panel.loc[panel["date"].eq(date)].dropna(subset=active_features).copy()
        if current.empty:
            window_rows.append(_skipped_window(date, "no_current_features"))
            continue

        train_start = date - pd.DateOffset(years=train_years)
        if pd.isna(first_valid_feature_date) or date < first_valid_feature_date + pd.DateOffset(years=train_years):
            window_rows.append(_skipped_window(date, "insufficient_training_years"))
            continue
        train = panel.loc[
            panel["date"].ge(train_start)
            & panel["date"].lt(date)
            & panel["label_end"].notna()
            & panel["label_end"].le(date)
        ].dropna(subset=active_features + ["target_excess"])
        if len(train) < min_train_rows:
            window_rows.append(_skipped_window(date, "insufficient_training_rows", train_rows=len(train)))
            continue

        model = (
            model_factory(random_state)
            if model_factory is not None
            else _default_extra_trees(n_estimators, min_samples_leaf, random_state)
        )
        model.fit(train[active_features], train["target_excess"])
        current["prediction"] = model.predict(current[active_features])
        selected = current.sort_values("prediction", ascending=False).head(top_n)
        if selected.empty:
            window_rows.append(_skipped_window(date, "no_selected_tickers", train_rows=len(train)))
            continue

        selected_weights = _selected_weight_series(selected, weighting)
        weights.loc[date, selected_weights.index] = selected_weights
        current["selected"] = current["ticker"].isin(selected_weights.index)
        prediction_frames.append(
            current[
                [
                    "date",
                    "ticker",
                    "prediction",
                    "selected",
                    "future_return",
                    "target_excess",
                    "label_end",
                    *active_features,
                ]
            ]
        )

        selected_realized = selected["future_return"].dropna()
        universe_realized = current["future_return"].dropna()
        window_rows.append(
            {
                "date": date,
                "status": "tested",
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "train_rows": int(len(train)),
                "candidate_count": int(len(current)),
                "selected_count": int(len(selected)),
                "mean_prediction": float(selected["prediction"].mean()),
                "selected_forward_return": float(selected_realized.mean()) if not selected_realized.empty else np.nan,
                "universe_forward_return": float(universe_realized.mean()) if not universe_realized.empty else np.nan,
                "forward_spread": (
                    float(selected_realized.mean() - universe_realized.mean())
                    if not selected_realized.empty and not universe_realized.empty
                    else np.nan
                ),
            }
        )
        for feature, importance in _model_feature_importance(model, active_features).items():
            importance_rows.append({"date": date, "feature": feature, "importance": float(importance)})

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    feature_importance = _summarize_feature_importance(pd.DataFrame(importance_rows))
    windows = pd.DataFrame(window_rows)
    return MLRankerResult(
        weights=weights.ffill().fillna(0.0),
        feature_panel=panel,
        predictions=predictions,
        feature_importance=feature_importance,
        walk_forward_windows=windows,
    )


def equal_weight_weights(close: pd.DataFrame, rebalance: str = "ME") -> pd.DataFrame:
    close = close.sort_index().ffill(limit=5)
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for date in rebalance_dates(close.index, rebalance):
        weights.loc[date] = 0.0
        available = close.loc[date].dropna().index
        if len(available):
            weights.loc[date, available] = 1 / len(available)
    return weights.ffill().fillna(0.0)


def momentum_weights(
    close: pd.DataFrame,
    *,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 30,
    rebalance: str = "ME",
) -> pd.DataFrame:
    close = close.sort_index().ffill(limit=5)
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for date in rebalance_dates(close.index, rebalance):
        weights.loc[date] = 0.0
        history = close.loc[:date]
        signal_close = history.iloc[:-skip] if skip else history
        if len(signal_close) <= lookback:
            continue
        momentum = signal_close.iloc[-1].div(signal_close.iloc[-lookback - 1]) - 1
        selected = momentum.dropna().sort_values(ascending=False).head(top_n).index
        if len(selected):
            weights.loc[date, selected] = 1 / len(selected)
    return weights.ffill().fillna(0.0)


def simulate_portfolio(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    name: str,
    *,
    cost_bps: float = 3.5,
) -> StrategyRun:
    close = close.sort_index().ffill(limit=5)
    weights = weights.reindex(close.index).ffill().fillna(0.0)
    row_sums = weights.sum(axis=1).replace(0, np.nan)
    overinvested = row_sums.gt(1.0)
    weights = weights.copy()
    weights.loc[overinvested] = weights.loc[overinvested].div(row_sums.loc[overinvested], axis=0)
    weights = weights.fillna(0.0)
    asset_returns = close.pct_change(fill_method=None).fillna(0.0)
    effective_weights = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    returns = (effective_weights * asset_returns).sum(axis=1) - turnover * (cost_bps / 10_000)
    return StrategyRun(name=name, returns=returns.rename(name), weights=weights, turnover=turnover.rename(name))


def buy_hold_run(close: pd.Series, name: str, *, cost_bps: float = 3.5) -> StrategyRun:
    close_frame = close.rename(name).to_frame()
    weights = pd.DataFrame({name: pd.Series(1.0, index=close_frame.index)})
    return simulate_portfolio(close_frame, weights, name, cost_bps=cost_bps)


def summarize_runs(
    runs: list[StrategyRun],
    *,
    annualization_days: int = 252,
    risk_free_rate: float = 0.03,
) -> pd.DataFrame:
    rows = []
    for run in runs:
        equity = run.equity
        trades = int((run.turnover > 0).sum())
        metrics = performance_metrics(run.returns, equity, trades, annualization_days, risk_free_rate)
        years = max(len(run.returns) / annualization_days, 1 / annualization_days)
        rows.append(
            {
                "strategy": run.name,
                **metrics,
                "avg_annual_turnover": float(run.turnover.sum() / years),
                "avg_gross_exposure": float(run.weights.shift(1).fillna(0.0).sum(axis=1).mean()),
                "rolling_63d_sharpe_last": float(rolling_sharpe(run.returns).dropna().iloc[-1])
                if not rolling_sharpe(run.returns).dropna().empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False).reset_index(drop=True)


def sector_neutral_selection(score: pd.Series, sector_map: dict[str, str], top_n: int) -> pd.Series:
    result = pd.Series(0.0, index=score.index)
    if score.empty:
        return result
    by_sector: dict[str, pd.Series] = {}
    for sector, tickers in score.groupby(score.index.map(lambda ticker: sector_map.get(str(ticker), "Unknown"))):
        if sector != "Unknown" and not tickers.empty:
            by_sector[str(sector)] = tickers.sort_values(ascending=False)
    if not by_sector:
        selected = score.head(top_n).index
        if len(selected):
            result.loc[selected] = 1 / len(selected)
        return result
    sectors = sorted(by_sector, key=lambda sector: by_sector[sector].iloc[0], reverse=True)
    base = top_n // len(sectors)
    remainder = top_n % len(sectors)
    selected_by_sector: dict[str, list[str]] = {}
    for idx, sector in enumerate(sectors):
        slots = base + (1 if idx < remainder else 0)
        selected_by_sector[sector] = by_sector[sector].head(max(1, slots)).index.tolist()
    active = {sector: tickers for sector, tickers in selected_by_sector.items() if tickers}
    for _sector, tickers in active.items():
        result.loc[tickers] = 1 / len(active) / len(tickers)
    return result


def _default_extra_trees(n_estimators: int, min_samples_leaf: int, random_state: int) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=0.8,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )


def _selected_weight_series(selected: pd.DataFrame, weighting: str) -> pd.Series:
    tickers = selected["ticker"].astype(str)
    if weighting == "equal":
        raw = pd.Series(1.0, index=tickers)
    elif weighting == "rank":
        raw = pd.Series(np.arange(len(selected), 0, -1, dtype=float), index=tickers)
    elif weighting == "inverse_vol":
        risk = selected["vol_3m"].replace(0, np.nan).astype(float)
        raw = pd.Series((1 / risk).to_numpy(), index=tickers)
    elif weighting == "rank_inverse_vol":
        risk = selected["vol_3m"].replace(0, np.nan).astype(float)
        rank_score = np.arange(len(selected), 0, -1, dtype=float)
        raw = pd.Series(rank_score / risk.to_numpy(), index=tickers)
    else:
        raise ValueError(f"Unsupported ML ranker weighting: {weighting}")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty or raw.sum() <= 0:
        raw = pd.Series(1.0, index=tickers)
    return raw / raw.sum()


def _model_feature_importance(model: Any, feature_columns: list[str]) -> dict[str, float]:
    estimator = getattr(model, "named_steps", {}).get("model", model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(np.asarray(estimator.coef_, dtype=float)))
    else:
        values = np.repeat(np.nan, len(feature_columns))
    if len(values) != len(feature_columns):
        values = np.resize(values, len(feature_columns))
    total = np.nansum(values)
    if total > 0:
        values = values / total
    return dict(zip(feature_columns, values, strict=True))


def _rsi_frame(close: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _skipped_window(date: pd.Timestamp, reason: str, train_rows: int = 0) -> dict[str, Any]:
    return {
        "date": date,
        "status": reason,
        "train_start": pd.NaT,
        "train_end": pd.NaT,
        "train_rows": int(train_rows),
        "candidate_count": 0,
        "selected_count": 0,
        "mean_prediction": np.nan,
        "selected_forward_return": np.nan,
        "universe_forward_return": np.nan,
        "forward_spread": np.nan,
    }


def _summarize_feature_importance(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["feature", "mean_importance", "std_importance", "windows"])
    summary = (
        raw.groupby("feature")["importance"]
        .agg(mean_importance="mean", std_importance="std", windows="count")
        .sort_values("mean_importance", ascending=False)
        .reset_index()
    )
    return summary
