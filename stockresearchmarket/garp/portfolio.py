from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockresearchmarket.features.indicators import sma
from stockresearchmarket.garp.config import get_config
from stockresearchmarket.garp.types import GarpDataBundle, PortfolioDecision, ScoreResult


def build_portfolio(
    scores: ScoreResult,
    bundle: GarpDataBundle,
    as_of_date: pd.Timestamp,
    config: dict[str, Any],
    previous_weights: pd.Series | None = None,
) -> PortfolioDecision:
    if scores.scores.empty:
        return PortfolioDecision(pd.Series(dtype=float), [], 0.0, ["No scores available."])
    top_n = int(get_config(config, "portfolio.top_n", 20))
    hold_until_rank = int(get_config(config, "portfolio.hold_until_rank", top_n))
    weighting = str(get_config(config, "portfolio.weighting", "equal"))
    ranked = scores.scores.dropna(subset=["total_score"]).sort_values("rank")
    selected = ranked.head(top_n).index.tolist()
    notes: list[str] = []

    if previous_weights is not None and not previous_weights.empty and hold_until_rank > top_n:
        keep_zone = set(ranked.head(hold_until_rank).index)
        previous_names = [ticker for ticker, weight in previous_weights.items() if ticker in keep_zone and weight > 0 and ticker != "CASH"]
        selected = list(dict.fromkeys(previous_names + selected))[:top_n]
        notes.append(f"Turnover reduction kept prior names while rank <= {hold_until_rank}.")

    selected = _apply_sector_momentum(selected, bundle, as_of_date, config, notes)
    if not selected:
        cash_asset = str(get_config(config, "data.cash_asset", "CASH")).upper()
        return PortfolioDecision(pd.Series({cash_asset: 1.0}), [], 1.0, notes + ["No selected names after filters; allocated to cash."])

    if weighting == "inverse_volatility":
        weights = _inverse_volatility_weights(selected, bundle.close, as_of_date)
    elif weighting in {"hrp", "equal_risk_contribution"}:
        weights = _inverse_volatility_weights(selected, bundle.close, as_of_date)
        notes.append("HRP/ERC requested; MVP uses inverse-volatility approximation until covariance clustering is added.")
    else:
        weights = pd.Series(1 / len(selected), index=selected, dtype=float)

    weights = _cap_position_weights(weights, float(get_config(config, "portfolio.max_position_weight", 0.08)))
    weights = _cap_sector_weights(weights, bundle.sectors, float(get_config(config, "portfolio.max_sector_weight", 0.35)))
    weights = _apply_market_filter(weights, bundle.close, as_of_date, config, notes)
    weights = _apply_volatility_target(weights, bundle.close, as_of_date, config, notes)
    weights = weights[weights.abs() > 1e-12]
    weights = weights / weights.sum() if weights.sum() > 1 else weights
    turnover = float((weights.sub(previous_weights, fill_value=0.0).abs()).sum()) if previous_weights is not None else float(weights.abs().sum())
    threshold = float(get_config(config, "portfolio.turnover_threshold", 0.0))
    if previous_weights is not None and threshold > 0 and turnover < threshold:
        notes.append(f"Skipped rebalance because turnover {turnover:.2%} < threshold {threshold:.2%}.")
        return PortfolioDecision(previous_weights, [ticker for ticker in previous_weights.index if previous_weights[ticker] > 0], 0.0, notes)
    return PortfolioDecision(weights=weights, selected=selected, turnover=turnover, notes=notes)


def _inverse_volatility_weights(selected: list[str], close: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.Series:
    vols = {}
    for ticker in selected:
        returns = close[ticker].loc[:as_of_date].pct_change().tail(63) if ticker in close.columns else pd.Series(dtype=float)
        vol = returns.std(ddof=0)
        vols[ticker] = 1 / vol if pd.notna(vol) and vol > 1e-12 else 0.0
    weights = pd.Series(vols, dtype=float)
    if weights.sum() <= 0:
        return pd.Series(1 / len(selected), index=selected, dtype=float)
    return weights / weights.sum()


def _cap_position_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    if max_weight <= 0:
        return weights
    capped = weights.clip(upper=max_weight)
    leftover = 1 - capped.sum()
    while leftover > 1e-9 and (capped < max_weight - 1e-12).any():
        room = max_weight - capped[capped < max_weight]
        add = room / room.sum() * leftover
        capped.loc[add.index] = (capped.loc[add.index] + add).clip(upper=max_weight)
        leftover = 1 - capped.sum()
    return capped / capped.sum() if capped.sum() > 0 else weights


def _cap_sector_weights(weights: pd.Series, sectors: dict[str, str], max_sector_weight: float) -> pd.Series:
    if max_sector_weight <= 0 or weights.empty:
        return weights
    adjusted = weights.copy()
    for _ in range(5):
        sector_totals = adjusted.groupby([sectors.get(ticker, "Unknown") for ticker in adjusted.index]).sum()
        offenders = sector_totals[sector_totals > max_sector_weight]
        if offenders.empty:
            break
        for sector, total in offenders.items():
            members = [ticker for ticker in adjusted.index if sectors.get(ticker, "Unknown") == sector]
            adjusted.loc[members] *= max_sector_weight / total
        adjusted /= adjusted.sum()
    return adjusted


def _apply_market_filter(weights: pd.Series, close: pd.DataFrame, as_of_date: pd.Timestamp, config: dict[str, Any], notes: list[str]) -> pd.Series:
    if not bool(get_config(config, "risk_management.market_filter.enabled", False)):
        return weights
    benchmark = str(get_config(config, "risk_management.market_filter.benchmark", "SPY")).upper()
    window = int(get_config(config, "risk_management.market_filter.sma_window", 200))
    cash_asset = str(get_config(config, "risk_management.market_filter.cash_asset", "CASH")).upper()
    if benchmark not in close.columns:
        notes.append("Market filter skipped: benchmark close unavailable.")
        return weights
    series = close[benchmark].loc[:as_of_date].dropna()
    if len(series) < window or pd.isna(sma(series, window).iloc[-1]):
        notes.append("Market filter skipped: insufficient SMA history.")
        return weights
    if series.iloc[-1] < sma(series, window).iloc[-1]:
        defensive_weight = float(get_config(config, "risk_management.market_filter.defensive_weight", 1.0))
        risky_scale = max(0.0, 1 - defensive_weight)
        filtered = weights * risky_scale
        filtered.loc[cash_asset] = filtered.get(cash_asset, 0.0) + defensive_weight
        notes.append(f"Market filter active: {benchmark} below SMA{window}.")
        return filtered
    return weights


def _apply_volatility_target(weights: pd.Series, close: pd.DataFrame, as_of_date: pd.Timestamp, config: dict[str, Any], notes: list[str]) -> pd.Series:
    if not bool(get_config(config, "risk_management.volatility_target.enabled", False)):
        return weights
    target = float(get_config(config, "risk_management.volatility_target.target_volatility", 0.15))
    lookback = int(get_config(config, "risk_management.volatility_target.lookback_days", 63))
    cash_asset = str(get_config(config, "data.cash_asset", "CASH")).upper()
    asset_cols = [ticker for ticker in weights.index if ticker in close.columns]
    if not asset_cols:
        return weights
    returns = close[asset_cols].loc[:as_of_date].pct_change().tail(lookback).fillna(0)
    portfolio_returns = (returns * weights.reindex(asset_cols).fillna(0)).sum(axis=1)
    realized = float(portfolio_returns.std(ddof=0) * np.sqrt(252))
    if realized > target and realized > 1e-12:
        scale = target / realized
        scaled = weights * scale
        scaled.loc[cash_asset] = scaled.get(cash_asset, 0.0) + (1 - scale)
        notes.append(f"Volatility target scaled risky exposure to {scale:.1%}.")
        return scaled
    return weights


def _apply_sector_momentum(
    selected: list[str],
    bundle: GarpDataBundle,
    as_of_date: pd.Timestamp,
    config: dict[str, Any],
    notes: list[str],
) -> list[str]:
    if not bool(get_config(config, "risk_management.sector_momentum.enabled", False)):
        return selected
    benchmark = str(get_config(config, "risk_management.sector_momentum.benchmark", "SPY")).upper()
    lookback = int(get_config(config, "risk_management.sector_momentum.lookback_days", 126))
    if benchmark not in bundle.close.columns:
        notes.append("Sector momentum skipped: benchmark unavailable.")
        return selected
    benchmark_return = _lookback_return(bundle.close[benchmark].loc[:as_of_date], lookback)
    sector_returns: dict[str, list[float]] = {}
    for ticker in selected:
        if ticker in bundle.close.columns:
            sector_returns.setdefault(bundle.sectors.get(ticker, "Unknown"), []).append(_lookback_return(bundle.close[ticker].loc[:as_of_date], lookback))
    passing = {sector for sector, values in sector_returns.items() if values and np.nanmean(values) > benchmark_return}
    filtered = [ticker for ticker in selected if bundle.sectors.get(ticker, "Unknown") in passing]
    notes.append(f"Sector momentum kept {len(filtered)} of {len(selected)} names.")
    return filtered or selected[: max(1, len(selected) // 2)]


def _lookback_return(close: pd.Series, lookback: int) -> float:
    close = close.dropna()
    if len(close) <= lookback:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-lookback - 1] - 1)

