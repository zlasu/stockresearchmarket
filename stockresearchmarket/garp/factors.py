from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockresearchmarket.features.indicators import drawdown, sma
from stockresearchmarket.garp.config import get_config
from stockresearchmarket.garp.data_loader import latest_snapshot
from stockresearchmarket.garp.types import FactorResult, GarpDataBundle

FACTOR_MAP: dict[str, dict[str, tuple[str, int]]] = {
    "growth": {
        "revenue_growth_3y": ("revenue_growth_3y", 1),
        "revenue_growth_5y": ("revenue_growth_5y", 1),
        "eps_growth_3y": ("eps_growth_3y", 1),
        "eps_growth_5y": ("eps_growth_5y", 1),
        "operating_cash_flow_growth": ("operating_cash_flow_growth", 1),
    },
    "value": {
        "pe": ("pe", -1),
        "forward_pe": ("forward_pe", -1),
        "ps": ("ps", -1),
        "ev_ebitda": ("ev_ebitda", -1),
        "fcf_yield": ("fcf_yield", 1),
    },
    "quality": {
        "roe": ("roe", 1),
        "roic": ("roic", 1),
        "gross_margin": ("gross_margin", 1),
        "net_margin": ("net_margin", 1),
        "operating_margin": ("operating_margin", 1),
        "debt_equity": ("debt_equity", -1),
        "interest_coverage": ("interest_coverage", 1),
        "positive_fcf": ("positive_fcf", 1),
        "altman_z": ("altman_z", 1),
    },
    "revisions": {
        "eps_revision_30d": ("eps_revision_30d", 1),
        "eps_revision_90d": ("eps_revision_90d", 1),
        "revenue_revision_90d": ("revenue_revision_90d", 1),
        "target_revision_90d": ("target_revision_90d", 1),
    },
}


def compute_factors(
    bundle: GarpDataBundle,
    members: list[str],
    as_of_date: pd.Timestamp,
    config: dict[str, Any],
) -> FactorResult:
    fundamentals = latest_snapshot(bundle.fundamentals, as_of_date)
    estimates = latest_snapshot(bundle.estimates, as_of_date)
    rows: dict[str, dict[str, float]] = {ticker: {} for ticker in members}
    metadata: list[dict[str, object]] = []
    _add_statement_factors(rows, metadata, fundamentals, members, bundle)
    _add_estimate_factors(rows, metadata, estimates, members, bundle)
    _add_price_factors(rows, metadata, bundle, members, as_of_date)
    _add_risk_factors(rows, metadata, bundle, members, as_of_date, config)
    values = pd.DataFrame.from_dict(rows, orient="index").rename_axis("ticker")
    return FactorResult(values=values, metadata=pd.DataFrame(metadata))


def _add_statement_factors(
    rows: dict[str, dict[str, float]],
    metadata: list[dict[str, object]],
    fundamentals: pd.DataFrame,
    members: list[str],
    bundle: GarpDataBundle,
) -> None:
    available_categories = {
        status.category: status.status
        for status in bundle.availability
        if status.category in {"growth", "value", "quality"} and status.factor == "*"
    }
    for category, factors in FACTOR_MAP.items():
        if category == "revisions":
            continue
        category_status = available_categories.get(category, "unavailable")
        for factor, (source_col, direction) in factors.items():
            column = f"{category}_{factor}"
            status = category_status if source_col in fundamentals.columns else "unavailable"
            reason = "available point-in-time snapshot" if status == "available" else "missing safe fundamental column"
            metadata.append(
                {"category": category, "factor": column, "source_column": source_col, "direction": direction, "status": status, "reason": reason}
            )
            if status != "available":
                continue
            for ticker in members:
                if ticker in fundamentals.index:
                    value = fundamentals.loc[ticker, source_col]
                    rows[ticker][column] = float(value) * direction if pd.notna(value) else np.nan


def _add_estimate_factors(
    rows: dict[str, dict[str, float]],
    metadata: list[dict[str, object]],
    estimates: pd.DataFrame,
    members: list[str],
    bundle: GarpDataBundle,
) -> None:
    revision_status = next((status.status for status in bundle.availability if status.category == "revisions"), "unavailable")
    for factor, (source_col, direction) in FACTOR_MAP["revisions"].items():
        column = f"revisions_{factor}"
        status = revision_status if source_col in estimates.columns else "unavailable"
        reason = "estimate snapshot has as_of_date" if status == "available" else "missing safe estimate column"
        metadata.append(
            {"category": "revisions", "factor": column, "source_column": source_col, "direction": direction, "status": status, "reason": reason}
        )
        if status != "available":
            continue
        for ticker in members:
            if ticker in estimates.index:
                value = estimates.loc[ticker, source_col]
                rows[ticker][column] = float(value) * direction if pd.notna(value) else np.nan


def _add_price_factors(
    rows: dict[str, dict[str, float]],
    metadata: list[dict[str, object]],
    bundle: GarpDataBundle,
    members: list[str],
    as_of_date: pd.Timestamp,
) -> None:
    price_specs = {
        "momentum_12_1": "12M minus last 1M momentum",
        "momentum_6m": "6M momentum",
        "momentum_3m": "3M momentum",
        "above_sma200": "close above SMA200",
        "relative_strength_spy": "6M return minus SPY 6M return",
    }
    for factor, reason in price_specs.items():
        metadata.append({"category": "momentum", "factor": f"momentum_{factor}", "source_column": "close", "direction": 1, "status": "available", "reason": reason})
    spy = bundle.close["SPY"].loc[:as_of_date] if "SPY" in bundle.close.columns else pd.Series(dtype=float)
    spy_6m = _return(spy, 126)
    for ticker in members:
        close = bundle.close[ticker].loc[:as_of_date].dropna() if ticker in bundle.close.columns else pd.Series(dtype=float)
        rows[ticker]["momentum_momentum_12_1"] = _return(close.iloc[:-21], 231) if len(close) > 252 else np.nan
        rows[ticker]["momentum_momentum_6m"] = _return(close, 126)
        rows[ticker]["momentum_momentum_3m"] = _return(close, 63)
        rows[ticker]["momentum_above_sma200"] = float(close.iloc[-1] > sma(close, 200).iloc[-1]) if len(close) >= 200 else np.nan
        rows[ticker]["momentum_relative_strength_spy"] = rows[ticker]["momentum_momentum_6m"] - spy_6m if pd.notna(spy_6m) else np.nan


def _add_risk_factors(
    rows: dict[str, dict[str, float]],
    metadata: list[dict[str, object]],
    bundle: GarpDataBundle,
    members: list[str],
    as_of_date: pd.Timestamp,
    config: dict[str, Any],
) -> None:
    for factor, reason in {
        "volatility_3m": "lower 3M volatility is better",
        "volatility_12m": "lower 12M volatility is better",
        "beta_vs_spy": "lower beta is defensive",
        "max_drawdown_12m": "less severe 12M drawdown is better",
        "liquidity": "higher dollar volume is better",
    }.items():
        metadata.append({"category": "risk", "factor": f"risk_{factor}", "source_column": "price", "direction": 1, "status": "available", "reason": reason})
    spy_returns = bundle.close["SPY"].loc[:as_of_date].pct_change().tail(252) if "SPY" in bundle.close.columns else pd.Series(dtype=float)
    for ticker in members:
        frame = bundle.frames.get(ticker)
        close = bundle.close[ticker].loc[:as_of_date].dropna() if ticker in bundle.close.columns else pd.Series(dtype=float)
        returns = close.pct_change().dropna()
        rows[ticker]["risk_volatility_3m"] = -float(returns.tail(63).std(ddof=0) * np.sqrt(252)) if len(returns) >= 63 else np.nan
        rows[ticker]["risk_volatility_12m"] = -float(returns.tail(252).std(ddof=0) * np.sqrt(252)) if len(returns) >= 252 else np.nan
        rows[ticker]["risk_beta_vs_spy"] = -_beta(returns.tail(252), spy_returns)
        rows[ticker]["risk_max_drawdown_12m"] = float(drawdown(close.tail(252)).min()) if len(close) >= 252 else np.nan
        if frame is not None and len(frame.loc[:as_of_date]) >= 63:
            hist = frame.loc[:as_of_date].tail(63)
            rows[ticker]["risk_liquidity"] = float(np.log1p((hist["close"] * hist["volume"]).mean()))
        _apply_risk_exclusion(rows[ticker], config)


def _apply_risk_exclusion(row: dict[str, float], config: dict[str, Any]) -> None:
    max_vol = float(get_config(config, "factors.risk_filters.max_volatility_3m", 0.8))
    max_dd = float(get_config(config, "factors.risk_filters.max_drawdown_12m", -0.7))
    row["risk_pass"] = float(
        pd.notna(row.get("risk_volatility_3m"))
        and -row["risk_volatility_3m"] <= max_vol
        and pd.notna(row.get("risk_max_drawdown_12m"))
        and row["risk_max_drawdown_12m"] >= max_dd
    )


def _return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1)


def _beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 60:
        return np.nan
    benchmark_var = float(aligned.iloc[:, 1].var(ddof=0))
    if benchmark_var <= 1e-12:
        return np.nan
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / benchmark_var)

