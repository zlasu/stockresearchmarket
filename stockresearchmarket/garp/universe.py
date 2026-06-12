from __future__ import annotations

from typing import Any

import pandas as pd

from stockresearchmarket.garp.config import get_config
from stockresearchmarket.garp.data_loader import latest_snapshot
from stockresearchmarket.garp.types import GarpDataBundle, UniverseResult


def build_universe(bundle: GarpDataBundle, as_of_date: pd.Timestamp, config: dict[str, Any]) -> UniverseResult:
    min_market_cap = float(get_config(config, "universe.min_market_cap_usd", 5_000_000_000))
    min_history_days = int(float(get_config(config, "universe.min_history_years", 3)) * 252)
    min_dollar_volume = float(get_config(config, "universe.min_avg_dollar_volume", 25_000_000))
    max_missing = float(get_config(config, "universe.max_missing_price_fraction", 0.05))
    configured = [str(item).upper() for item in get_config(config, "universe.tickers", [])]
    fundamentals = latest_snapshot(bundle.fundamentals, as_of_date)
    rows = []
    for ticker in configured:
        frame = bundle.frames.get(ticker)
        if frame is None or frame.empty:
            rows.append(_row(ticker, False, "missing_price_frame"))
            continue
        history = frame[frame.index <= as_of_date]
        if len(history) < min_history_days:
            rows.append(_row(ticker, False, "insufficient_history", rows=len(history)))
            continue
        missing_fraction = float(history["close"].tail(min_history_days).isna().mean())
        if missing_fraction > max_missing:
            rows.append(_row(ticker, False, "too_many_price_gaps", missing_fraction=missing_fraction))
            continue
        avg_dollar_volume = float((history["close"] * history["volume"]).tail(63).mean())
        if avg_dollar_volume < min_dollar_volume:
            rows.append(_row(ticker, False, "insufficient_liquidity", avg_dollar_volume=avg_dollar_volume))
            continue
        market_cap = None
        if not fundamentals.empty and ticker in fundamentals.index and "market_cap" in fundamentals.columns:
            market_cap = float(fundamentals.loc[ticker, "market_cap"])
        if market_cap is None and bool(get_config(config, "universe.require_fundamentals", False)):
            rows.append(_row(ticker, False, "missing_fundamentals"))
            continue
        if market_cap is not None and market_cap < min_market_cap:
            rows.append(_row(ticker, False, "below_market_cap", market_cap=market_cap))
            continue
        rows.append(
            _row(
                ticker,
                True,
                "ok",
                rows=len(history),
                market_cap=market_cap,
                avg_dollar_volume=avg_dollar_volume,
                sector=bundle.sectors.get(ticker, "Unknown"),
            )
        )
    diagnostics = pd.DataFrame(rows)
    members = diagnostics.loc[diagnostics["eligible"], "ticker"].tolist() if not diagnostics.empty else []
    return UniverseResult(members=members, diagnostics=diagnostics)


def _row(ticker: str, eligible: bool, reason: str, **values: object) -> dict[str, object]:
    return {"ticker": ticker, "eligible": eligible, "reason": reason, **values}

