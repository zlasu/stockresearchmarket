from __future__ import annotations

import pandas as pd

from stockresearchmarket.data.sources import normalize_ohlcv


def test_normalize_yfinance_single_ticker_multiindex() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "SPY"),
            ("Close", "SPY"),
            ("High", "SPY"),
            ("Low", "SPY"),
            ("Open", "SPY"),
            ("Volume", "SPY"),
        ],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [[101.0, 102.0, 103.0, 99.0, 100.0, 1_000_000]],
        index=pd.to_datetime(["2026-01-02"]),
        columns=columns,
    )
    normalized = normalize_ohlcv(raw)
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume", "raw_close"]
    assert normalized["close"].iloc[0] == 101.0
    assert normalized["raw_close"].iloc[0] == 102.0

