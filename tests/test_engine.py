from __future__ import annotations

import pandas as pd

from stockresearchmarket.data.sources import synthetic_ohlcv
from stockresearchmarket.engine.vectorized import CostModel, run_signal_backtest
from stockresearchmarket.strategies.basic import buy_hold, sma_cross


def test_buy_hold_tracks_positive_synthetic_series() -> None:
    data = synthetic_ohlcv("SPY", "2018-01-01", "2022-01-01")
    result = run_signal_backtest(data, buy_hold(data), "buy_hold", "SPY", costs=CostModel(0, 0, 0))
    assert len(result.equity) == len(data)
    assert "total_return" in result.metrics
    assert result.metrics["trades"] == 1


def test_signal_is_shifted_to_avoid_lookahead() -> None:
    data = pd.DataFrame(
        {
            "open": [100, 100, 110],
            "high": [100, 110, 121],
            "low": [100, 100, 110],
            "close": [100, 110, 121],
            "volume": [1, 1, 1],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    signal = pd.Series([1, 1, 1], index=data.index)
    result = run_signal_backtest(data, signal, "test", "TST", costs=CostModel(0, 0, 0), initial_capital=100)
    assert round(result.equity.iloc[-1], 2) == 121.00


def test_sma_cross_invalid_windows_returns_flat_signal() -> None:
    data = synthetic_ohlcv("QQQ", "2020-01-01", "2021-01-01")
    signal = sma_cross(data, fast_window=200, slow_window=50)
    assert signal.sum() == 0

