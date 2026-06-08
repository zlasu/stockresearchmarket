from __future__ import annotations

import pandas as pd

from stockresearchmarket.features.indicators import rsi, sma


def buy_hold(data: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=data.index)


def sma_cross(data: pd.DataFrame, fast_window: int = 50, slow_window: int = 200) -> pd.Series:
    if fast_window >= slow_window:
        return pd.Series(0.0, index=data.index)
    close = data["close"]
    fast = sma(close, fast_window)
    slow = sma(close, slow_window)
    return (fast > slow).astype(float).fillna(0)


def rsi_mean_reversion(
    data: pd.DataFrame,
    rsi_window: int = 14,
    entry_rsi: float = 35,
    exit_rsi: float = 55,
    trend_window: int = 200,
) -> pd.Series:
    close = data["close"]
    oscillator = rsi(close, int(rsi_window))
    trend = sma(close, int(trend_window))
    values: list[float] = []
    position = 0.0
    for timestamp in data.index:
        if position > 0 and (oscillator.loc[timestamp] >= exit_rsi or close.loc[timestamp] < trend.loc[timestamp]):
            position = 0.0
        if position == 0 and close.loc[timestamp] > trend.loc[timestamp] and oscillator.loc[timestamp] <= entry_rsi:
            position = 1.0
        values.append(position)
    return pd.Series(values, index=data.index)


def donchian_breakout(data: pd.DataFrame, breakout_window: int = 55, exit_window: int = 20) -> pd.Series:
    close = data["close"]
    breakout = close > close.rolling(int(breakout_window), min_periods=int(breakout_window)).max().shift(1)
    exit_line = close < close.rolling(int(exit_window), min_periods=max(2, int(exit_window) // 2)).min().shift(1)
    values: list[float] = []
    position = 0.0
    for timestamp in data.index:
        if position > 0 and bool(exit_line.loc[timestamp]):
            position = 0.0
        if position == 0 and bool(breakout.loc[timestamp]):
            position = 1.0
        values.append(position)
    return pd.Series(values, index=data.index)

