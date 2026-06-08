from __future__ import annotations

from stockresearchmarket.data.sources import synthetic_ohlcv
from stockresearchmarket.optimization.search import expand_grid, optimize_strategy


def test_expand_grid() -> None:
    grid = {"a": [1, 2], "b": ["x", "y"]}
    assert len(expand_grid(grid)) == 4


def test_optimize_strategy_returns_ranked_results() -> None:
    data = synthetic_ohlcv("SPY", "2016-01-01", "2022-01-01")
    results = optimize_strategy(
        data,
        "sma_cross",
        {"fast_window": [20, 50], "slow_window": [100, 150]},
        min_trades=1,
        ticker="SPY",
    )
    assert results
    assert results[0].strategy == "sma_cross"

