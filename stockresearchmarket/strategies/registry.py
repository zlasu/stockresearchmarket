from __future__ import annotations

from stockresearchmarket.strategies.base import Strategy
from stockresearchmarket.strategies.basic import buy_hold, donchian_breakout, rsi_mean_reversion, sma_cross

STRATEGIES: dict[str, Strategy] = {
    "buy_hold": Strategy("buy_hold", "Long-only buy and hold benchmark.", buy_hold),
    "sma_cross": Strategy("sma_cross", "Long when fast SMA is above slow SMA.", sma_cross),
    "rsi_mean_reversion": Strategy(
        "rsi_mean_reversion",
        "Buy pullbacks in an uptrend and exit on RSI recovery or trend break.",
        rsi_mean_reversion,
    ),
    "donchian_breakout": Strategy("donchian_breakout", "Trend-following breakout with Donchian exit.", donchian_breakout),
}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}") from exc

