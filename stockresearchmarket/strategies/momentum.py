from __future__ import annotations

import pandas as pd


def dual_momentum_weights(
    close: pd.DataFrame,
    lookback_days: int = 126,
    top_n: int = 4,
    rebalance: str = "ME",
    cash_asset: str | None = "TLT",
    min_momentum: float = 0.0,
) -> pd.DataFrame:
    close = close.sort_index().dropna(how="all")
    momentum = close.pct_change(int(lookback_days))
    if rebalance == "M":
        rebalance = "ME"
    rebalance_dates = close.resample(rebalance).last().index
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")

    for timestamp in rebalance_dates:
        if timestamp not in momentum.index:
            timestamp = momentum.index[momentum.index.searchsorted(timestamp) - 1]
        scores = momentum.loc[timestamp].dropna().sort_values(ascending=False)
        selected = list(scores[scores >= float(min_momentum)].head(int(top_n)).index)
        row = pd.Series(0.0, index=close.columns)
        if selected:
            row.loc[selected] = 1 / len(selected)
        elif cash_asset and cash_asset in row.index:
            row.loc[cash_asset] = 1.0
        weights.loc[timestamp] = row

    return weights.ffill().fillna(0.0)
