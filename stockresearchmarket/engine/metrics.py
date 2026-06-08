from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trades: int,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int]:
    returns = returns.fillna(0)
    if returns.empty or equity.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "trades": trades,
        }

    years = max(len(returns) / annualization_days, 1 / annualization_days)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1.0
    excess = returns - risk_free_rate / annualization_days
    volatility = float(returns.std(ddof=0) * np.sqrt(annualization_days))
    sharpe = float(excess.mean() / returns.std(ddof=0) * np.sqrt(annualization_days)) if returns.std(ddof=0) > 1e-12 else 0.0
    downside = returns[returns < 0].std(ddof=0)
    sortino = float(excess.mean() / downside * np.sqrt(annualization_days)) if downside > 1e-12 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < -1e-12 else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "trades": int(trades),
    }


def trade_stats(returns: pd.Series, position: pd.Series) -> dict[str, float]:
    position = position.fillna(0)
    entries = (position.ne(0) & position.shift(1, fill_value=0).eq(0)).astype(int)
    trade_id = entries.cumsum().where(position.ne(0))
    trade_returns = returns.groupby(trade_id).apply(lambda values: float((1 + values).prod() - 1)).dropna()
    if trade_returns.empty:
        return {"win_rate": 0.0, "profit_factor": 0.0, "avg_trade_return": 0.0}
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    return {
        "win_rate": float(len(wins) / len(trade_returns)),
        "profit_factor": gross_profit / gross_loss if gross_loss > 1e-12 else 0.0,
        "avg_trade_return": float(trade_returns.mean()),
    }

