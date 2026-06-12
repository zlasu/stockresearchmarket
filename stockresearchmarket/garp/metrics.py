from __future__ import annotations

import numpy as np
import pandas as pd

from stockresearchmarket.engine.metrics import performance_metrics


def garp_metrics(
    returns: pd.Series,
    equity: pd.Series,
    benchmark_returns: pd.DataFrame,
    turnover: pd.Series,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | str]:
    metrics = performance_metrics(returns, equity, int((turnover > 0).sum()), annualization_days, risk_free_rate)
    primary = benchmark_returns.iloc[:, 0] if not benchmark_returns.empty else pd.Series(dtype=float)
    aligned = pd.concat([returns, primary], axis=1).dropna()
    if len(aligned) >= 60 and aligned.iloc[:, 1].var(ddof=0) > 1e-12:
        beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var(ddof=0))
        alpha_daily = aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()
        metrics["beta_vs_benchmark"] = beta
        metrics["alpha_annualized"] = float(alpha_daily * annualization_days)
    else:
        metrics["beta_vs_benchmark"] = 0.0
        metrics["alpha_annualized"] = 0.0
    if not benchmark_returns.empty:
        benchmark_equity = (1 + primary.fillna(0)).cumprod()
        benchmark_total = float(benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1) if not benchmark_equity.empty else 0.0
        metrics["benchmark_total_return"] = benchmark_total
        metrics["alpha_total_return"] = float(metrics["total_return"] - benchmark_total)
    metrics["avg_annual_turnover"] = float(turnover.resample("YE").sum().mean()) if not turnover.empty else 0.0
    metrics["transaction_cost_drag"] = float((turnover * 0).sum())
    metrics["monthly_win_rate"] = float(monthly_returns(returns).gt(0).mean()) if not returns.empty else 0.0
    return metrics


def monthly_returns(returns: pd.Series) -> pd.Series:
    return returns.resample("ME").apply(lambda values: float((1 + values).prod() - 1))


def yearly_returns(returns: pd.Series) -> pd.Series:
    return returns.resample("YE").apply(lambda values: float((1 + values).prod() - 1))


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 126) -> pd.Series:
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    cov = aligned.iloc[:, 0].rolling(window).cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].rolling(window).var()
    return cov / var.replace(0, np.nan)


def top_drawdown_periods(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    drawdown = equity / equity.cummax() - 1
    rows = []
    in_dd = False
    start = None
    trough = None
    trough_value = 0.0
    for timestamp, value in drawdown.items():
        if value < 0 and not in_dd:
            in_dd = True
            start = timestamp
            trough = timestamp
            trough_value = float(value)
        elif value < 0 and in_dd and value < trough_value:
            trough = timestamp
            trough_value = float(value)
        elif value >= 0 and in_dd:
            rows.append({"start": start, "trough": trough, "recovery": timestamp, "drawdown": trough_value})
            in_dd = False
    if in_dd:
        rows.append({"start": start, "trough": trough, "recovery": pd.NaT, "drawdown": trough_value})
    return pd.DataFrame(rows).sort_values("drawdown").head(top_n) if rows else pd.DataFrame()

