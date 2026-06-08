from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockresearchmarket.engine.metrics import performance_metrics, trade_stats
from stockresearchmarket.features.indicators import drawdown


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    spread_bps: float = 1.0

    @property
    def one_way_cost(self) -> float:
        return (self.fee_bps + self.slippage_bps + self.spread_bps / 2) / 10_000


@dataclass
class BacktestResult:
    strategy: str
    ticker: str
    params: dict
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    metrics: dict
    benchmark_metrics: dict
    drawdown: pd.Series

    @property
    def alpha_total_return(self) -> float:
        return float(self.metrics["total_return"] - self.benchmark_metrics["total_return"])


def run_signal_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    strategy: str,
    ticker: str,
    params: dict | None = None,
    initial_capital: float = 10_000,
    costs: CostModel | None = None,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    costs = costs or CostModel()
    params = params or {}
    frame = data.sort_index().copy()
    close = frame["close"].astype(float)
    signal = signal.reindex(close.index).fillna(0).astype(float).clip(lower=-1, upper=1)
    position = signal.shift(1).fillna(0)
    asset_returns = close.pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    returns = position * asset_returns - turnover * costs.one_way_cost
    equity = initial_capital * (1 + returns).cumprod()

    benchmark_position = pd.Series(1.0, index=close.index).shift(1).fillna(0)
    benchmark_turnover = benchmark_position.diff().abs().fillna(benchmark_position.abs())
    benchmark_returns = benchmark_position * asset_returns - benchmark_turnover * costs.one_way_cost
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()

    trades = int((position.ne(0) & position.shift(1, fill_value=0).eq(0)).sum())
    metrics = performance_metrics(returns, equity, trades, annualization_days, risk_free_rate)
    metrics.update(trade_stats(returns, position))
    benchmark_metrics = performance_metrics(benchmark_returns, benchmark_equity, 1, annualization_days, risk_free_rate)
    benchmark_metrics.update(trade_stats(benchmark_returns, benchmark_position))

    return BacktestResult(
        strategy=strategy,
        ticker=ticker,
        params=params,
        equity=equity.rename("strategy"),
        returns=returns.rename("strategy_return"),
        position=position.rename("position"),
        benchmark_equity=benchmark_equity.rename("buy_hold"),
        benchmark_returns=benchmark_returns.rename("buy_hold_return"),
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        drawdown=drawdown(equity).rename("drawdown"),
    )


@dataclass
class PortfolioBacktestResult:
    strategy: str
    params: dict
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    metrics: dict
    benchmark_metrics: dict
    drawdown: pd.Series

    @property
    def alpha_total_return(self) -> float:
        return float(self.metrics["total_return"] - self.benchmark_metrics["total_return"])


def run_portfolio_backtest(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    strategy: str,
    params: dict | None = None,
    initial_capital: float = 10_000,
    costs: CostModel | None = None,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
) -> PortfolioBacktestResult:
    costs = costs or CostModel()
    params = params or {}
    close = close.sort_index().dropna(how="all").ffill()
    weights = weights.reindex(close.index).ffill().fillna(0.0)
    effective_weights = weights.shift(1).fillna(0.0)
    asset_returns = close.pct_change().fillna(0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    returns = (effective_weights * asset_returns).sum(axis=1) - turnover * costs.one_way_cost
    equity = initial_capital * (1 + returns).cumprod()

    equal_weights = close.notna().astype(float)
    equal_weights = equal_weights.div(equal_weights.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
    benchmark_weights = equal_weights.shift(1).fillna(0.0)
    benchmark_returns = (benchmark_weights * asset_returns).sum(axis=1)
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()

    trades = int((turnover > 0).sum())
    metrics = performance_metrics(returns, equity, trades, annualization_days, risk_free_rate)
    benchmark_metrics = performance_metrics(benchmark_returns, benchmark_equity, 1, annualization_days, risk_free_rate)
    return PortfolioBacktestResult(
        strategy=strategy,
        params=params,
        equity=equity.rename("strategy"),
        returns=returns.rename("strategy_return"),
        weights=weights,
        benchmark_equity=benchmark_equity.rename("equal_weight"),
        benchmark_returns=benchmark_returns.rename("equal_weight_return"),
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        drawdown=drawdown(equity).rename("drawdown"),
    )

