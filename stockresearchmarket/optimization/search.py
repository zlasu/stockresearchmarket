from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from stockresearchmarket.engine.vectorized import BacktestResult, CostModel, run_signal_backtest
from stockresearchmarket.strategies.registry import get_strategy

try:
    import optuna
except ModuleNotFoundError:  # pragma: no cover - exercised when optional dep is missing
    optuna = None

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    if not keys:
        return [{}]
    return [dict(zip(keys, values, strict=True)) for values in product(*[grid[key] for key in keys])]


def score_result(result: BacktestResult, objective: str = "sharpe_drawdown") -> float:
    if objective == "cagr":
        return float(result.metrics["cagr"])
    if objective == "alpha":
        return result.alpha_total_return
    sharpe = float(result.metrics["sharpe"])
    max_dd = abs(float(result.metrics["max_drawdown"]))
    alpha = result.alpha_total_return
    return sharpe + 0.35 * alpha - 0.75 * max_dd


def optimize_strategy(
    data: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list[Any]],
    method: str = "grid",
    trials: int = 80,
    objective: str = "sharpe_drawdown",
    min_trades: int = 5,
    initial_capital: float = 10_000,
    costs: CostModel | None = None,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
    ticker: str = "",
) -> list[BacktestResult]:
    if method == "optuna" and optuna is not None and grid:
        return _optuna_search(
            data=data,
            strategy_name=strategy_name,
            grid=grid,
            trials=trials,
            objective=objective,
            min_trades=min_trades,
            initial_capital=initial_capital,
            costs=costs,
            annualization_days=annualization_days,
            risk_free_rate=risk_free_rate,
            ticker=ticker,
        )

    strategy = get_strategy(strategy_name)
    results: list[BacktestResult] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for params in expand_grid(grid):
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        signal = strategy.generate(data, **params)
        result = run_signal_backtest(
            data=data,
            signal=signal,
            strategy=strategy_name,
            ticker=ticker,
            params=params,
            initial_capital=initial_capital,
            costs=costs,
            annualization_days=annualization_days,
            risk_free_rate=risk_free_rate,
        )
        if int(result.metrics["trades"]) >= min_trades:
            results.append(result)
    return sorted(results, key=lambda item: score_result(item, objective), reverse=True)


def _optuna_search(
    data: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list[Any]],
    trials: int,
    objective: str,
    min_trades: int,
    initial_capital: float,
    costs: CostModel | None,
    annualization_days: int,
    risk_free_rate: float,
    ticker: str,
) -> list[BacktestResult]:
    assert optuna is not None
    strategy = get_strategy(strategy_name)
    results: dict[tuple[tuple[str, Any], ...], BacktestResult] = {}

    def objective_fn(trial: Any) -> float:
        params = {name: trial.suggest_categorical(name, values) for name, values in grid.items()}
        key = tuple(sorted(params.items()))
        if key in results:
            return score_result(results[key], objective)
        signal = strategy.generate(data, **params)
        result = run_signal_backtest(
            data=data,
            signal=signal,
            strategy=strategy_name,
            ticker=ticker,
            params=params,
            initial_capital=initial_capital,
            costs=costs,
            annualization_days=annualization_days,
            risk_free_rate=risk_free_rate,
        )
        if int(result.metrics["trades"]) < min_trades:
            return -1_000_000 + int(result.metrics["trades"])
        results[key] = result
        return score_result(result, objective)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_fn, n_trials=trials, show_progress_bar=False)
    return sorted(results.values(), key=lambda item: score_result(item, objective), reverse=True)


def summarize_results(results: list[BacktestResult], objective: str = "sharpe_drawdown") -> pd.DataFrame:
    rows = []
    for rank, result in enumerate(results, start=1):
        row = {
            "rank": rank,
            "ticker": result.ticker,
            "strategy": result.strategy,
            "score": score_result(result, objective),
            "alpha_total_return": result.alpha_total_return,
            **result.metrics,
            **{f"benchmark_{key}": value for key, value in result.benchmark_metrics.items()},
            **{f"param_{key}": value for key, value in result.params.items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward_validate(
    data: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list[Any]],
    train_years: int = 5,
    test_years: int = 1,
    objective: str = "sharpe_drawdown",
    min_trades: int = 3,
    run_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    run_kwargs = run_kwargs or {}
    if data.empty:
        return pd.DataFrame()
    start = data.index.min()
    final = data.index.max()
    rows = []
    window_start = start
    while True:
        train_end = window_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > final:
            break
        train = data.loc[(data.index >= window_start) & (data.index < train_end)]
        test = data.loc[(data.index >= train_end) & (data.index < test_end)]
        if len(train) < 252 or len(test) < 60:
            window_start = window_start + pd.DateOffset(years=test_years)
            continue
        train_results = optimize_strategy(
            train,
            strategy_name,
            grid,
            method="grid",
            objective=objective,
            min_trades=min_trades,
            **run_kwargs,
        )
        if not train_results:
            window_start = window_start + pd.DateOffset(years=test_years)
            continue
        params = train_results[0].params
        strategy = get_strategy(strategy_name)
        signal = strategy.generate(test, **params)
        test_result = run_signal_backtest(test, signal, strategy_name, params=params, **run_kwargs)
        rows.append(
            {
                "train_start": window_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": train_end.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "train_score": score_result(train_results[0], objective),
                "test_sharpe": test_result.metrics["sharpe"],
                "test_cagr": test_result.metrics["cagr"],
                "test_max_drawdown": test_result.metrics["max_drawdown"],
                "test_total_return": test_result.metrics["total_return"],
                "test_alpha_total_return": test_result.alpha_total_return,
                "test_trades": test_result.metrics["trades"],
                "params": params,
            }
        )
        window_start = window_start + pd.DateOffset(years=test_years)
    return pd.DataFrame(rows)
