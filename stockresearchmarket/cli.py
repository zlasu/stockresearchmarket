from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from stockresearchmarket.config import get_nested, load_config, strategy_defaults, strategy_grid, universe_from_config
from stockresearchmarket.data.sources import (
    DataRequest,
    data_quality_summary,
    load_close_panel,
    load_history,
    parse_tickers,
)
from stockresearchmarket.engine.vectorized import CostModel, run_portfolio_backtest, run_signal_backtest
from stockresearchmarket.garp.autoresearch import load_leaderboard, run_autoresearch
from stockresearchmarket.garp.backtest import run_garp_backtest
from stockresearchmarket.garp.comparison import compare_garp_runs
from stockresearchmarket.optimization.search import optimize_strategy, summarize_results, walk_forward_validate
from stockresearchmarket.reports.plots import write_optimizer_report, write_portfolio_report, write_single_asset_report
from stockresearchmarket.strategies.momentum import dual_momentum_weights
from stockresearchmarket.strategies.registry import get_strategy

app = typer.Typer(help="StockResearchMarket research CLI")
console = Console()


def _date_bounds(config: dict, years: int | None, start: str | None, end: str | None) -> tuple[str | None, str | None]:
    end_value = end or get_nested(config, "data.end")
    start_value = start or get_nested(config, "data.start")
    if years:
        end_ts = pd.Timestamp(end_value or datetime.now().date())
        start_value = (end_ts - pd.DateOffset(years=years)).date().isoformat()
        end_value = end_ts.date().isoformat()
    return start_value, end_value


def _costs(config: dict) -> CostModel:
    return CostModel(
        fee_bps=float(get_nested(config, "backtest.fee_bps", 1.0)),
        slippage_bps=float(get_nested(config, "backtest.slippage_bps", 2.0)),
        spread_bps=float(get_nested(config, "backtest.spread_bps", 1.0)),
    )


def _run_kwargs(config: dict) -> dict:
    return {
        "initial_capital": float(get_nested(config, "backtest.initial_capital", 10_000)),
        "costs": _costs(config),
        "annualization_days": int(get_nested(config, "backtest.annualization_days", 252)),
        "risk_free_rate": float(get_nested(config, "backtest.risk_free_rate", 0.0)),
    }


def _experiment_dir(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = Path("experiments") / f"{stamp}_{prefix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_frames(
    config: dict,
    tickers: list[str],
    provider: str | None,
    years: int | None,
    start: str | None,
    end: str | None,
    refresh: bool,
) -> dict[str, pd.DataFrame]:
    start_value, end_value = _date_bounds(config, years, start, end)
    request = DataRequest(
        tickers=tickers,
        start=start_value,
        end=end_value,
        provider=provider or get_nested(config, "data.provider", "yfinance"),
        cache_dir=Path(get_nested(config, "data.cache_dir", "data/historical")),
        raw_dir=Path(get_nested(config, "data.raw_dir", "data/raw")),
        refresh=refresh,
    )
    return load_history(request)


@app.command()
def data(
    tickers: Annotated[str, typer.Option(help="Comma-separated tickers. Empty uses configured core universe.")] = "",
    universe: Annotated[str, typer.Option(help="Config universe groups, e.g. core,sectors,megacap.")] = "core",
    provider: Annotated[str | None, typer.Option(help="yfinance, stooq, csv, synthetic")] = None,
    years: Annotated[int | None, typer.Option(help="Override date range with last N years.")] = 20,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD start date.")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD end date.")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore cache and download again.")] = False,
    config: Annotated[Path, typer.Option(help="Config YAML path.")] = Path("configs/default.yaml"),
) -> None:
    cfg = load_config(config)
    selected = parse_tickers(tickers) if tickers else universe_from_config(cfg, universe)
    frames = _load_frames(cfg, selected, provider, years, start, end, refresh)
    quality = data_quality_summary(frames)
    _print_dataframe(quality, "Data quality")


@app.command()
def run(
    strategy: Annotated[str, typer.Option(help="Strategy name.")] = "sma_cross",
    tickers: Annotated[str, typer.Option(help="Comma-separated tickers.")] = "SPY",
    provider: Annotated[str | None, typer.Option(help="yfinance, stooq, csv, synthetic")] = None,
    years: Annotated[int | None, typer.Option(help="Backtest last N years.")] = 20,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD start date.")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD end date.")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore cache and download again.")] = False,
    config: Annotated[Path, typer.Option(help="Config YAML path.")] = Path("configs/default.yaml"),
) -> None:
    cfg = load_config(config)
    selected = parse_tickers(tickers)
    frames = _load_frames(cfg, selected, provider, years, start, end, refresh)
    strategy_obj = get_strategy(strategy)
    params = strategy_defaults(cfg, strategy)
    output_root = _experiment_dir(f"{strategy}_run")
    rows = []
    for ticker, frame in frames.items():
        signal = strategy_obj.generate(frame, **params)
        result = run_signal_backtest(frame, signal, strategy, ticker, params=params, **_run_kwargs(cfg))
        report = write_single_asset_report(result, output_root / ticker)
        rows.append(
            {
                "ticker": ticker,
                "total_return": result.metrics["total_return"],
                "buy_hold_return": result.benchmark_metrics["total_return"],
                "alpha": result.alpha_total_return,
                "sharpe": result.metrics["sharpe"],
                "max_drawdown": result.metrics["max_drawdown"],
                "trades": result.metrics["trades"],
                "report": str(report),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "summary.csv", index=False)
    _print_dataframe(summary, f"Backtest saved to {output_root}")


@app.command()
def optimize(
    strategy: Annotated[str, typer.Option(help="Strategy name.")] = "sma_cross",
    ticker: Annotated[str, typer.Option(help="Single ticker.")] = "SPY",
    method: Annotated[str, typer.Option(help="grid or optuna.")] = "grid",
    provider: Annotated[str | None, typer.Option(help="yfinance, stooq, csv, synthetic")] = None,
    years: Annotated[int | None, typer.Option(help="Backtest last N years.")] = 20,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD start date.")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD end date.")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore cache and download again.")] = False,
    config: Annotated[Path, typer.Option(help="Config YAML path.")] = Path("configs/default.yaml"),
) -> None:
    cfg = load_config(config)
    frames = _load_frames(cfg, [ticker.upper()], provider, years, start, end, refresh)
    frame = frames[ticker.upper()]
    grid = strategy_grid(cfg, strategy)
    objective = str(get_nested(cfg, "optimization.objective", "sharpe_drawdown"))
    results = optimize_strategy(
        frame,
        strategy,
        grid,
        method=method,
        trials=int(get_nested(cfg, "optimization.optuna_trials", 80)),
        objective=objective,
        min_trades=int(get_nested(cfg, "optimization.min_trades", 5)),
        ticker=ticker.upper(),
        **_run_kwargs(cfg),
    )
    output_root = _experiment_dir(f"{strategy}_{ticker.upper()}_optimize")
    summary = summarize_results(results, objective).head(int(get_nested(cfg, "optimization.top_n", 20)))
    optimizer_report = write_optimizer_report(summary, output_root)
    if results:
        write_single_asset_report(results[0], output_root / "best")
    wf = walk_forward_validate(
        frame,
        strategy,
        grid,
        train_years=int(get_nested(cfg, "optimization.walk_forward.train_years", 5)),
        test_years=int(get_nested(cfg, "optimization.walk_forward.test_years", 1)),
        objective=objective,
        min_trades=max(1, int(get_nested(cfg, "optimization.min_trades", 5)) // 2),
        run_kwargs={"ticker": ticker.upper(), **_run_kwargs(cfg)},
    )
    wf.to_csv(output_root / "walk_forward.csv", index=False)
    console.print(f"[green]Optimizer report:[/] {optimizer_report}")
    _print_dataframe(summary, f"Top candidates saved to {output_root}")
    if not wf.empty:
        _print_dataframe(wf, "Walk-forward")


@app.command()
def portfolio(
    universe: Annotated[str, typer.Option(help="Config universe groups, e.g. core,sectors.")] = "core,sectors",
    tickers: Annotated[str, typer.Option(help="Optional explicit comma-separated tickers.")] = "",
    provider: Annotated[str | None, typer.Option(help="yfinance, stooq, csv, synthetic")] = None,
    years: Annotated[int | None, typer.Option(help="Backtest last N years.")] = 20,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD start date.")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD end date.")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore cache and download again.")] = False,
    config: Annotated[Path, typer.Option(help="Config YAML path.")] = Path("configs/default.yaml"),
) -> None:
    cfg = load_config(config)
    selected = parse_tickers(tickers) if tickers else universe_from_config(cfg, universe)
    frames = _load_frames(cfg, selected, provider, years, start, end, refresh)
    close = load_close_panel(frames)
    params = strategy_defaults(cfg, "dual_momentum")
    weights = dual_momentum_weights(close, **params)
    result = run_portfolio_backtest(close, weights, "dual_momentum", params=params, **_run_kwargs(cfg))
    output_root = _experiment_dir("dual_momentum_portfolio")
    report = write_portfolio_report(result, output_root)
    console.print(f"[green]Portfolio report:[/] {report}")
    _print_dataframe(pd.DataFrame([{**result.metrics, "alpha": result.alpha_total_return}]), f"Portfolio saved to {output_root}")


@app.command()
def smoke() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    tickers = ["SPY", "QQQ", "TLT"]
    frames = _load_frames(cfg, tickers, "synthetic", 10, None, None, True)
    output_root = _experiment_dir("smoke")
    rows = []
    for ticker, frame in frames.items():
        params = strategy_defaults(cfg, "sma_cross")
        signal = get_strategy("sma_cross").generate(frame, **params)
        result = run_signal_backtest(frame, signal, "sma_cross", ticker, params=params, **_run_kwargs(cfg))
        report = write_single_asset_report(result, output_root / ticker)
        rows.append({"ticker": ticker, **result.metrics, "alpha": result.alpha_total_return, "report": str(report)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "summary.csv", index=False)
    _print_dataframe(summary, f"Smoke test saved to {output_root}")


@app.command("garp-run")
def garp_run(
    experiment: Annotated[str, typer.Option(help="Experiment id/path, e.g. 001_baseline_garp.")] = "001_baseline_garp",
    provider: Annotated[str | None, typer.Option(help="Price provider override: synthetic, yfinance, stooq, csv.")] = None,
    years: Annotated[int | None, typer.Option(help="Backtest last N years.")] = 10,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD start date.")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD end date.")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore cache and rebuild data.")] = False,
) -> None:
    result = run_garp_backtest(experiment=experiment, provider=provider, years=years, start=start, end=end, refresh=refresh)
    console.print(f"[green]GARP report:[/] {result.output_dir / 'report.html'}")
    _print_dataframe(pd.DataFrame([{**result.metrics, "output_dir": str(result.output_dir)}]), f"GARP result {result.experiment_id}")


@app.command("garp-autoresearch")
def garp_autoresearch(
    base_experiment: Annotated[str, typer.Option(help="Base experiment id/path.")] = "001_baseline_garp",
    provider: Annotated[str | None, typer.Option(help="Price provider override.")] = None,
    years: Annotated[int | None, typer.Option(help="Backtest last N years.")] = 8,
    max_experiments: Annotated[int | None, typer.Option(help="Max generated variants to run.")] = 6,
) -> None:
    leaderboard = run_autoresearch(base_experiment=base_experiment, provider=provider, years=years, max_experiments=max_experiments)
    _print_dataframe(leaderboard, "GARP autoresearch leaderboard")


@app.command("garp-leaderboard")
def garp_leaderboard() -> None:
    leaderboard = load_leaderboard()
    _print_dataframe(leaderboard, "GARP autoresearch leaderboard")


@app.command("garp-compare")
def garp_compare() -> None:
    comparison = compare_garp_runs()
    _print_dataframe(comparison, "GARP experiment comparison")


def _print_dataframe(frame: pd.DataFrame, title: str) -> None:
    console.print(f"\n[bold]{title}[/]")
    if frame.empty:
        console.print("[yellow]No rows.[/]")
        return
    priority = [
        "rank",
        "ticker",
        "strategy",
        "score",
        "total_return",
        "buy_hold_return",
        "benchmark_total_return",
        "alpha",
        "alpha_total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "trades",
        "report",
    ]
    param_columns = [column for column in frame.columns if str(column).startswith("param_")][:4]
    selected_columns = [column for column in priority if column in frame.columns] + param_columns
    if len(frame.columns) > 12 and selected_columns:
        frame = frame[selected_columns]
    if len(frame.columns) > 12:
        frame = frame.iloc[:, :12]
    table = Table(show_header=True, header_style="bold")
    for column in frame.columns:
        table.add_column(str(column))
    for _, row in frame.head(30).iterrows():
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        table.add_row(*rendered)
    console.print(table)


if __name__ == "__main__":
    app()
