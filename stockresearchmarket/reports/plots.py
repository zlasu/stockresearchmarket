from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stockresearchmarket.engine.vectorized import BacktestResult, PortfolioBacktestResult
from stockresearchmarket.features.indicators import rolling_sharpe


def write_single_asset_report(result: BacktestResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.concat([result.equity, result.benchmark_equity], axis=1)
    returns = pd.concat([result.returns, result.benchmark_returns], axis=1)
    summary = _summary_payload(result)
    equity.to_csv(output_dir / "equity_curve.csv")
    returns.to_csv(output_dir / "daily_returns.csv")
    result.position.to_csv(output_dir / "position.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    html_path = output_dir / "report.html"
    _write_html(html_path, _single_asset_figure(result), summary)
    return html_path


def write_portfolio_report(result: PortfolioBacktestResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.concat([result.equity, result.benchmark_equity], axis=1)
    returns = pd.concat([result.returns, result.benchmark_returns], axis=1)
    summary = _summary_payload(result)
    equity.to_csv(output_dir / "equity_curve.csv")
    returns.to_csv(output_dir / "daily_returns.csv")
    result.weights.to_csv(output_dir / "weights.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    html_path = output_dir / "report.html"
    _write_html(html_path, _portfolio_figure(result), summary)
    return html_path


def write_optimizer_report(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "optimizer_results.csv", index=False)
    html_path = output_dir / "optimizer.html"
    figure = go.Figure()
    if not summary.empty:
        figure.add_trace(
            go.Scatter(
                x=summary["max_drawdown"],
                y=summary["cagr"],
                mode="markers",
                marker={"size": 9, "color": summary["sharpe"], "colorscale": "Viridis", "showscale": True},
                text=summary.apply(lambda row: f"rank={row['rank']} score={row['score']:.2f}", axis=1),
                name="candidates",
            )
        )
    figure.update_layout(
        title="Optimization Map",
        template="plotly_white",
        xaxis_title="Max drawdown",
        yaxis_title="CAGR",
        height=620,
    )
    _write_html(html_path, figure, {"rows": len(summary), "columns": list(summary.columns)})
    return html_path


def _single_asset_figure(result: BacktestResult) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=("Equity vs Buy & Hold", "Drawdown", "Rolling 63D Sharpe", "Monthly Strategy Returns"),
        row_heights=[0.42, 0.2, 0.18, 0.2],
    )
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="Strategy", line={"width": 2.4}), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=result.benchmark_equity.index, y=result.benchmark_equity, name="Buy & Hold", line={"width": 1.8}),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown, name="Strategy drawdown", fill="tozeroy"), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.returns.index, y=rolling_sharpe(result.returns), name="Rolling Sharpe"), row=3, col=1)
    monthly = result.returns.resample("ME").apply(lambda values: (1 + values).prod() - 1)
    fig.add_trace(go.Bar(x=monthly.index, y=monthly, name="Monthly return"), row=4, col=1)
    fig.update_layout(
        title=f"{result.ticker} {result.strategy} | alpha {result.alpha_total_return:.1%}",
        template="plotly_white",
        height=980,
        legend_orientation="h",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    return fig


def _portfolio_figure(result: PortfolioBacktestResult) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=("Portfolio Equity vs Equal Weight", "Drawdown", "Rolling 63D Sharpe", "Monthly Portfolio Returns"),
        row_heights=[0.42, 0.2, 0.18, 0.2],
    )
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="Strategy", line={"width": 2.4}), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=result.benchmark_equity.index, y=result.benchmark_equity, name="Equal Weight", line={"width": 1.8}),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown, name="Strategy drawdown", fill="tozeroy"), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.returns.index, y=rolling_sharpe(result.returns), name="Rolling Sharpe"), row=3, col=1)
    monthly = result.returns.resample("ME").apply(lambda values: (1 + values).prod() - 1)
    fig.add_trace(go.Bar(x=monthly.index, y=monthly, name="Monthly return"), row=4, col=1)
    fig.update_layout(
        title=f"{result.strategy} | alpha {result.alpha_total_return:.1%}",
        template="plotly_white",
        height=980,
        legend_orientation="h",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    return fig


def _summary_payload(result: BacktestResult | PortfolioBacktestResult) -> dict[str, Any]:
    return {
        "strategy": result.strategy,
        "ticker": getattr(result, "ticker", "portfolio"),
        "params": result.params,
        "metrics": result.metrics,
        "benchmark_metrics": result.benchmark_metrics,
        "alpha_total_return": result.alpha_total_return,
        "start": result.equity.index.min().date().isoformat() if not result.equity.empty else None,
        "end": result.equity.index.max().date().isoformat() if not result.equity.empty else None,
    }


def _write_html(path: Path, figure: go.Figure, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    benchmark = summary.get("benchmark_metrics", {})
    header = f"""
    <section style="font-family:Inter,Arial,sans-serif;max-width:1180px;margin:24px auto 8px;">
      <h1 style="margin:0 0 8px;">{summary.get("strategy")} research report</h1>
      <p style="margin:0;color:#445;">{summary.get("ticker")} | {summary.get("start")} to {summary.get("end")}</p>
      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px;">
        {_metric_card("Total Return", metrics.get("total_return"), True)}
        {_metric_card("Buy & Hold", benchmark.get("total_return"), True)}
        {_metric_card("Sharpe", metrics.get("sharpe"), False)}
        {_metric_card("Max Drawdown", metrics.get("max_drawdown"), True)}
      </div>
    </section>
    """
    html = header + figure.to_html(full_html=False, include_plotlyjs="cdn")
    path.write_text(html, encoding="utf-8")


def _metric_card(label: str, value: Any, pct: bool) -> str:
    if isinstance(value, int | float):
        rendered = f"{value:.1%}" if pct else f"{value:.2f}"
    else:
        rendered = "n/a"
    return (
        "<div style='border:1px solid #dbe1ea;border-radius:8px;padding:12px;background:#fbfcfe;'>"
        f"<div style='font-size:12px;color:#586272;text-transform:uppercase;'>{label}</div>"
        f"<div style='font-size:24px;font-weight:700;color:#111827;margin-top:4px;'>{rendered}</div>"
        "</div>"
    )

