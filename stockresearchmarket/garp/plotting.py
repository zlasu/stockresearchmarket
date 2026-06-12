from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stockresearchmarket.features.indicators import drawdown, rolling_sharpe
from stockresearchmarket.garp.metrics import rolling_beta, top_drawdown_periods
from stockresearchmarket.garp.types import GarpBacktestResult


def write_garp_report(result: GarpBacktestResult) -> Path:
    html_path = result.output_dir / "report.html"
    md_path = result.output_dir / "report.md"
    fig = _tear_sheet(result)
    summary_html = _summary_html(result)
    html_path.write_text(summary_html + fig.to_html(full_html=False, include_plotlyjs="cdn"), encoding="utf-8")
    md_path.write_text(_markdown_report(result), encoding="utf-8")
    return html_path


def _tear_sheet(result: GarpBacktestResult) -> go.Figure:
    benchmark = result.benchmark_returns.iloc[:, 0] if not result.benchmark_returns.empty else pd.Series(dtype=float)
    beta = rolling_beta(result.returns, benchmark) if not benchmark.empty else pd.Series(dtype=float)
    sector_exposure = _sector_exposure(result.holdings)
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.045,
        subplot_titles=(
            "Equity Curve vs Benchmarks",
            "Underwater / Drawdown",
            "Rolling 6M Sharpe",
            "Rolling 6M Beta vs Primary Benchmark",
            "Monthly Returns",
            "Sector Exposure",
        ),
        row_heights=[0.26, 0.16, 0.14, 0.14, 0.14, 0.16],
    )
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="GARP", line={"width": 2.4}), row=1, col=1)
    for column in result.benchmark_equity.columns:
        fig.add_trace(go.Scatter(x=result.benchmark_equity.index, y=result.benchmark_equity[column], name=column, line={"width": 1.5}), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.equity.index, y=drawdown(result.equity), name="Drawdown", fill="tozeroy"), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.returns.index, y=rolling_sharpe(result.returns, 126), name="Rolling Sharpe"), row=3, col=1)
    if not beta.empty:
        fig.add_trace(go.Scatter(x=beta.index, y=beta, name="Rolling Beta"), row=4, col=1)
    fig.add_trace(go.Bar(x=result.monthly_returns.index, y=result.monthly_returns, name="Monthly return"), row=5, col=1)
    for sector in sector_exposure.columns[:12]:
        fig.add_trace(go.Scatter(x=sector_exposure.index, y=sector_exposure[sector], stackgroup="one", name=sector), row=6, col=1)
    fig.update_layout(
        title=f"{result.config.get('experiment', {}).get('name', result.experiment_id)}",
        template="plotly_white",
        height=1320,
        hovermode="x unified",
        legend_orientation="h",
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=5, col=1)
    fig.update_yaxes(tickformat=".0%", row=6, col=1)
    return fig


def _summary_html(result: GarpBacktestResult) -> str:
    metrics = result.metrics
    cards = "".join(
        _card(label, value, pct)
        for label, value, pct in [
            ("CAGR", metrics.get("cagr"), True),
            ("Sharpe", metrics.get("sharpe"), False),
            ("Calmar", metrics.get("calmar"), False),
            ("Max DD", metrics.get("max_drawdown"), True),
            ("Alpha", metrics.get("alpha_total_return"), True),
            ("Turnover", metrics.get("avg_annual_turnover"), False),
        ]
    )
    limitations = "".join(f"<li>{item}</li>" for item in result.limitations[:8])
    return f"""
    <section style="font-family:Inter,Arial,sans-serif;max-width:1180px;margin:24px auto 8px;">
      <h1 style="margin:0 0 8px;">{result.config.get("experiment", {}).get("name", result.experiment_id)}</h1>
      <p style="margin:0;color:#4b5563;">{result.equity.index.min().date()} to {result.equity.index.max().date()}</p>
      <div style="display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-top:16px;">{cards}</div>
      <h2 style="font-size:18px;margin:20px 0 8px;">Data limits and bias notes</h2>
      <ul style="margin-top:0;color:#374151;">{limitations}</ul>
    </section>
    """


def _markdown_report(result: GarpBacktestResult) -> str:
    metrics = json.dumps(result.metrics, indent=2, default=str)
    drawdown_rows = top_drawdown_periods(result.equity)
    drawdowns = _markdown_table(drawdown_rows) if not drawdown_rows.empty else "No drawdowns."
    yearly = _markdown_table(result.yearly_returns.to_frame("return").reset_index().rename(columns={"index": "date"}))
    limitations = "\n".join(f"- {item}" for item in result.limitations)
    return f"""# {result.config.get("experiment", {}).get("name", result.experiment_id)}

Measured period: {result.equity.index.min().date()} to {result.equity.index.max().date()}

## Metrics

```json
{metrics}
```

## Top Drawdowns

{drawdowns}

## Yearly Returns

{yearly}

## Data Limits And Bias Notes

{limitations}
"""


def _sector_exposure(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    pivot = holdings.pivot_table(index="date", columns="sector", values="weight", aggfunc="sum").fillna(0.0)
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()


def _card(label: str, value: object, pct: bool) -> str:
    if isinstance(value, int | float):
        rendered = f"{value:.1%}" if pct else f"{value:.2f}"
    else:
        rendered = "n/a"
    return (
        "<div style='border:1px solid #dbe1ea;border-radius:8px;padding:10px;background:#fbfcfe;'>"
        f"<div style='font-size:11px;color:#586272;text-transform:uppercase;'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:#111827;margin-top:4px;'>{rendered}</div>"
        "</div>"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                values.append(value.date().isoformat())
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
