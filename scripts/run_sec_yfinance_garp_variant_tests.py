from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stockresearchmarket.garp.autoresearch import rank_experiments
from stockresearchmarket.garp.backtest import run_garp_backtest
from stockresearchmarket.garp.config import deep_merge, load_garp_config

VARIANTS: list[dict[str, Any]] = [
    {
        "id": "baseline_top10_monthly",
        "name": "Baseline top 10 monthly",
        "overrides": {"portfolio": {"top_n": 10, "hold_until_rank": 15}, "rebalance": {"frequency": "ME"}},
    },
    {
        "id": "concentrated_top5",
        "name": "Concentrated top 5",
        "overrides": {"portfolio": {"top_n": 5, "hold_until_rank": 8, "max_position_weight": 0.20}},
    },
    {
        "id": "broader_top15",
        "name": "Broader top 15",
        "overrides": {"portfolio": {"top_n": 15, "hold_until_rank": 20, "max_position_weight": 0.09}},
    },
    {
        "id": "quarterly_rebalance",
        "name": "Quarterly rebalance",
        "overrides": {"rebalance": {"frequency": "QE"}},
    },
    {
        "id": "sma200_market_filter",
        "name": "SMA200 market filter",
        "overrides": {
            "risk_management": {
                "market_filter": {
                    "enabled": True,
                    "benchmark": "SPY",
                    "sma_window": 200,
                    "defensive_weight": 1.0,
                    "cash_asset": "CASH",
                }
            }
        },
    },
    {
        "id": "vol_target_10",
        "name": "10% volatility target",
        "overrides": {
            "risk_management": {
                "volatility_target": {"enabled": True, "target_volatility": 0.10, "lookback_days": 63}
            }
        },
    },
    {
        "id": "vol_target_15",
        "name": "15% volatility target",
        "overrides": {
            "risk_management": {
                "volatility_target": {"enabled": True, "target_volatility": 0.15, "lookback_days": 63}
            }
        },
    },
    {
        "id": "turnover_band",
        "name": "Turnover band",
        "overrides": {"portfolio": {"top_n": 10, "hold_until_rank": 20, "turnover_threshold": 0.10}},
    },
    {
        "id": "momentum_tilt",
        "name": "Momentum tilt",
        "overrides": {"factors": {"base_weights": {"growth": 0.25, "quality": 0.20, "value": 0.10, "momentum": 0.45, "revisions": 0.0}}},
    },
    {
        "id": "quality_growth_tilt",
        "name": "Quality-growth tilt",
        "overrides": {"factors": {"base_weights": {"growth": 0.40, "quality": 0.35, "value": 0.10, "momentum": 0.15, "revisions": 0.0}}},
    },
    {
        "id": "value_quality_tilt",
        "name": "Value-quality tilt",
        "overrides": {"factors": {"base_weights": {"growth": 0.20, "quality": 0.35, "value": 0.30, "momentum": 0.15, "revisions": 0.0}}},
    },
    {
        "id": "risk_off_combo",
        "name": "SMA200 + 10% volatility target",
        "overrides": {
            "risk_management": {
                "market_filter": {
                    "enabled": True,
                    "benchmark": "SPY",
                    "sma_window": 200,
                    "defensive_weight": 1.0,
                    "cash_asset": "CASH",
                },
                "volatility_target": {"enabled": True, "target_volatility": 0.10, "lookback_days": 63},
            }
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEC/yfinance GARP variant tests and write comparison charts.")
    parser.add_argument("--base-experiment", default="sec_yfinance_largecap")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--max-variants", type=int, default=len(VARIANTS))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-root", default="experiments/garp_sec_yfinance_variants")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / stamp
    config_dir = output_dir / "configs"
    runs_dir = output_dir / "runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_garp_config(args.base_experiment)
    base_overrides = {key: deepcopy(value) for key, value in base_config.items() if key != "experiment"}
    rows: list[dict[str, Any]] = []
    first_result = None

    for variant in VARIANTS[: args.max_variants]:
        variant_path = config_dir / f"{variant['id']}.yaml"
        overrides = deep_merge(base_overrides, variant["overrides"])
        payload = {
            "id": f"sec_yf_{variant['id']}",
            "name": variant["name"],
            "description": f"{variant['name']} on SEC EDGAR fundamentals and yfinance prices.",
            "overrides": overrides,
        }
        variant_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        result = run_garp_backtest(
            variant_path,
            years=args.years,
            refresh=args.refresh,
            output_root=runs_dir,
        )
        if first_result is None:
            first_result = result
        rows.append(
            {
                "variant_id": variant["id"],
                "name": variant["name"],
                "config_path": str(variant_path),
                "output_dir": str(result.output_dir),
                **result.metrics,
            }
        )

    leaderboard = rank_experiments(pd.DataFrame(rows))
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    (output_dir / "leaderboard.json").write_text(json.dumps(leaderboard.to_dict("records"), indent=2, default=str), encoding="utf-8")
    equity = _collect_equity_curves(leaderboard)
    equity.to_csv(output_dir / "equity_curves.csv")
    _write_plotly_report(output_dir, leaderboard, equity)
    _write_equity_svg(output_dir / "equity_curves.svg", equity, leaderboard)
    _write_return_bars_svg(output_dir / "return_bars.svg", leaderboard)
    _write_markdown_summary(output_dir, leaderboard, equity)
    print(output_dir)


def _collect_equity_curves(leaderboard: pd.DataFrame) -> pd.DataFrame:
    curves: list[pd.Series] = []
    benchmark_curves: list[pd.Series] = []
    for _, row in leaderboard.iterrows():
        path = Path(str(row["output_dir"]))
        equity = pd.read_csv(path / "equity_curve.csv", index_col=0, parse_dates=True).iloc[:, 0]
        name = str(row["variant_id"])
        curves.append((equity / equity.iloc[0]).rename(name))
        if not benchmark_curves:
            benchmarks = pd.read_csv(path / "benchmark_equity.csv", index_col=0, parse_dates=True)
            for column in benchmarks.columns:
                benchmark_curves.append((benchmarks[column] / benchmarks[column].iloc[0]).rename(column))
    return pd.concat(curves + benchmark_curves, axis=1).sort_index().ffill().dropna(how="all")


def _write_plotly_report(output_dir: Path, leaderboard: pd.DataFrame, equity: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError:
        return
    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=("Normalized equity curves", "CAGR by variant", "Max drawdown by variant"),
        row_heights=[0.52, 0.24, 0.24],
    )
    top_ids = leaderboard["variant_id"].head(8).tolist()
    columns = [column for column in top_ids + ["SPY", "QQQ"] if column in equity.columns]
    for column in columns:
        width = 2.6 if column in {"SPY", "QQQ"} else 1.8
        dash = "dash" if column in {"SPY", "QQQ"} else None
        fig.add_trace(go.Scatter(x=equity.index, y=equity[column], name=column, line={"width": width, "dash": dash}), row=1, col=1)
    fig.add_trace(go.Bar(x=leaderboard["variant_id"], y=leaderboard["cagr"], name="CAGR"), row=2, col=1)
    fig.add_trace(go.Bar(x=leaderboard["variant_id"], y=leaderboard["max_drawdown"], name="Max drawdown"), row=3, col=1)
    fig.update_layout(template="plotly_white", height=1050, title="SEC EDGAR + yfinance GARP variant comparison", hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    table_html = leaderboard.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    html = "<html><body style='font-family:Inter,Arial,sans-serif;max-width:1220px;margin:24px auto;'>"
    html += "<h1>SEC EDGAR + yfinance GARP variant comparison</h1>"
    html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "<h2>Leaderboard</h2>" + table_html + "</body></html>"
    (output_dir / "comparison_report.html").write_text(html, encoding="utf-8")


def _write_markdown_summary(output_dir: Path, leaderboard: pd.DataFrame, equity: pd.DataFrame) -> None:
    period_start = equity.index.min().date().isoformat()
    period_end = equity.index.max().date().isoformat()
    columns = ["variant_id", "total_return", "benchmark_total_return", "alpha_total_return", "cagr", "sharpe", "max_drawdown", "trades", "output_dir"]
    table = _markdown_table(leaderboard[[column for column in columns if column in leaderboard.columns]])
    text = f"""# SEC EDGAR + yfinance GARP Variant Tests

Measured period: {period_start} to {period_end}

Data: yfinance daily adjusted prices, SEC EDGAR companyfacts fundamentals using filing dates as point-in-time `as_of_date`, analyst estimates disabled.

## Leaderboard

{table}

## Artifacts

- `comparison_report.html`
- `equity_curves.svg`
- `return_bars.svg`
- `leaderboard.csv`
- `equity_curves.csv`
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_equity_svg(path: Path, equity: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    width, height = 1180, 650
    margin = {"left": 70, "right": 190, "top": 52, "bottom": 62}
    columns = [column for column in leaderboard["variant_id"].head(6).tolist() + ["SPY", "QQQ"] if column in equity.columns]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#0891b2", "#111827", "#6b7280"]
    x_values = pd.Series(equity.index.view("int64"), index=equity.index)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min = float(equity[columns].min().min())
    y_max = float(equity[columns].max().max())
    y_pad = (y_max - y_min) * 0.08
    y_min, y_max = max(0.0, y_min - y_pad), y_max + y_pad

    def sx(value: float) -> float:
        return margin["left"] + (value - x_min) / (x_max - x_min) * (width - margin["left"] - margin["right"])

    def sy(value: float) -> float:
        return height - margin["bottom"] - (value - y_min) / (y_max - y_min) * (height - margin["top"] - margin["bottom"])

    parts = [_svg_header(width, height, "Normalized equity curves")]
    parts.append(_axis(width, height, margin))
    for tick in range(6):
        y = y_min + (y_max - y_min) * tick / 5
        py = sy(y)
        parts.append(f"<line x1='{margin['left']}' y1='{py:.1f}' x2='{width - margin['right']}' y2='{py:.1f}' stroke='#e5e7eb'/>")
        parts.append(f"<text x='18' y='{py + 4:.1f}' font-size='12' fill='#4b5563'>{y:.1f}x</text>")
    for idx, column in enumerate(columns):
        series = equity[column].dropna()
        points = " ".join(f"{sx(float(x_values.loc[date])):.1f},{sy(float(value)):.1f}" for date, value in series.items())
        dash = " stroke-dasharray='7 5'" if column in {"SPY", "QQQ"} else ""
        parts.append(f"<polyline points='{points}' fill='none' stroke='{colors[idx % len(colors)]}' stroke-width='2.2'{dash}/>")
        y_legend = margin["top"] + 26 * idx
        parts.append(f"<line x1='{width - 165}' y1='{y_legend}' x2='{width - 132}' y2='{y_legend}' stroke='{colors[idx % len(colors)]}' stroke-width='2.8'{dash}/>")
        parts.append(f"<text x='{width - 124}' y='{y_legend + 4}' font-size='12' fill='#111827'>{escape(str(column))}</text>")
    start_label = equity.index.min().date().isoformat()
    end_label = equity.index.max().date().isoformat()
    parts.append(f"<text x='{margin['left']}' y='{height - 24}' font-size='12' fill='#4b5563'>{start_label}</text>")
    parts.append(f"<text x='{width - margin['right'] - 72}' y='{height - 24}' font-size='12' fill='#4b5563'>{end_label}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_return_bars_svg(path: Path, leaderboard: pd.DataFrame) -> None:
    frame = leaderboard.head(12).copy()
    width, height = 1180, 620
    margin = {"left": 78, "right": 40, "top": 52, "bottom": 170}
    values = frame["total_return"].astype(float)
    min_value, max_value = min(0.0, float(values.min())), max(0.0, float(values.max()))
    y_pad = (max_value - min_value) * 0.10 or 0.1
    min_value, max_value = min_value - y_pad, max_value + y_pad
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    bar_w = plot_w / max(len(frame), 1) * 0.68

    def sy(value: float) -> float:
        return margin["top"] + (max_value - value) / (max_value - min_value) * plot_h

    parts = [_svg_header(width, height, "Total return by variant")]
    parts.append(_axis(width, height, margin))
    zero_y = sy(0)
    parts.append(f"<line x1='{margin['left']}' y1='{zero_y:.1f}' x2='{width - margin['right']}' y2='{zero_y:.1f}' stroke='#6b7280'/>")
    for tick in range(5):
        value = min_value + (max_value - min_value) * tick / 4
        py = sy(value)
        parts.append(f"<line x1='{margin['left']}' y1='{py:.1f}' x2='{width - margin['right']}' y2='{py:.1f}' stroke='#e5e7eb'/>")
        parts.append(f"<text x='20' y='{py + 4:.1f}' font-size='12' fill='#4b5563'>{value:.0%}</text>")
    for idx, (_, row) in enumerate(frame.iterrows()):
        value = float(row["total_return"])
        x_center = margin["left"] + plot_w * (idx + 0.5) / len(frame)
        y = sy(max(value, 0))
        h = abs(sy(value) - zero_y)
        fill = "#2563eb" if value >= 0 else "#dc2626"
        parts.append(f"<rect x='{x_center - bar_w / 2:.1f}' y='{min(y, zero_y):.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{fill}' opacity='0.86'/>")
        parts.append(f"<text x='{x_center:.1f}' y='{min(y, zero_y) - 8:.1f}' font-size='11' fill='#111827' text-anchor='middle'>{value:.0%}</text>")
        label = escape(str(row["variant_id"]))
        parts.append(f"<text transform='translate({x_center - 4:.1f},{height - margin['bottom'] + 18}) rotate(55)' font-size='11' fill='#111827'>{label}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text x='70' y='32' font-size='22' font-family='Arial' font-weight='700' fill='#111827'>{escape(title)}</text>"
    )


def _axis(width: int, height: int, margin: dict[str, int]) -> str:
    return (
        f"<line x1='{margin['left']}' y1='{height - margin['bottom']}' x2='{width - margin['right']}' y2='{height - margin['bottom']}' stroke='#9ca3af'/>"
        f"<line x1='{margin['left']}' y1='{margin['top']}' x2='{margin['left']}' y2='{height - margin['bottom']}' stroke='#9ca3af'/>"
    )


if __name__ == "__main__":
    main()
