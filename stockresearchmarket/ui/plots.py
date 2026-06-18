from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from stockresearchmarket.ui.models import ExperimentRecord

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#A3BEFA",
    "gold": "#FFE15B",
    "orange": "#F0986E",
    "olive": "#A3D576",
    "pink": "#F390CA",
    "blue_dark": "#2E4780",
    "orange_dark": "#804126",
}


def render_plot(experiment: ExperimentRecord, plot_type: str, *, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{experiment.id}:{plot_type}:{len(experiment.variants)}".encode()).hexdigest()[:16]
    output = cache_dir / f"{experiment.id}-{plot_type}-{key}.png"
    if output.exists():
        return output
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    _use_theme()
    frame = _metrics_frame(experiment)
    if frame.empty:
        _empty_plot(experiment, output, plot_type.replace("-", " ").title(), "No variants were found for this experiment.")
        return output
    if plot_type == "risk-return":
        _risk_return(frame, experiment, output)
    elif plot_type == "metric-distribution":
        _metric_distribution(frame, experiment, output)
    elif plot_type == "correlation-heatmap":
        _correlation_heatmap(frame, experiment, output)
    elif plot_type == "forest":
        _forest(frame, experiment, output)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    return output


def _metrics_frame(experiment: ExperimentRecord) -> pd.DataFrame:
    rows = []
    for variant in experiment.variants:
        row = {"variant": variant.name, "id": variant.id, "score": variant.computed_score}
        row.update(variant.metrics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column not in {"variant", "id"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        },
    )


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.08, 0.98, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.08, 0.935, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def _save(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def _empty_plot(experiment: ExperimentRecord, output: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    _header(fig, title, f"{experiment.name} | static diagnostic plot")
    ax.text(
        0.5,
        0.52,
        message,
        ha="center",
        va="center",
        fontsize=12,
        color=TOKENS["muted"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        "The dashboard still exposes raw metrics, tables and interactive comparisons when available.",
        ha="center",
        va="center",
        fontsize=9,
        color=TOKENS["muted"],
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    fig.subplots_adjust(top=0.82)
    _save(fig, output)


def _first_available(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns and frame[column].notna().any():
            return column
    return None


def _risk_return(frame: pd.DataFrame, experiment: ExperimentRecord, output: Path) -> None:
    if "drawdown_magnitude" not in frame.columns and "max_drawdown" in frame.columns:
        frame = frame.assign(drawdown_magnitude=frame["max_drawdown"].abs())
    y_metric = _first_available(frame, ["cagr", "total_return", "sharpe", "calmar", "score"])
    if "drawdown_magnitude" not in frame.columns or y_metric is None:
        _empty_plot(
            experiment,
            output,
            "Return versus drawdown",
            "Not enough numeric return and drawdown metrics to render this diagnostic.",
        )
        return
    plot_df = frame.dropna(subset=["drawdown_magnitude", y_metric]).copy()
    if plot_df.empty:
        _empty_plot(
            experiment,
            output,
            "Return versus drawdown",
            "Not enough populated return and drawdown metrics to render this diagnostic.",
        )
        return
    fig, ax = plt.subplots(figsize=(9.5, 6))
    _header(fig, "Return versus drawdown", f"{experiment.name} | each point is a variant; x is max drawdown magnitude.")
    sns.scatterplot(
        data=plot_df,
        x="drawdown_magnitude",
        y=y_metric,
        size="trades" if "trades" in plot_df.columns else None,
        sizes=(60, 360),
        color=COLORS["orange"],
        edgecolor=COLORS["orange_dark"],
        linewidth=0.8,
        alpha=0.72,
        legend=False,
        ax=ax,
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if y_metric in {"cagr", "total_return"}:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel(y_metric.replace("_", " ").title())
    fig.subplots_adjust(top=0.84)
    _save(fig, output)


def _metric_distribution(frame: pd.DataFrame, experiment: ExperimentRecord, output: Path) -> None:
    metric = _first_available(frame, ["sharpe", "cagr", "total_return", "calmar", "score"])
    if metric is None:
        _empty_plot(
            experiment,
            output,
            "Metric distribution",
            "No populated numeric metrics were found for this experiment.",
        )
        return
    plot_df = frame.dropna(subset=[metric])
    fig, ax = plt.subplots(figsize=(9.5, 6))
    _header(fig, f"{metric.replace('_', ' ').title()} distribution", f"{experiment.name} | empirical spread across variants.")
    sns.histplot(data=plot_df, x=metric, bins=min(24, max(6, len(plot_df) // 3)), color=COLORS["blue"], edgecolor=COLORS["blue_dark"], ax=ax)
    if not plot_df.empty:
        ax.axvline(plot_df[metric].median(), color=TOKENS["ink"], linestyle=":", linewidth=1.1)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Variants")
    fig.subplots_adjust(top=0.84)
    _save(fig, output)


def _correlation_heatmap(frame: pd.DataFrame, experiment: ExperimentRecord, output: Path) -> None:
    columns = [column for column in ["cagr", "sharpe", "calmar", "drawdown_magnitude", "avg_annual_turnover", "trades", "capital"] if column in frame]
    columns = [column for column in columns if frame[column].notna().any()]
    if len(columns) < 2:
        _empty_plot(
            experiment,
            output,
            "Metric correlation heatmap",
            "At least two populated numeric metric columns are needed for a correlation heatmap.",
        )
        return
    matrix = frame[columns].corr(numeric_only=True)
    if matrix.empty:
        _empty_plot(
            experiment,
            output,
            "Metric correlation heatmap",
            "At least two populated numeric metric columns are needed for a correlation heatmap.",
        )
        return
    fig, ax = plt.subplots(figsize=(9.5, 7))
    _header(fig, "Metric correlation heatmap", f"{experiment.name} | pairwise correlations across numeric variant metrics.")
    cmap = sns.blend_palette([TOKENS["panel"], "#CEDFFE", COLORS["blue"], COLORS["blue_dark"]], as_cmap=True)
    sns.heatmap(matrix, cmap=cmap, annot=True, fmt=".2f", linewidths=1, linecolor=TOKENS["panel"], ax=ax)
    fig.subplots_adjust(top=0.84)
    _save(fig, output)


def _forest(frame: pd.DataFrame, experiment: ExperimentRecord, output: Path) -> None:
    metric = "computed_score"
    frame = frame.assign(computed_score=frame["score"])
    if frame[metric].isna().all():
        metric = _first_available(frame, ["sharpe", "calmar", "cagr", "total_return"])
    if metric is None:
        _empty_plot(
            experiment,
            output,
            "Top variants by research score",
            "No populated ranking metric was found for this experiment.",
        )
        return
    plot_df = frame.dropna(subset=[metric]).sort_values(metric).tail(20)
    if plot_df.empty:
        _empty_plot(
            experiment,
            output,
            "Top variants by research score",
            "No populated ranking metric was found for this experiment.",
        )
        return
    fig, ax = plt.subplots(figsize=(10, max(5.5, len(plot_df) * 0.34)))
    _header(fig, "Top variants by research score", f"{experiment.name} | deterministic shortlist, not an investment recommendation.")
    sns.scatterplot(data=plot_df, x=metric, y="variant", color=COLORS["olive"], edgecolor="#386411", s=90, ax=ax)
    ax.hlines(plot_df["variant"], 0, plot_df[metric], color="#C5CAD3", linewidth=1.0)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    fig.subplots_adjust(top=0.88)
    _save(fig, output)
