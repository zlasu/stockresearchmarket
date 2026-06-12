from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots

from stockresearchmarket.engine.metrics import performance_metrics

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class StrategyRun:
    name: str
    returns: pd.Series
    weights: pd.DataFrame

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod().rename(self.name)


def main() -> None:
    output_dir = Path("experiments") / "sp500_current_price_tests" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=10)
    constituents = fetch_current_sp500_constituents()
    constituents.to_csv(output_dir / "current_sp500_constituents.csv", index=False)

    tickers = sorted(set(constituents["yf_ticker"].tolist() + ["SPY", "QQQ", "RSP"]))
    close = load_yfinance_close(tickers, start=start, end=end, output_dir=output_dir)
    quality = data_quality(close, start=start, end=end)
    quality.to_csv(output_dir / "data_quality.csv", index=False)

    eligible = quality.loc[
        quality["ticker"].isin(constituents["yf_ticker"])
        & (quality["rows"] >= 252 * 7)
        & (quality["missing_fraction"] <= 0.10),
        "ticker",
    ].tolist()
    eligible_close = close[eligible].dropna(how="all").ffill(limit=5)
    benchmark_close = close[[column for column in ["SPY", "QQQ", "RSP"] if column in close.columns]].dropna(how="all").ffill()
    runs = [
        run_equal_weight(eligible_close, "current_sp500_equal_weight", rebalance="ME"),
        run_momentum(eligible_close, "momentum_12_1_top50", lookback=252, skip=21, top_n=50, rebalance="ME"),
        run_momentum(eligible_close, "momentum_6m_top50", lookback=126, skip=0, top_n=50, rebalance="ME"),
        run_inverse_vol(eligible_close, "inverse_vol_top100", lookback=63, top_n=100, rebalance="ME"),
    ]
    benchmark_runs = [buy_hold(benchmark_close[column], column) for column in benchmark_close.columns]

    summary = summarize_runs(runs + benchmark_runs)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    write_outputs(output_dir, runs, benchmark_runs, eligible, quality)
    print(f"Output: {output_dir}")
    print(summary.to_string(index=False))


def fetch_current_sp500_constituents() -> pd.DataFrame:
    response = requests.get(
        WIKIPEDIA_SP500_URL,
        headers={"User-Agent": "StockResearchMarket/0.1 research script"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    table = tables[0].copy()
    table = table.rename(columns={"Symbol": "ticker", "Security": "security", "GICS Sector": "sector", "GICS Sub-Industry": "industry"})
    table["yf_ticker"] = table["ticker"].astype(str).str.replace(".", "-", regex=False)
    table["source_url"] = WIKIPEDIA_SP500_URL
    return table[["ticker", "yf_ticker", "security", "sector", "industry", "source_url"]]


def load_yfinance_close(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, output_dir: Path) -> pd.DataFrame:
    cache_path = Path("data/historical") / f"current_sp500_yfinance_close_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    if cache_path.exists():
        close = pd.read_parquet(cache_path)
    else:
        chunks = []
        for chunk in _chunks(tickers, 80):
            raw = yf.download(
                tickers=chunk,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                field = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
                chunk_close = raw[field].copy()
            else:
                chunk_close = raw[["Close"]].rename(columns={"Close": chunk[0]})
            chunks.append(chunk_close)
        close = pd.concat(chunks, axis=1).sort_index()
        close = close.loc[:, ~close.columns.duplicated()]
        close.to_parquet(cache_path)
    close.to_csv(output_dir / "close_panel.csv")
    return close


def data_quality(close: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    expected = len(pd.bdate_range(start, end))
    for ticker in close.columns:
        series = close[ticker].dropna()
        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(series)),
                "start": series.index.min().date().isoformat() if not series.empty else None,
                "end": series.index.max().date().isoformat() if not series.empty else None,
                "missing_fraction": float(1 - len(series) / expected) if expected else 1.0,
                "first_close": float(series.iloc[0]) if not series.empty else np.nan,
                "last_close": float(series.iloc[-1]) if not series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_fraction", "ticker"])


def run_equal_weight(close: pd.DataFrame, name: str, rebalance: str = "ME") -> StrategyRun:
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for date in _rebalance_dates(close.index, rebalance):
        weights.loc[date] = 0.0
        available = close.loc[:date].iloc[-1].dropna().index
        if len(available):
            weights.loc[date, available] = 1 / len(available)
    return _simulate(close, weights, name)


def run_momentum(close: pd.DataFrame, name: str, lookback: int, skip: int, top_n: int, rebalance: str = "ME") -> StrategyRun:
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for date in _rebalance_dates(close.index, rebalance):
        weights.loc[date] = 0.0
        history = close.loc[:date]
        signal_close = history.iloc[:-skip] if skip else history
        if len(signal_close) <= lookback:
            continue
        momentum = signal_close.iloc[-1].div(signal_close.iloc[-lookback - 1]) - 1
        selected = momentum.dropna().sort_values(ascending=False).head(top_n).index
        if len(selected):
            weights.loc[date, selected] = 1 / len(selected)
    return _simulate(close, weights, name)


def run_inverse_vol(close: pd.DataFrame, name: str, lookback: int, top_n: int, rebalance: str = "ME") -> StrategyRun:
    weights = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    returns = close.pct_change()
    trailing_return = close.pct_change(126)
    for date in _rebalance_dates(close.index, rebalance):
        weights.loc[date] = 0.0
        candidates = trailing_return.loc[:date].iloc[-1].dropna().sort_values(ascending=False).head(top_n).index
        vol = returns.loc[:date, candidates].tail(lookback).std(ddof=0).replace(0, np.nan)
        inv = 1 / vol.dropna()
        if inv.sum() > 0:
            weights.loc[date, inv.index] = inv / inv.sum()
    return _simulate(close, weights, name)


def buy_hold(close: pd.Series, name: str) -> StrategyRun:
    returns = close.pct_change().fillna(0.0)
    weights = pd.DataFrame({name: pd.Series(1.0, index=close.index)})
    return StrategyRun(name=name, returns=returns.rename(name), weights=weights)


def _simulate(close: pd.DataFrame, weights: pd.DataFrame, name: str, cost_bps: float = 3.5) -> StrategyRun:
    weights = weights.ffill().fillna(0.0)
    row_sums = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sums.where(row_sums <= 1.0, row_sums), axis=0).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    effective_weights = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    portfolio_returns = (effective_weights * returns).sum(axis=1) - turnover * (cost_bps / 10_000)
    return StrategyRun(name=name, returns=portfolio_returns.rename(name), weights=weights)


def summarize_runs(runs: list[StrategyRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        equity = run.equity
        metrics = performance_metrics(run.returns, equity, int((run.weights.diff().abs().sum(axis=1) > 0).sum()))
        rows.append({"strategy": run.name, **metrics})
    return pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)


def write_outputs(output_dir: Path, runs: list[StrategyRun], benchmarks: list[StrategyRun], eligible: list[str], quality: pd.DataFrame) -> None:
    all_runs = runs + benchmarks
    equity = pd.concat([run.equity for run in all_runs], axis=1)
    returns = pd.concat([run.returns for run in all_runs], axis=1)
    equity.to_csv(output_dir / "equity_curves.csv")
    returns.to_csv(output_dir / "daily_returns.csv")
    for run in runs:
        run.weights.to_csv(output_dir / f"weights_{run.name}.csv")
    metadata = {
        "source": "current S&P 500 constituents from Wikipedia, prices from yfinance",
        "source_url": WIKIPEDIA_SP500_URL,
        "survivorship_bias_warning": "Uses current S&P 500 constituents for the full 10-year history; not historical membership.",
        "eligible_count": len(eligible),
        "downloaded_tickers": int(len(quality)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_html_report(output_dir, equity, returns, quality, metadata)


def write_html_report(output_dir: Path, equity: pd.DataFrame, returns: pd.DataFrame, quality: pd.DataFrame, metadata: dict[str, object]) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=("Equity Curves", "Drawdowns", "Monthly Returns"),
        row_heights=[0.45, 0.25, 0.30],
    )
    for column in equity.columns:
        fig.add_trace(go.Scatter(x=equity.index, y=equity[column], name=column), row=1, col=1)
        dd = equity[column] / equity[column].cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"{column} DD", showlegend=False), row=2, col=1)
    monthly = returns.resample("ME").apply(lambda values: (1 + values).prod() - 1)
    for column in monthly.columns:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[column], name=f"{column} monthly", visible=column == monthly.columns[0]), row=3, col=1)
    fig.update_layout(template="plotly_white", height=1050, title="Current S&P 500 10Y Price Tests", hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    header = f"""
    <section style="font-family:Inter,Arial,sans-serif;max-width:1180px;margin:24px auto 8px;">
      <h1 style="margin:0 0 8px;">Current S&P 500 10Y Price Tests</h1>
      <p style="color:#4b5563;margin:0;">{metadata["survivorship_bias_warning"]}</p>
      <p style="color:#4b5563;">Eligible tickers: {metadata["eligible_count"]} / downloaded columns: {metadata["downloaded_tickers"]}</p>
    </section>
    """
    (output_dir / "report.html").write_text(header + fig.to_html(full_html=False, include_plotlyjs="cdn"), encoding="utf-8")


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    dates = pd.Series(index=index, data=1).resample(frequency).last().dropna().index
    return [index[index.searchsorted(date, side="right") - 1] for date in dates if index.searchsorted(date, side="right") > 0]


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


if __name__ == "__main__":
    main()
