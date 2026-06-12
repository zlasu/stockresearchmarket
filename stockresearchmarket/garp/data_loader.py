from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockresearchmarket.data.sources import DataRequest, load_close_panel, load_history
from stockresearchmarket.garp.adapters.sec_edgar import SecEdgarFundamentalsAdapter
from stockresearchmarket.garp.config import get_config
from stockresearchmarket.garp.types import FactorStatus, GarpDataBundle

SECTORS = [
    "Information Technology",
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Financials",
    "Health Care",
    "Industrials",
    "Energy",
]


def load_garp_data(
    config: dict[str, Any],
    provider: str | None = None,
    years: int | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> GarpDataBundle:
    price_provider = provider or str(get_config(config, "data.price_provider", "synthetic"))
    tickers = [str(item).upper() for item in get_config(config, "universe.tickers", [])]
    benchmarks = [str(item).upper() for item in get_config(config, "data.benchmark_tickers", ["SPY", "QQQ"])]
    cash_asset = str(get_config(config, "data.cash_asset", "CASH")).upper()
    all_tickers = list(dict.fromkeys(tickers + benchmarks + ([] if cash_asset == "CASH" else [cash_asset])))
    start_value, end_value = _date_bounds(config, years, start, end)
    frames = load_history(
        DataRequest(
            tickers=all_tickers,
            start=start_value,
            end=end_value,
            provider=price_provider,
            cache_dir=Path(get_config(config, "data.cache_dir", "data/historical")),
            refresh=refresh,
        )
    )
    close = load_close_panel(frames)
    if cash_asset == "CASH":
        close["CASH"] = 1.0

    fundamentals_provider = str(get_config(config, "data.fundamentals_provider", "synthetic"))
    estimates_provider = str(get_config(config, "data.estimates_provider", "synthetic"))
    sectors = load_sectors(
        tickers=tickers,
        provider=str(get_config(config, "data.sector_provider", "synthetic")),
        cache_dir=Path(get_config(config, "data.fundamentals_cache_dir", "data/fundamentals")),
        refresh=refresh,
    )
    fundamentals, fundamental_status = load_fundamentals(
        tickers=tickers,
        dates=close.index,
        provider=fundamentals_provider,
        sectors=sectors,
        cache_dir=Path(get_config(config, "data.fundamentals_cache_dir", "data/fundamentals")),
        close=close,
        sec_user_agent=get_config(config, "data.sec_user_agent"),
        refresh=refresh,
    )
    estimates, estimates_status = load_estimates(
        tickers=tickers,
        dates=close.index,
        provider=estimates_provider,
        cache_dir=Path(get_config(config, "data.fundamentals_cache_dir", "data/fundamentals")),
        refresh=refresh,
    )
    return GarpDataBundle(
        frames=frames,
        close=close,
        fundamentals=fundamentals,
        estimates=estimates,
        sectors=sectors,
        availability=fundamental_status + estimates_status,
        source_notes=[
            f"prices={price_provider}",
            f"fundamentals={fundamentals_provider}",
            f"estimates={estimates_provider}",
            "Synthetic fundamentals/estimates are for framework validation, not investable evidence."
            if fundamentals_provider == "synthetic" or estimates_provider == "synthetic"
            else "",
        ],
    )


def load_fundamentals(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    provider: str,
    sectors: dict[str, str],
    cache_dir: Path,
    close: pd.DataFrame | None = None,
    sec_user_agent: str | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[FactorStatus]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"fundamentals_{provider}_{_date_part(dates)}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path), _fundamental_status(provider)
    if provider == "synthetic":
        frame = synthetic_fundamentals(tickers, dates, sectors)
        frame.to_parquet(cache_path)
        return frame, _fundamental_status(provider)
    if provider == "sec_edgar":
        adapter = SecEdgarFundamentalsAdapter(cache_dir=cache_dir / "sec_edgar", user_agent=sec_user_agent)
        frame = adapter.load(tickers, close=close, refresh=refresh)
        if not frame.empty:
            frame["sector"] = frame["ticker"].map(sectors).fillna("Unknown")
            frame.to_parquet(cache_path)
        return frame, _fundamental_status(provider)
    if provider == "csv":
        path = cache_dir / "fundamentals.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["as_of_date", "period_end"])
            return frame, _fundamental_status(provider)
    return pd.DataFrame(), [
        FactorStatus("growth", "*", "unavailable", f"fundamentals provider {provider} is not configured", provider),
        FactorStatus("quality", "*", "unavailable", f"fundamentals provider {provider} is not configured", provider),
        FactorStatus("value", "*", "unavailable", f"fundamentals provider {provider} is not configured", provider),
    ]


def load_estimates(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    provider: str,
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[FactorStatus]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"estimates_{provider}_{_date_part(dates)}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path), _estimates_status(provider)
    if provider in {"none", "unavailable", ""}:
        return pd.DataFrame(), _estimates_status(provider)
    if provider == "synthetic":
        frame = synthetic_estimates(tickers, dates)
        frame.to_parquet(cache_path)
        return frame, _estimates_status(provider)
    if provider == "csv":
        path = cache_dir / "estimates.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["as_of_date"])
            return frame, _estimates_status(provider)
    return pd.DataFrame(), [
        FactorStatus("revisions", "*", "unavailable", f"estimates provider {provider} is not configured", provider)
    ]


def synthetic_sectors(tickers: list[str]) -> dict[str, str]:
    return {ticker: SECTORS[abs(hash(ticker)) % len(SECTORS)] for ticker in tickers}


def load_sectors(tickers: list[str], provider: str, cache_dir: Path, refresh: bool = False) -> dict[str, str]:
    if provider != "yfinance":
        return synthetic_sectors(tickers)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "yfinance_sectors.parquet"
    if path.exists() and not refresh:
        frame = pd.read_parquet(path)
        return dict(zip(frame["ticker"], frame["sector"], strict=False))
    sectors: dict[str, str] = {}
    try:
        import yfinance as yf
    except ModuleNotFoundError:
        return synthetic_sectors(tickers)
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).get_info()
            sectors[ticker] = str(info.get("sector") or "Unknown")
        except Exception:
            sectors[ticker] = "Unknown"
    pd.DataFrame([{"ticker": ticker, "sector": sector} for ticker, sector in sectors.items()]).to_parquet(path)
    return sectors


def synthetic_fundamentals(tickers: list[str], dates: pd.DatetimeIndex, sectors: dict[str, str]) -> pd.DataFrame:
    quarter_ends = pd.date_range(dates.min(), dates.max(), freq="QE")
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        rng = np.random.default_rng(abs(hash(("fundamentals", ticker))) % (2**32))
        quality_bias = rng.normal(0.0, 0.25)
        growth_bias = rng.normal(0.0, 0.20)
        value_bias = rng.normal(0.0, 0.20)
        cap_base = 6e9 * (1 + abs(rng.normal(4, 8)))
        for idx, period_end in enumerate(quarter_ends):
            as_of = period_end + pd.Timedelta(days=45)
            cycle = np.sin(idx / 4 + rng.normal(0, 0.1))
            revenue_growth = np.clip(0.08 + growth_bias + 0.04 * cycle + rng.normal(0, 0.035), -0.35, 0.60)
            eps_growth = np.clip(0.10 + growth_bias * 1.2 + 0.05 * cycle + rng.normal(0, 0.05), -0.50, 0.90)
            ocf_growth = np.clip(0.07 + growth_bias + rng.normal(0, 0.04), -0.45, 0.70)
            pe = np.clip(24 - 9 * value_bias + 12 * growth_bias + rng.normal(0, 4), 5, 90)
            ps = np.clip(5 - 1.8 * value_bias + 2 * growth_bias + rng.normal(0, 1.2), 0.4, 35)
            ev_ebitda = np.clip(15 - 4 * value_bias + rng.normal(0, 3), 3, 60)
            fcf_yield = np.clip(0.035 + 0.025 * value_bias + rng.normal(0, 0.02), -0.08, 0.20)
            roe = np.clip(0.12 + quality_bias * 0.08 + rng.normal(0, 0.03), -0.20, 0.70)
            roic = np.clip(0.10 + quality_bias * 0.07 + rng.normal(0, 0.03), -0.15, 0.60)
            gross_margin = np.clip(0.42 + quality_bias * 0.08 + rng.normal(0, 0.04), 0.05, 0.95)
            net_margin = np.clip(0.12 + quality_bias * 0.04 + rng.normal(0, 0.025), -0.20, 0.50)
            operating_margin = np.clip(0.18 + quality_bias * 0.05 + rng.normal(0, 0.03), -0.20, 0.60)
            debt_equity = np.clip(1.2 - quality_bias + rng.normal(0, 0.35), 0, 5)
            rows.append(
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "as_of_date": as_of,
                    "source": "synthetic",
                    "sector": sectors[ticker],
                    "market_cap": cap_base * (1 + revenue_growth) ** (idx / 4),
                    "revenue_growth_3y": revenue_growth,
                    "revenue_growth_5y": revenue_growth * 0.85,
                    "eps_growth_3y": eps_growth,
                    "eps_growth_5y": eps_growth * 0.80,
                    "operating_cash_flow_growth": ocf_growth,
                    "pe": pe,
                    "forward_pe": pe * np.clip(1 - eps_growth * 0.3, 0.55, 1.3),
                    "ps": ps,
                    "ev_ebitda": ev_ebitda,
                    "fcf_yield": fcf_yield,
                    "roe": roe,
                    "roic": roic,
                    "gross_margin": gross_margin,
                    "net_margin": net_margin,
                    "operating_margin": operating_margin,
                    "debt_equity": debt_equity,
                    "interest_coverage": np.clip(8 + quality_bias * 6 + rng.normal(0, 2), -2, 50),
                    "positive_fcf": float(fcf_yield > 0),
                    "altman_z": np.clip(3 + quality_bias * 1.3 + rng.normal(0, 0.4), -1, 8),
                }
            )
    return pd.DataFrame(rows).sort_values(["as_of_date", "ticker"]).reset_index(drop=True)


def synthetic_estimates(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    month_ends = pd.date_range(dates.min(), dates.max(), freq="ME")
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        rng = np.random.default_rng(abs(hash(("estimates", ticker))) % (2**32))
        revision_bias = rng.normal(0.0, 0.03)
        for idx, as_of in enumerate(month_ends):
            cycle = np.cos(idx / 5 + rng.normal(0, 0.05))
            rows.append(
                {
                    "ticker": ticker,
                    "as_of_date": as_of,
                    "source": "synthetic",
                    "eps_revision_30d": np.clip(revision_bias + 0.02 * cycle + rng.normal(0, 0.015), -0.25, 0.25),
                    "eps_revision_90d": np.clip(revision_bias + 0.03 * cycle + rng.normal(0, 0.020), -0.30, 0.30),
                    "revenue_revision_90d": np.clip(revision_bias + 0.02 * cycle + rng.normal(0, 0.020), -0.25, 0.25),
                    "target_revision_90d": np.clip(revision_bias + 0.025 * cycle + rng.normal(0, 0.025), -0.35, 0.35),
                    "analyst_count": int(np.clip(8 + rng.normal(0, 4), 1, 45)),
                }
            )
    return pd.DataFrame(rows).sort_values(["as_of_date", "ticker"]).reset_index(drop=True)


def latest_snapshot(frame: pd.DataFrame, as_of_date: pd.Timestamp, by: str = "ticker") -> pd.DataFrame:
    if frame.empty:
        return frame
    available = frame[pd.to_datetime(frame["as_of_date"]) <= pd.Timestamp(as_of_date)].copy()
    if available.empty:
        return available
    return available.sort_values("as_of_date").groupby(by, as_index=False).tail(1).set_index(by)


def _fundamental_status(provider: str) -> list[FactorStatus]:
    status = "available" if provider in {"synthetic", "csv", "sec_edgar"} else "unsafe"
    reason = (
        "SEC EDGAR company facts use filing date as as_of_date"
        if provider == "sec_edgar"
        else "point-in-time synthetic/csv rows include as_of_date"
        if status == "available"
        else "provider not point-in-time safe"
    )
    return [
        FactorStatus("growth", "*", status, reason, provider),
        FactorStatus("quality", "*", status, reason, provider),
        FactorStatus("value", "*", status, reason, provider),
    ]


def _estimates_status(provider: str) -> list[FactorStatus]:
    status = "available" if provider in {"synthetic", "csv"} else "unavailable"
    reason = "estimate rows include as_of_date" if status == "available" else "estimate provider not configured or not point-in-time safe"
    return [FactorStatus("revisions", "*", status, reason, provider)]


def _date_bounds(config: dict[str, Any], years: int | None, start: str | None, end: str | None) -> tuple[str | None, str | None]:
    end_value = end or get_config(config, "data.end")
    start_value = start or get_config(config, "data.start")
    if years:
        end_ts = pd.Timestamp(end_value or date.today().isoformat())
        start_value = (end_ts - pd.DateOffset(years=years)).date().isoformat()
        end_value = end_ts.date().isoformat()
    return start_value, end_value


def _date_part(dates: pd.DatetimeIndex) -> str:
    return f"{dates.min().strftime('%Y%m%d')}_{dates.max().strftime('%Y%m%d')}"
