from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


class DataSourceError(RuntimeError):
    """Raised when a market data provider cannot return usable OHLCV data."""


@dataclass(frozen=True)
class DataRequest:
    tickers: list[str]
    start: str | None
    end: str | None
    provider: str
    cache_dir: Path
    raw_dir: Path = Path("data/raw")
    refresh: bool = False


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_history(request: DataRequest) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    request.cache_dir.mkdir(parents=True, exist_ok=True)

    for ticker in request.tickers:
        frames[ticker] = _load_one(ticker, request)
    return frames


def load_close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = {ticker: frame["close"] for ticker, frame in frames.items() if not frame.empty}
    if not close:
        return pd.DataFrame()
    return pd.DataFrame(close).sort_index().dropna(how="all")


def _load_one(ticker: str, request: DataRequest) -> pd.DataFrame:
    cache_path = _cache_path(request.cache_dir, ticker, request.provider, request.start, request.end)
    if cache_path.exists() and not request.refresh:
        return pd.read_parquet(cache_path)

    if request.provider == "synthetic":
        frame = synthetic_ohlcv(ticker, request.start, request.end)
    elif request.provider == "csv":
        frame = _load_csv(ticker, request.raw_dir)
    elif request.provider == "stooq":
        frame = _fetch_stooq(ticker, request.start, request.end)
    elif request.provider == "yfinance":
        frame = _fetch_yfinance(ticker, request.start, request.end)
    else:
        raise DataSourceError(f"Unsupported provider: {request.provider}")

    frame = normalize_ohlcv(frame)
    if frame.empty:
        raise DataSourceError(f"No OHLCV rows returned for {ticker} from {request.provider}")
    frame.to_parquet(cache_path)
    return frame


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        price_level = None
        for level in range(data.columns.nlevels):
            values = {str(value).strip().lower().replace(" ", "_") for value in data.columns.get_level_values(level)}
            if {"open", "high", "low", "close"}.issubset(values):
                price_level = level
                break
        if price_level is None:
            data.columns = ["_".join([str(part) for part in col if part]).strip() for col in data.columns]
        else:
            data.columns = data.columns.get_level_values(price_level)
    rename = {str(col): str(col).strip().lower().replace(" ", "_") for col in data.columns}
    data = data.rename(columns=rename)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    if "date" in data.columns:
        data.index = pd.to_datetime(data.pop("date")).dt.tz_localize(None)

    if "adj_close" in data.columns:
        data["raw_close"] = data.get("close", data["adj_close"])
        data["close"] = data["adj_close"]
    if "volume" not in data.columns:
        data["volume"] = 0.0

    missing = [col for col in ["open", "high", "low", "close"] if col not in data.columns]
    if missing:
        raise DataSourceError(f"Missing required OHLC columns: {missing}")

    data = data.sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    data = data[["open", "high", "low", "close", "volume"] + [col for col in ["raw_close"] if col in data.columns]]
    return data.dropna(subset=["close"])


def synthetic_ohlcv(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start or "2006-01-01")
    end_ts = pd.Timestamp(end or date.today().isoformat())
    index = pd.bdate_range(start_ts, end_ts)
    seed = abs(hash(ticker)) % (2**32)
    rng = np.random.default_rng(seed)
    drift = 0.00025 + (seed % 17) / 100000
    volatility = 0.010 + (seed % 11) / 2500
    shocks = rng.normal(drift, volatility, len(index))
    trend_cycle = 0.0008 * np.sin(np.linspace(0, 18, len(index)))
    close = 100 * np.cumprod(1 + shocks + trend_cycle)
    spread = np.maximum(close * np.abs(rng.normal(0.004, 0.0015, len(index))), 0.01)
    open_ = close * (1 + rng.normal(0, 0.002, len(index)))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(500_000, 5_000_000, len(index))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def _fetch_yfinance(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise DataSourceError("Install yfinance or use provider=stooq/csv/synthetic") from exc

    frame = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if frame.empty:
        raise DataSourceError(f"yfinance returned no rows for {ticker}")
    return frame


def _fetch_stooq(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        from pandas_datareader import data as web
    except ModuleNotFoundError as exc:
        raise DataSourceError("Install pandas-datareader or use provider=yfinance/csv/synthetic") from exc

    symbol = ticker if "." in ticker else f"{ticker}.US"
    frame = web.DataReader(symbol, "stooq", start=start, end=end)
    if frame.empty:
        raise DataSourceError(f"stooq returned no rows for {ticker}")
    return frame.sort_index()


def _load_csv(ticker: str, raw_dir: Path) -> pd.DataFrame:
    candidates = [raw_dir / f"{ticker}.csv", raw_dir / f"{ticker.upper()}.csv", raw_dir / f"{ticker.lower()}.csv"]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise DataSourceError(f"CSV file not found for {ticker} in {raw_dir}")


def _cache_path(cache_dir: Path, ticker: str, provider: str, start: str | None, end: str | None) -> Path:
    safe = ticker.replace("/", "-").replace("^", "")
    start_part = (start or "auto").replace("-", "")
    end_part = (end or "latest").replace("-", "")
    return cache_dir / f"{safe}_{provider}_{start_part}_{end_part}.parquet"


def data_quality_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, frame in frames.items():
        missing = frame[OHLCV_COLUMNS].isna().sum().to_dict() if not frame.empty else {}
        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(frame)),
                "start": frame.index.min().date().isoformat() if not frame.empty else None,
                "end": frame.index.max().date().isoformat() if not frame.empty else None,
                "missing_ohlcv": int(sum(missing.values())) if missing else 0,
                "first_close": float(frame["close"].iloc[0]) if not frame.empty else None,
                "last_close": float(frame["close"].iloc[-1]) if not frame.empty else None,
            }
        )
    return pd.DataFrame(rows)


def parse_tickers(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip().upper() for item in value.split(",") if item.strip()]
    return [str(item).strip().upper() for item in value if str(item).strip()]
