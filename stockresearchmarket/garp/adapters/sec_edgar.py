from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
LOGGER = logging.getLogger(__name__)

DURATION_TAGS = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "SalesRevenueNet", "USD"),
    ],
    "net_income": [("us-gaap", "NetIncomeLoss", "USD")],
    "operating_cash_flow": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD")],
    "capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD")],
    "gross_profit": [("us-gaap", "GrossProfit", "USD")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss", "USD")],
    "interest_expense": [
        ("us-gaap", "InterestExpenseNonOperating", "USD"),
        ("us-gaap", "InterestExpense", "USD"),
    ],
    "eps_diluted": [("us-gaap", "EarningsPerShareDiluted", "USD/shares")],
    "diluted_shares": [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")],
    "depreciation_amortization": [
        ("us-gaap", "DepreciationDepletionAndAmortization", "USD"),
        ("us-gaap", "DepreciationDepletionAndAmortizationExpense", "USD"),
    ],
}

INSTANT_TAGS = {
    "assets": [("us-gaap", "Assets", "USD")],
    "liabilities": [("us-gaap", "Liabilities", "USD")],
    "equity": [
        ("us-gaap", "StockholdersEquity", "USD"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
    ],
    "cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "USD"),
    ],
    "debt_current": [("us-gaap", "ShortTermBorrowings", "USD"), ("us-gaap", "LongTermDebtCurrent", "USD")],
    "debt_noncurrent": [("us-gaap", "LongTermDebtNoncurrent", "USD")],
    "current_assets": [("us-gaap", "AssetsCurrent", "USD")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent", "USD")],
    "retained_earnings": [("us-gaap", "RetainedEarningsAccumulatedDeficit", "USD")],
    "shares_outstanding": [("dei", "EntityCommonStockSharesOutstanding", "shares")],
}


@dataclass(frozen=True)
class SecEdgarFundamentalsAdapter:
    cache_dir: Path
    user_agent: str | None = None
    sleep_seconds: float = 0.12
    strict: bool = False

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent
            or os.environ.get("SEC_USER_AGENT")
            or "StockResearchMarket/0.1 research@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }

    def load(self, tickers: list[str], close: pd.DataFrame | None = None, refresh: bool = False) -> pd.DataFrame:
        cik_map = self.load_ticker_cik_map(refresh=refresh)
        frames = []
        for ticker in tickers:
            ticker_label = str(ticker).upper()
            cik = _lookup_cik(cik_map, ticker_label)
            if cik is None:
                continue
            try:
                facts = self.load_companyfacts(cik, refresh=refresh)
                frame = self.companyfacts_to_rows(ticker_label, facts, _close_for_ticker(close, ticker_label))
            except (OSError, ValueError, requests.RequestException) as exc:
                if self.strict:
                    raise
                LOGGER.warning("Skipping SEC fundamentals for %s: %s", ticker_label, exc)
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["as_of_date", "ticker"]).reset_index(drop=True)

    def load_ticker_cik_map(self, refresh: bool = False) -> dict[str, str]:
        path = self.cache_dir / "sec_company_tickers.json"
        payload = _load_or_fetch_json(path, SEC_TICKERS_URL, self.headers, refresh, self.sleep_seconds)
        rows = payload.values() if isinstance(payload, dict) else payload
        mapping: dict[str, str] = {}
        for row in rows:
            ticker = str(row.get("ticker", "")).upper()
            cik = str(row.get("cik_str", "")).zfill(10)
            if ticker and cik:
                for alias in _ticker_aliases(ticker):
                    mapping[alias] = cik
        return mapping

    def load_companyfacts(self, cik: str, refresh: bool = False) -> dict[str, Any]:
        path = self.cache_dir / "companyfacts" / f"CIK{cik}.json"
        url = SEC_COMPANYFACTS_URL.format(cik=cik)
        return _load_or_fetch_json(path, url, self.headers, refresh, self.sleep_seconds)

    def companyfacts_to_rows(self, ticker: str, facts: dict[str, Any], close: pd.Series | None = None) -> pd.DataFrame:
        duration_values = {name: _extract_facts(facts, specs, annual_only=True) for name, specs in DURATION_TAGS.items()}
        instant_values = {name: _extract_facts(facts, specs, annual_only=False) for name, specs in INSTANT_TAGS.items()}
        revenue = duration_values["revenue"]
        if revenue.empty:
            return pd.DataFrame()
        base = revenue[["fy", "period_end", "as_of_date", "accession"]].drop_duplicates().sort_values("as_of_date")
        rows: list[dict[str, Any]] = []
        for _, record in base.iterrows():
            fy = int(record["fy"])
            as_of = pd.Timestamp(record["as_of_date"])
            period_end = pd.Timestamp(record["period_end"])
            values = {name: _latest_value(frame, as_of, fy=fy) for name, frame in duration_values.items()}
            values.update({name: _latest_value(frame, as_of) for name, frame in instant_values.items()})
            market_cap = _market_cap(values, close, as_of)
            enterprise_value = _enterprise_value(market_cap, values)
            revenue_now = values.get("revenue")
            net_income = values.get("net_income")
            ocf = values.get("operating_cash_flow")
            capex = abs(values.get("capex") or np.nan)
            fcf = ocf - capex if pd.notna(ocf) and pd.notna(capex) else np.nan
            ebitda = _safe_add(values.get("operating_income"), values.get("depreciation_amortization"))
            equity = values.get("equity")
            operating_income = values.get("operating_income")
            invested_capital = _safe_add(values.get("equity"), _safe_add(values.get("debt_current"), values.get("debt_noncurrent")))
            rows.append(
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "as_of_date": as_of,
                    "source": "sec_edgar",
                    "fiscal_year": fy,
                    "accession": record["accession"],
                    "market_cap": market_cap,
                    "revenue_growth_3y": _growth(duration_values["revenue"], as_of, fy, 3),
                    "revenue_growth_5y": _growth(duration_values["revenue"], as_of, fy, 5),
                    "eps_growth_3y": _growth(duration_values["eps_diluted"], as_of, fy, 3),
                    "eps_growth_5y": _growth(duration_values["eps_diluted"], as_of, fy, 5),
                    "operating_cash_flow_growth": _growth(duration_values["operating_cash_flow"], as_of, fy, 3),
                    "pe": _safe_div(market_cap, net_income),
                    "forward_pe": np.nan,
                    "ps": _safe_div(market_cap, revenue_now),
                    "ev_ebitda": _safe_div(enterprise_value, ebitda),
                    "fcf_yield": _safe_div(fcf, market_cap),
                    "roe": _safe_div(net_income, equity),
                    "roic": _safe_div(operating_income, invested_capital),
                    "gross_margin": _safe_div(values.get("gross_profit"), revenue_now),
                    "net_margin": _safe_div(net_income, revenue_now),
                    "operating_margin": _safe_div(operating_income, revenue_now),
                    "debt_equity": _safe_div(_safe_add(values.get("debt_current"), values.get("debt_noncurrent")), equity),
                    "interest_coverage": _safe_div(operating_income, abs(values.get("interest_expense") or np.nan)),
                    "positive_fcf": float(pd.notna(fcf) and fcf > 0),
                    "altman_z": _altman_z(values, market_cap, revenue_now, operating_income),
                    "sec_url": _sec_archive_url(facts, record["accession"]),
                }
            )
        return pd.DataFrame(rows)


def _load_or_fetch_json(path: Path, url: str, headers: dict[str, str], refresh: bool, sleep_seconds: float) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    response = None
    for attempt in range(3):
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code not in RETRYABLE_STATUS_CODES:
            break
        time.sleep(max(sleep_seconds, 0.1) * (attempt + 1))
    if response is None:
        raise RuntimeError(f"No response from {url}")
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(sleep_seconds)
    return payload


def _extract_facts(facts: dict[str, Any], specs: list[tuple[str, str, str]], annual_only: bool) -> pd.DataFrame:
    for taxonomy, tag, unit in specs:
        units = facts.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {})
        rows = units.get(unit)
        if not rows and unit == "USD/shares":
            rows = units.get("USD/shares") or units.get("USD/shares")
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        if frame.empty or "val" not in frame.columns:
            continue
        frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
        frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
        if "start" in frame.columns:
            frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
        frame = frame[frame["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A"])]
        if annual_only:
            if "fp" in frame.columns:
                frame = frame[frame["fp"].eq("FY") & frame["form"].isin(["10-K", "10-K/A"])]
            else:
                frame = frame[frame["form"].isin(["10-K", "10-K/A"])]
        if "fy" not in frame.columns:
            frame["fy"] = frame["end"].dt.year
        frame = frame.dropna(subset=["filed", "end", "val", "fy"])
        if frame.empty:
            continue
        frame = frame.rename(columns={"filed": "as_of_date", "end": "period_end", "accn": "accession", "val": "value"})
        frame["tag"] = tag
        return (
            frame[["fy", "period_end", "as_of_date", "accession", "value", "tag"]]
            .sort_values(["fy", "as_of_date", "period_end"])
            .drop_duplicates(["fy", "as_of_date", "accession"], keep="last")
        )
    return pd.DataFrame(columns=["fy", "period_end", "as_of_date", "accession", "value", "tag"])


def _lookup_cik(cik_map: dict[str, str], ticker: str) -> str | None:
    for alias in _ticker_aliases(ticker):
        if alias in cik_map:
            return cik_map[alias]
    return None


def _ticker_aliases(ticker: str) -> list[str]:
    raw = str(ticker).strip().upper()
    normalized = raw.replace(".", "-")
    aliases = [raw, normalized]
    if "-" in normalized:
        aliases.append(normalized.replace("-", "."))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _close_for_ticker(close: pd.DataFrame | None, ticker: str) -> pd.Series | None:
    if close is None:
        return None
    for alias in _ticker_aliases(ticker):
        if alias in close.columns:
            return close[alias]
    return None


def _latest_value(frame: pd.DataFrame, as_of: pd.Timestamp, fy: int | None = None) -> float:
    if frame.empty:
        return np.nan
    available = frame[pd.to_datetime(frame["as_of_date"]) <= as_of]
    if fy is not None:
        available = available[available["fy"].astype(int) == int(fy)]
    if available.empty:
        return np.nan
    return float(available.sort_values("as_of_date")["value"].iloc[-1])


def _growth(frame: pd.DataFrame, as_of: pd.Timestamp, fy: int, years: int) -> float:
    if frame.empty:
        return np.nan
    if not _has_continuous_scale(frame, as_of, fy, years):
        return np.nan
    current = _latest_value(frame, as_of, fy)
    prior = _latest_value(frame, as_of, fy - years)
    if pd.isna(current) or pd.isna(prior) or prior == 0:
        return np.nan
    return float((abs(current) / abs(prior)) ** (1 / years) - 1)


def _has_continuous_scale(frame: pd.DataFrame, as_of: pd.Timestamp, fy: int, years: int) -> bool:
    available = frame[pd.to_datetime(frame["as_of_date"]) <= as_of].copy()
    available = available[available["fy"].astype(int).between(int(fy - years), int(fy))]
    if available.empty:
        return False
    yearly = available.sort_values("as_of_date").drop_duplicates("fy", keep="last").sort_values("fy")
    if len(yearly) < years + 1:
        return False
    values = yearly["value"].abs().replace(0, np.nan)
    ratios = values.div(values.shift(1)).dropna()
    return bool(ratios.between(0.25, 4.0).all())


def _market_cap(values: dict[str, float], close: pd.Series | None, as_of: pd.Timestamp) -> float:
    shares = values.get("shares_outstanding")
    if pd.isna(shares) or shares <= 0:
        shares = values.get("diluted_shares")
    if close is None or pd.isna(shares) or shares <= 0:
        return np.nan
    history = close.loc[:as_of].dropna()
    if history.empty:
        return np.nan
    return float(history.iloc[-1] * shares)


def _enterprise_value(market_cap: float, values: dict[str, float]) -> float:
    if pd.isna(market_cap):
        return np.nan
    debt = _safe_add(values.get("debt_current"), values.get("debt_noncurrent"))
    cash = values.get("cash")
    return float(market_cap + (0 if pd.isna(debt) else debt) - (0 if pd.isna(cash) else cash))


def _altman_z(values: dict[str, float], market_cap: float, revenue: float, operating_income: float) -> float:
    assets = values.get("assets")
    liabilities = values.get("liabilities")
    if pd.isna(assets) or assets == 0:
        return np.nan
    working_capital = _safe_add(values.get("current_assets"), -values.get("current_liabilities") if pd.notna(values.get("current_liabilities")) else np.nan)
    retained = values.get("retained_earnings")
    if pd.isna(liabilities) or liabilities == 0:
        market_liability = np.nan
    else:
        market_liability = market_cap / liabilities
    parts = [
        1.2 * _safe_div(working_capital, assets),
        1.4 * _safe_div(retained, assets),
        3.3 * _safe_div(operating_income, assets),
        0.6 * market_liability,
        _safe_div(revenue, assets),
    ]
    valid = [part for part in parts if pd.notna(part)]
    return float(sum(valid)) if len(valid) >= 3 else np.nan


def _safe_div(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def _safe_add(left: float | None, right: float | None) -> float:
    if pd.isna(left) and pd.isna(right):
        return np.nan
    return float((0 if pd.isna(left) else left) + (0 if pd.isna(right) else right))


def _sec_archive_url(facts: dict[str, Any], accession: str) -> str:
    cik = str(facts.get("cik", "")).lstrip("0")
    accession_clean = str(accession).replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/"
