from __future__ import annotations

import pandas as pd
import requests

from stockresearchmarket.garp.adapters import sec_edgar
from stockresearchmarket.garp.adapters.sec_edgar import SecEdgarFundamentalsAdapter


def test_sec_companyfacts_to_rows_uses_filed_date_as_as_of(tmp_path) -> None:
    facts = {
        "cik": "0000000001",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _duration(2020, "2021-02-15", 100),
                            _duration(2021, "2022-02-15", 125),
                            _duration(2022, "2023-02-15", 160),
                            _duration(2023, "2024-02-15", 200),
                        ]
                    }
                },
                "NetIncomeLoss": {"units": {"USD": [_duration(2020, "2021-02-15", 10), _duration(2023, "2024-02-15", 20)]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_duration(2020, "2021-02-15", 12), _duration(2023, "2024-02-15", 24)]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_duration(2023, "2024-02-15", 4)]}
                },
                "OperatingIncomeLoss": {"units": {"USD": [_duration(2023, "2024-02-15", 30)]}},
                "GrossProfit": {"units": {"USD": [_duration(2023, "2024-02-15", 80)]}},
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {"shares": [_duration(2023, "2024-02-15", 10)]}
                },
                "StockholdersEquity": {"units": {"USD": [_instant(2023, "2024-02-15", 100)]}},
                "Assets": {"units": {"USD": [_instant(2023, "2024-02-15", 300)]}},
                "Liabilities": {"units": {"USD": [_instant(2023, "2024-02-15", 150)]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {"shares": [_instant(2023, "2024-02-15", 10)]}},
            },
        },
    }
    close = pd.Series([9.0, 10.0, 11.0], index=pd.to_datetime(["2024-02-13", "2024-02-15", "2024-02-16"]))
    adapter = SecEdgarFundamentalsAdapter(cache_dir=tmp_path)
    rows = adapter.companyfacts_to_rows("AAA", facts, close)
    latest = rows.sort_values("as_of_date").iloc[-1]
    assert latest["as_of_date"] == pd.Timestamp("2024-02-15")
    assert latest["market_cap"] == 100
    assert round(latest["revenue_growth_3y"], 4) == 0.2599
    assert latest["source"] == "sec_edgar"


def test_sec_ticker_cik_map_includes_dot_and_dash_aliases(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sec_edgar,
        "_load_or_fetch_json",
        lambda *args, **kwargs: {"0": {"ticker": "BRK-B", "cik_str": 1067983}},
    )

    adapter = SecEdgarFundamentalsAdapter(cache_dir=tmp_path)
    mapping = adapter.load_ticker_cik_map()

    assert mapping["BRK-B"] == "0001067983"
    assert mapping["BRK.B"] == "0001067983"


def test_sec_load_uses_close_alias_and_keeps_requested_ticker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(SecEdgarFundamentalsAdapter, "load_ticker_cik_map", lambda self, refresh=False: {"BRK-B": "0001067983"})
    monkeypatch.setattr(SecEdgarFundamentalsAdapter, "load_companyfacts", lambda self, cik, refresh=False: _facts_fixture())
    close = pd.DataFrame(
        {"BRK-B": [99.0, 100.0]},
        index=pd.to_datetime(["2024-02-14", "2024-02-15"]),
    )

    adapter = SecEdgarFundamentalsAdapter(cache_dir=tmp_path)
    rows = adapter.load(["BRK.B"], close=close)
    latest = rows.sort_values("as_of_date").iloc[-1]

    assert latest["ticker"] == "BRK.B"
    assert latest["market_cap"] == 1000


def test_sec_load_skips_failed_ticker_when_not_strict(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        SecEdgarFundamentalsAdapter,
        "load_ticker_cik_map",
        lambda self, refresh=False: {"GOOD": "0000000001", "BAD": "0000000002"},
    )

    def load_companyfacts(self, cik, refresh=False):
        if cik == "0000000002":
            raise requests.HTTPError("SEC unavailable")
        return _facts_fixture()

    monkeypatch.setattr(SecEdgarFundamentalsAdapter, "load_companyfacts", load_companyfacts)

    adapter = SecEdgarFundamentalsAdapter(cache_dir=tmp_path)
    rows = adapter.load(["GOOD", "BAD"])

    assert set(rows["ticker"]) == {"GOOD"}


def _facts_fixture() -> dict[str, object]:
    return {
        "cik": "0000000001",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _duration(2020, "2021-02-15", 100),
                            _duration(2021, "2022-02-15", 125),
                            _duration(2022, "2023-02-15", 160),
                            _duration(2023, "2024-02-15", 200),
                        ]
                    }
                },
                "NetIncomeLoss": {"units": {"USD": [_duration(2023, "2024-02-15", 20)]}},
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {"shares": [_duration(2023, "2024-02-15", 10)]}
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {"shares": [_instant(2023, "2024-02-15", 10)]}},
            },
        },
    }


def _duration(fy: int, filed: str, value: float) -> dict[str, object]:
    return {
        "fy": fy,
        "fp": "FY",
        "form": "10-K",
        "filed": filed,
        "start": f"{fy}-01-01",
        "end": f"{fy}-12-31",
        "accn": f"0000000001-{fy}",
        "val": value,
    }


def _instant(fy: int, filed: str, value: float) -> dict[str, object]:
    return {
        "fy": fy,
        "fp": "FY",
        "form": "10-K",
        "filed": filed,
        "end": f"{fy}-12-31",
        "accn": f"0000000001-{fy}",
        "val": value,
    }
