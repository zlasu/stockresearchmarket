from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from stockresearchmarket.ui.api import app
from stockresearchmarket.ui.indexer import ExperimentIndex


def test_indexer_detects_optimizer_and_normalizes_metrics(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    experiment = root / "2026-06-16_101010_sma_cross_SPY_optimize"
    best = experiment / "best"
    best.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "strategy": "sma_cross",
                "ticker": "SPY",
                "total_return": 0.42,
                "cagr": 0.12,
                "sharpe": 1.2,
                "max_drawdown": -0.18,
                "calmar": 0.67,
                "trades": 9,
                "param_fast_window": 20,
            }
        ]
    ).to_csv(experiment / "optimizer_results.csv", index=False)
    pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "strategy": [10000, 11420]}).to_csv(
        best / "equity_curve.csv",
        index=False,
    )

    index = ExperimentIndex(root)

    records = index.list()
    assert len(records) == 1
    variant = records[0].variants[0]
    assert variant.metrics["drawdown_magnitude"] == 0.18
    assert variant.metrics["capital"] == 11420
    assert variant.params["fast_window"] == 20


def test_indexer_detects_quickmoney_variant_json(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    experiment = root / "quickmoney_slippage"
    variant_dir = experiment / "variant_a"
    variant_dir.mkdir(parents=True)
    equity = variant_dir / "equity_curve.csv"
    trades = variant_dir / "trades.csv"
    pd.DataFrame({"timestamp": ["2026-01-01T14:30:00+00:00"], "equity": [121000]}).to_csv(equity, index=False)
    pd.DataFrame({"symbol": ["SPY"], "pnl": [100]}).to_csv(trades, index=False)
    (experiment / "summary.json").write_text(
        json.dumps(
            {
                "source": "quickmoney fixture",
                "variants": {
                    "variant_a": {
                        "variant": {"entry_mode": "immediate"},
                        "result": {
                            "description": "Variant A",
                            "return_pct": 21.0,
                            "max_drawdown_pct": 4.5,
                            "daily_sharpe": 2.1,
                            "closed_trades": 4,
                            "equity_csv": str(equity),
                            "trade_csv": str(trades),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    index = ExperimentIndex(root)
    variant = index.list()[0].variants[0]
    assert variant.metrics["total_return"] == 0.21
    assert variant.metrics["max_drawdown"] == -0.045
    assert variant.table_paths["trades"] == trades.resolve()


def test_api_endpoints_with_temp_index(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "experiments"
    experiment = root / "2026-06-16_101010_smoke"
    ticker_dir = experiment / "SPY"
    ticker_dir.mkdir(parents=True)
    pd.DataFrame([{"ticker": "SPY", "cagr": 0.1, "sharpe": 1.0, "max_drawdown": -0.2}]).to_csv(
        experiment / "summary.csv",
        index=False,
    )
    pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "strategy": [1.0, 1.1]}).to_csv(
        ticker_dir / "equity_curve.csv",
        index=False,
    )

    import stockresearchmarket.ui.api as api_module

    monkeypatch.setattr(api_module, "index", ExperimentIndex(root))
    client = TestClient(app)

    listed = client.get("/api/experiments").json()
    experiment_id = listed["items"][0]["id"]
    assert listed["total"] == 1
    assert client.get(f"/api/experiments/{experiment_id}").json()["variant_count"] == 1
    assert client.get(f"/api/experiments/{experiment_id}/scatter").json()["rows"][0]["y"] == 0.2
    assert client.get(f"/api/experiments/{experiment_id}/compare?variant_ids=SPY").json()["series"]["equity"]["SPY"]


def test_plot_endpoint_handles_missing_return_and_drawdown_metrics(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "experiments"
    experiment = root / "2026-06-16_101010_sparse"
    experiment.mkdir(parents=True)
    pd.DataFrame([{"ticker": "SPY", "note": "metadata only"}]).to_csv(experiment / "summary.csv", index=False)

    import stockresearchmarket.ui.api as api_module

    monkeypatch.setattr(api_module, "index", ExperimentIndex(root))
    monkeypatch.setattr(api_module, "CACHE_ROOT", tmp_path / "cache")
    client = TestClient(app)

    experiment_id = client.get("/api/experiments").json()["items"][0]["id"]
    response = client.get(f"/api/experiments/{experiment_id}/plots/risk-return.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
