from __future__ import annotations

import numpy as np
import pandas as pd

from stockresearchmarket.strategies.ml_ranker import (
    build_ml_ranker_weights,
    make_price_feature_panel,
    rebalance_dates,
    sector_neutral_selection,
    simulate_portfolio,
)


def _synthetic_close(tickers: int = 12, periods: int = 900) -> pd.DataFrame:
    index = pd.bdate_range("2018-01-01", periods=periods)
    rng = np.random.default_rng(42)
    values = {}
    for idx in range(tickers):
        drift = 0.00015 + idx / 100_000
        shocks = rng.normal(drift, 0.012 + idx / 10_000, len(index))
        values[f"T{idx:02d}"] = 100 * np.cumprod(1 + shocks)
    return pd.DataFrame(values, index=index)


def test_feature_panel_has_known_label_end_after_feature_date() -> None:
    close = _synthetic_close(tickers=4, periods=320)
    dates = rebalance_dates(close.index, "ME")
    panel = make_price_feature_panel(close, dates, horizon_days=21)
    labeled = panel.dropna(subset=["label_end"])
    assert not labeled.empty
    assert (labeled["label_end"] > labeled["date"]).all()


def test_ml_ranker_waits_for_training_history_and_selects_top_n() -> None:
    close = _synthetic_close()
    result = build_ml_ranker_weights(
        close,
        top_n=3,
        train_years=1,
        horizon_days=21,
        min_train_rows=80,
        n_estimators=10,
        min_samples_leaf=3,
        random_state=11,
    )
    tested = result.walk_forward_windows.loc[result.walk_forward_windows["status"].eq("tested")]
    assert not tested.empty
    assert tested["selected_count"].eq(3).all()
    assert result.weights.sum(axis=1).max() <= 1.0 + 1e-12
    assert (result.weights.sum(axis=1) > 0).any()


def test_simulate_portfolio_preserves_cash_weight() -> None:
    close = pd.DataFrame({"AAA": [100, 110, 121]}, index=pd.date_range("2024-01-01", periods=3, freq="B"))
    half_weights = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]}, index=close.index)
    result = simulate_portfolio(close, half_weights, "half", cost_bps=0)
    assert round(result.returns.iloc[1], 4) == 0.05
    assert round(result.equity.iloc[-1], 4) == 1.1025


def test_sector_neutral_selection_sums_to_one_across_active_sectors() -> None:
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0})
    sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health", "E": "Energy"}
    weights = sector_neutral_selection(scores, sectors, top_n=3)
    assert round(weights.sum(), 12) == 1.0
    assert weights.loc["A"] > 0
    assert weights.loc["C"] > 0
    assert weights.loc["E"] > 0
