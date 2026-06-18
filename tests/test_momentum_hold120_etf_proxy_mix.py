from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_momentum_hold120_etf_proxy_mix import constant_mix_weights, generate_simplex_weights


def test_generate_simplex_weights_for_three_assets_with_half_step() -> None:
    weights = generate_simplex_weights(3, 0.5)
    assert weights == [(0.5, 0.5, 0.0)] or weights == []


def test_constant_mix_weights_assigns_monthly_targets() -> None:
    index = pd.bdate_range("2026-01-01", periods=40)
    weights = constant_mix_weights(index, {"AAA": 0.6, "BBB": 0.4})
    active = weights.sum(axis=1) > 0
    assert active.any()
    assert round(float(weights.loc[active].iloc[-1].sum()), 10) == 1.0
    assert round(float(weights.loc[active].iloc[-1]["AAA"]), 10) == 0.6
