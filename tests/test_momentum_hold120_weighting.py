from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_momentum_hold120_weighting_sweep import SelectionSnapshot, cap_weights, scheme_weights


def test_cap_weights_respects_cap_and_sums_to_one() -> None:
    weights = pd.Series({"A": 0.60, "B": 0.25, "C": 0.15}, dtype="float64")
    capped = cap_weights(weights, 0.40)
    assert abs(float(capped.sum()) - 1.0) < 1e-9
    assert float(capped.max()) <= 0.40 + 1e-9


def test_scheme_weights_inverse_vol_capped_uses_cap() -> None:
    index = pd.date_range("2026-01-01", periods=80, freq="B")
    close = pd.DataFrame(
        {
            "A": 100 * (1.001 ** pd.RangeIndex(len(index))),
            "B": 100 * (1.01 ** (pd.RangeIndex(len(index)) / 10)),
            "C": 100 * (1.002 ** pd.RangeIndex(len(index))),
        },
        index=index,
    )
    score = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}, dtype="float64")
    ranks = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}, dtype="float64")
    snapshot = SelectionSnapshot(index[-1], ["A", "B", "C"], score, ranks)

    weights = scheme_weights(
        close,
        snapshot,
        scheme="momentum_hold120_inverse_vol_cap6",
        cap_weight=0.40,
        vol_lookback=20,
    )

    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert float(weights.max()) <= 0.40 + 1e-9
