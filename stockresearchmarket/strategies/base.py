from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

SignalFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    signal: Callable[..., pd.Series]
    portfolio: bool = False

    def generate(self, data: pd.DataFrame, **params: Any) -> pd.Series:
        return self.signal(data, **params).rename("signal")

