from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def strategy_defaults(config: dict[str, Any], strategy: str) -> dict[str, Any]:
    return deepcopy(get_nested(config, f"strategies.{strategy}.defaults", {}))


def strategy_grid(config: dict[str, Any], strategy: str) -> dict[str, list[Any]]:
    return deepcopy(get_nested(config, f"strategies.{strategy}.grid", {}))


def universe_from_config(config: dict[str, Any], names: str | None = None) -> list[str]:
    universe = config.get("universe", {})
    if not names:
        names = "core"
    tickers: list[str] = []
    for name in [item.strip() for item in names.split(",") if item.strip()]:
        values = universe.get(name)
        if values is None:
            tickers.append(name.upper())
        else:
            tickers.extend(str(ticker).upper() for ticker in values)
    return list(dict.fromkeys(tickers))

