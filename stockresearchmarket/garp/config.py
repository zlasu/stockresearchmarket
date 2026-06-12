from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_GARP_CONFIG = Path("configs/garp_default.yaml")
EXPERIMENT_DIR = Path("configs/garp_experiments")


def load_garp_config(experiment: str | Path | None = None, base_path: str | Path = DEFAULT_GARP_CONFIG) -> dict[str, Any]:
    with Path(base_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if experiment:
        experiment_path = resolve_experiment_path(experiment)
        with experiment_path.open("r", encoding="utf-8") as handle:
            experiment_config = yaml.safe_load(handle) or {}
        config["experiment"] = {key: value for key, value in experiment_config.items() if key != "overrides"}
        config = deep_merge(config, experiment_config.get("overrides", {}))
    else:
        config["experiment"] = {"id": "adhoc_garp", "name": "Ad hoc GARP run"}
    return config


def resolve_experiment_path(experiment: str | Path) -> Path:
    path = Path(experiment)
    if path.exists():
        return path
    if path.suffix:
        candidate = EXPERIMENT_DIR / path.name
        if candidate.exists():
            return candidate
    matches = sorted(EXPERIMENT_DIR.glob(f"{path}*.yaml"))
    if matches:
        return matches[0]
    candidate = EXPERIMENT_DIR / f"{path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"GARP experiment config not found: {experiment}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_config(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor

