from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stockresearchmarket.ui.models import ExperimentRecord, VariantRecord
from stockresearchmarket.ui.normalization import (
    experiment_id_for,
    first_numeric_last,
    infer_created_at,
    normalize_metrics,
    params_from_row,
    stable_id,
)
from stockresearchmarket.ui.recommendations import build_recommendations, score_variants
from stockresearchmarket.ui.serialization import clean_value, public_path


class ExperimentIndex:
    def __init__(self, root: Path = Path("experiments")) -> None:
        self.root = root.resolve()
        self._experiments: dict[str, ExperimentRecord] = {}
        self.refresh()

    def refresh(self) -> None:
        self._experiments = {}
        if not self.root.exists():
            return
        marker_names = {
            "bot_candidate_summary.csv",
            "leaderboard.csv",
            "optimizer_results.csv",
            "summary.csv",
            "summary.json",
        }
        marker_dirs = sorted({path.parent for path in self.root.rglob("*") if path.name in marker_names})
        for directory in marker_dirs:
            try:
                experiment = self._parse_directory(directory)
            except Exception:
                continue
            if experiment and experiment.variants:
                self._finalize(experiment)
                self._experiments[experiment.id] = experiment

    def list(self) -> list[ExperimentRecord]:
        return sorted(self._experiments.values(), key=lambda item: item.created_at or item.name, reverse=True)

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._experiments.get(experiment_id)

    def as_summary(self, experiment: ExperimentRecord) -> dict[str, Any]:
        best = self.best_variant(experiment)
        return clean_value(
            {
                "id": experiment.id,
                "name": experiment.name,
                "kind": experiment.kind,
                "created_at": experiment.created_at,
                "variant_count": len(experiment.variants),
                "best_variant_id": best.id if best else None,
                "best_variant_name": best.name if best else None,
                "metrics": best.metrics if best else experiment.metrics,
                "path": public_path(experiment.path, self.root),
                "caveat_count": len(experiment.caveats),
            }
        )

    def as_detail(self, experiment: ExperimentRecord) -> dict[str, Any]:
        return clean_value(
            {
                **self.as_summary(experiment),
                "metadata": experiment.metadata,
                "caveats": experiment.caveats,
                "recommendations": experiment.recommendations,
                "available_tables": sorted(set(experiment.table_paths) | {key for v in experiment.variants for key in v.table_paths}),
                "available_series": sorted(set(experiment.series_paths) | {key for v in experiment.variants for key in v.series_paths}),
                "artifacts": experiment.artifacts,
                "variants": [self.as_variant(variant, experiment) for variant in experiment.variants],
            }
        )

    def as_variant(self, variant: VariantRecord, experiment: ExperimentRecord | None = None) -> dict[str, Any]:
        root = self.root if experiment is None else self.root
        return clean_value(
            {
                "id": variant.id,
                "name": variant.name,
                "kind": variant.kind,
                "role": variant.role,
                "rank": variant.rank,
                "source": variant.source,
                "metrics": variant.metrics,
                "original_metrics": variant.original_metrics,
                "params": variant.params,
                "computed_score": variant.computed_score,
                "path": public_path(variant.path, root),
                "available_tables": sorted(variant.table_paths),
                "available_series": sorted(variant.series_paths),
                "series_columns": variant.series_columns,
                "artifacts": variant.artifacts,
                "start": variant.start,
                "end": variant.end,
            }
        )

    def best_variant(self, experiment: ExperimentRecord) -> VariantRecord | None:
        if not experiment.variants:
            return None
        with_score = [variant for variant in experiment.variants if variant.computed_score is not None]
        if with_score:
            return max(with_score, key=lambda variant: float(variant.computed_score or 0.0))
        return experiment.variants[0]

    def _parse_directory(self, directory: Path) -> ExperimentRecord | None:
        if (directory / "bot_candidate_summary.csv").exists():
            return self._parse_bot_pack(directory)
        if (directory / "leaderboard.csv").exists():
            return self._parse_leaderboard(directory)
        if (directory / "optimizer_results.csv").exists():
            return self._parse_optimizer(directory)
        if (directory / "summary.csv").exists():
            return self._parse_summary_csv(directory)
        if (directory / "summary.json").exists():
            return self._parse_summary_json(directory)
        return None

    def _base_experiment(self, directory: Path, kind: str, name: str | None = None) -> ExperimentRecord:
        return ExperimentRecord(
            id=experiment_id_for(directory, self.root),
            name=name or directory.name,
            path=directory,
            kind=kind,
            created_at=infer_created_at(directory),
            metadata=_read_json(directory / "run_metadata.json") or {},
            caveats=_caveats(directory),
            artifacts=_artifacts(directory, self.root),
        )

    def _parse_bot_pack(self, directory: Path) -> ExperimentRecord:
        experiment = self._base_experiment(directory, "chart_pack", "Bot candidate chart pack")
        frame = pd.read_csv(directory / "bot_candidate_summary.csv")
        for idx, row in frame.iterrows():
            payload = row.to_dict()
            name = str(payload.get("strategy") or f"variant_{idx + 1}")
            variant = VariantRecord(
                id=stable_id(name),
                name=name,
                path=directory,
                kind="chart_pack_variant",
                metrics=normalize_metrics(payload, equity_last=first_numeric_last(directory / "equity_curves.csv", name)),
                original_metrics=payload,
                params=params_from_row(payload),
                role=str(payload.get("bot_role")) if pd.notna(payload.get("bot_role")) else None,
                rank=idx + 1,
                series_paths=_series_paths(directory),
                series_columns={kind: name for kind in ["equity", "drawdown", "turnover", "monthly_returns", "yearly_returns"]},
            )
            experiment.variants.append(variant)
        experiment.series_paths = _series_paths(directory)
        experiment.table_paths = _table_paths(directory)
        return experiment

    def _parse_leaderboard(self, directory: Path) -> ExperimentRecord:
        experiment = self._base_experiment(directory, "leaderboard")
        frame = pd.read_csv(directory / "leaderboard.csv")
        for idx, row in frame.iterrows():
            payload = row.to_dict()
            name = str(payload.get("name") or payload.get("experiment_id") or payload.get("variant_id") or f"rank_{idx + 1}")
            variant_dir = _path_from_payload(payload.get("output_dir"), directory)
            variant = VariantRecord(
                id=stable_id(str(payload.get("variant_id") or payload.get("experiment_id") or name)),
                name=name,
                path=variant_dir or directory,
                kind="leaderboard_variant",
                metrics=normalize_metrics(payload, equity_last=first_numeric_last((variant_dir or directory) / "equity_curve.csv")),
                original_metrics=payload,
                params=params_from_row(payload),
                rank=idx + 1,
                source=str(payload.get("config_path")) if payload.get("config_path") else None,
                table_paths=_table_paths(variant_dir or directory),
                series_paths=_series_paths(variant_dir or directory),
            )
            experiment.variants.append(variant)
        experiment.table_paths = _table_paths(directory)
        return experiment

    def _parse_optimizer(self, directory: Path) -> ExperimentRecord:
        experiment = self._base_experiment(directory, "optimization")
        frame = pd.read_csv(directory / "optimizer_results.csv")
        for idx, row in frame.iterrows():
            payload = row.to_dict()
            rank = int(payload.get("rank") or idx + 1)
            variant_dir = directory / "best" if rank == 1 and (directory / "best").exists() else directory
            name = f"rank {rank}"
            params = params_from_row(payload)
            if params:
                name = f"rank {rank} | " + ", ".join(f"{k}={v}" for k, v in params.items())
            variant = VariantRecord(
                id=stable_id(f"rank-{rank}"),
                name=name,
                path=variant_dir,
                kind="optimizer_candidate",
                metrics=normalize_metrics(payload, equity_last=first_numeric_last(variant_dir / "equity_curve.csv")),
                original_metrics=payload,
                params=params,
                rank=rank,
                table_paths=_table_paths(variant_dir),
                series_paths=_series_paths(variant_dir),
            )
            experiment.variants.append(variant)
        experiment.table_paths = {"optimizer_results": directory / "optimizer_results.csv", **_table_paths(directory)}
        return experiment

    def _parse_summary_csv(self, directory: Path) -> ExperimentRecord:
        experiment = self._base_experiment(directory, "multi_run")
        frame = pd.read_csv(directory / "summary.csv")
        for idx, row in frame.iterrows():
            payload = row.to_dict()
            name = str(payload.get("ticker") or payload.get("strategy") or f"variant_{idx + 1}")
            variant_dir = directory / name if (directory / name).exists() else directory
            variant = VariantRecord(
                id=stable_id(name),
                name=name,
                path=variant_dir,
                kind="summary_row",
                metrics=normalize_metrics(payload, equity_last=first_numeric_last(variant_dir / "equity_curve.csv")),
                original_metrics=payload,
                params=params_from_row(payload),
                rank=idx + 1,
                table_paths=_table_paths(variant_dir),
                series_paths=_series_paths(variant_dir),
                artifacts=_artifacts(variant_dir, self.root),
            )
            experiment.variants.append(variant)
        experiment.table_paths = _table_paths(directory)
        return experiment

    def _parse_summary_json(self, directory: Path) -> ExperimentRecord:
        payload = _read_json(directory / "summary.json") or {}
        if isinstance(payload.get("variants"), dict):
            return self._parse_variant_json(directory, payload)
        name = _name_from_config(directory) or str(payload.get("strategy") or payload.get("source") or directory.name)
        kind = "garp_run" if (directory / "resolved_config.yaml").exists() else "single_run"
        experiment = self._base_experiment(directory, kind, name=name)
        metrics = normalize_metrics(payload, equity_last=first_numeric_last(directory / "equity_curve.csv"))
        variant = VariantRecord(
            id="main",
            name=name,
            path=directory,
            kind=kind,
            metrics=metrics,
            original_metrics=payload,
            params=payload.get("params", {}) if isinstance(payload.get("params"), dict) else {},
            table_paths=_table_paths(directory),
            series_paths=_series_paths(directory),
            artifacts=_artifacts(directory, self.root),
            start=payload.get("start"),
            end=payload.get("end"),
        )
        experiment.variants.append(variant)
        experiment.metrics = metrics
        experiment.table_paths = _table_paths(directory)
        experiment.series_paths = _series_paths(directory)
        return experiment

    def _parse_variant_json(self, directory: Path, payload: dict[str, Any]) -> ExperimentRecord:
        experiment = self._base_experiment(directory, "variant_json", name=str(payload.get("source") or directory.name))
        metadata = {key: value for key, value in payload.items() if key != "variants"}
        experiment.metadata.update(metadata)
        if isinstance(payload.get("data_quality"), dict):
            experiment.caveats.append(
                "Data quality is experiment-specific; inspect the source rows before promoting a variant."
            )
        for idx, (variant_id, variant_payload) in enumerate(payload.get("variants", {}).items(), start=1):
            result = variant_payload.get("result", {}) if isinstance(variant_payload, dict) else {}
            raw_variant = variant_payload.get("variant", {}) if isinstance(variant_payload, dict) else {}
            variant_dir = _path_from_payload(result.get("equity_csv"), directory)
            if variant_dir is not None:
                variant_dir = variant_dir.parent
            else:
                variant_dir = directory / str(variant_id)
            equity_path = _path_from_payload(result.get("equity_csv"), directory)
            trades_path = _path_from_payload(result.get("trade_csv"), directory)
            series_paths = _series_paths(variant_dir)
            if equity_path is not None:
                series_paths["equity"] = equity_path
            table_paths = _table_paths(variant_dir)
            if trades_path is not None:
                table_paths["trades"] = trades_path
            variant = VariantRecord(
                id=stable_id(str(variant_id)),
                name=str(result.get("description") or variant_id),
                path=variant_dir,
                kind="json_variant",
                metrics=normalize_metrics(result, equity_last=first_numeric_last(equity_path or variant_dir / "equity_curve.csv", "equity")),
                original_metrics=result,
                params=raw_variant if isinstance(raw_variant, dict) else {},
                role=str(result.get("entry_slug")) if result.get("entry_slug") else None,
                rank=idx,
                table_paths=table_paths,
                series_paths=series_paths,
            )
            experiment.variants.append(variant)
        return experiment

    def _finalize(self, experiment: ExperimentRecord) -> None:
        score_variants(experiment.variants)
        experiment.recommendations = build_recommendations(experiment.variants)
        best = self.best_variant(experiment)
        experiment.best_variant_id = best.id if best else None
        if best:
            experiment.metrics = best.metrics


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _name_from_config(directory: Path) -> str | None:
    config = _read_yaml(directory / "resolved_config.yaml")
    if not config:
        return None
    experiment = config.get("experiment")
    if isinstance(experiment, dict):
        return experiment.get("name") or experiment.get("id")
    return None


def _path_from_payload(value: Any, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.exists():
        return path.resolve()
    candidate = base / Path(str(value)).name
    return candidate.resolve() if candidate.exists() else None


def _table_paths(directory: Path) -> dict[str, Path]:
    candidates = {
        "trades": directory / "trades.csv",
        "weights": directory / "weights.csv",
        "position": directory / "position.csv",
        "holdings": directory / "holdings_history.csv",
        "scores": directory / "scores.csv",
        "data_quality": directory / "data_quality.csv",
        "data_availability": directory / "data_availability.csv",
        "factor_values": directory / "factor_values.csv",
        "walk_forward": directory / "walk_forward.csv",
        "summary": directory / "summary.csv",
    }
    return {key: path for key, path in candidates.items() if path.exists()}


def _series_paths(directory: Path) -> dict[str, Path]:
    candidates = {
        "equity": directory / "equity_curve.csv",
        "drawdown": directory / "drawdowns.csv",
        "turnover": directory / "turnover.csv",
        "monthly_returns": directory / "monthly_returns.csv",
        "yearly_returns": directory / "yearly_returns.csv",
    }
    if (directory / "equity_curves.csv").exists():
        candidates["equity"] = directory / "equity_curves.csv"
    return {key: path for key, path in candidates.items() if path.exists()}


def _artifacts(directory: Path, root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in [directory / "report.html", directory / "optimizer.html", directory / "report.md", directory / "summary.md"]:
        if path.exists():
            artifacts[path.name] = public_path(path, root)
    return artifacts


def _caveats(directory: Path) -> list[str]:
    caveats: list[str] = []
    metadata = _read_json(directory / "run_metadata.json") or {}
    warning = metadata.get("survivorship_bias_warning")
    if warning:
        caveats.append(str(warning))
    if (directory / "data_availability.csv").exists():
        caveats.append("Data availability file is present; inspect coverage before trusting rankings.")
    if "quickmoney" in directory.name:
        caveats.append("Intraday quickmoney experiment; compare slippage and execution assumptions carefully.")
    return caveats

