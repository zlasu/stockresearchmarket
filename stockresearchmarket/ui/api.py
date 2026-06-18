from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from stockresearchmarket.ui.indexer import ExperimentIndex
from stockresearchmarket.ui.plots import render_plot
from stockresearchmarket.ui.series import compare_variants, load_table, load_variant_series

EXPERIMENTS_ROOT = Path(os.getenv("STOCKRESEARCH_EXPERIMENTS_ROOT", "experiments"))
CACHE_ROOT = Path(os.getenv("STOCKRESEARCH_UI_CACHE", ".cache/stockresearchmarket-ui"))

app = FastAPI(title="StockResearchMarket Research Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = ExperimentIndex(EXPERIMENTS_ROOT)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "experiments": len(index.list()), "root": str(index.root)}


@app.post("/api/index/refresh")
def refresh_index() -> dict[str, object]:
    index.refresh()
    return {"status": "ok", "experiments": len(index.list())}


@app.get("/api/experiments")
def list_experiments(
    query: str = "",
    kind: str = "",
    sort: str = "created_at",
    direction: str = "desc",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
    experiments = index.list()
    if query:
        q = query.lower()
        experiments = [item for item in experiments if q in item.name.lower() or q in item.id.lower() or q in str(item.path).lower()]
    if kind:
        experiments = [item for item in experiments if item.kind == kind]
    reverse = direction.lower() != "asc"
    experiments = sorted(experiments, key=lambda item: _sort_key(item, sort), reverse=reverse)
    total = len(experiments)
    return {"total": total, "items": [index.as_summary(item) for item in experiments[offset : offset + limit]]}


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, object]:
    experiment = _experiment_or_404(experiment_id)
    return index.as_detail(experiment)


@app.get("/api/experiments/{experiment_id}/variants/{variant_id}")
def get_variant(experiment_id: str, variant_id: str) -> dict[str, object]:
    experiment = _experiment_or_404(experiment_id)
    variant = _variant_or_404(experiment, variant_id)
    return index.as_variant(variant, experiment)


@app.get("/api/experiments/{experiment_id}/compare")
def compare_experiment_variants(experiment_id: str, variant_ids: str = "") -> dict[str, object]:
    experiment = _experiment_or_404(experiment_id)
    selected = [item for item in variant_ids.split(",") if item]
    variants = [_variant_or_404(experiment, variant_id) for variant_id in selected] if selected else experiment.variants[:5]
    return compare_variants(experiment, variants)


@app.get("/api/experiments/{experiment_id}/scatter")
def scatter(
    experiment_id: str,
    x: str = "capital",
    y: str = "max_drawdown",
    color: str = "role",
    size: str = "trades",
) -> dict[str, object]:
    experiment = _experiment_or_404(experiment_id)
    rows = []
    for variant in experiment.variants:
        x_value = _metric_or_attr(variant, x)
        y_key = "drawdown_magnitude" if y in {"max_drawdown", "dd", "drawdown"} else y
        y_value = _metric_or_attr(variant, y_key)
        if x_value is None or y_value is None:
            continue
        rows.append(
            {
                "variant_id": variant.id,
                "variant_name": variant.name,
                "x": x_value,
                "y": y_value,
                "color": _metric_or_attr(variant, color),
                "size": _metric_or_attr(variant, size),
                "metrics": variant.metrics,
            }
        )
    return {"x": x, "y": y, "rows": rows}


@app.get("/api/experiments/{experiment_id}/tables/{table}")
def table(
    experiment_id: str,
    table: str,
    variant_id: str = "",
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
    experiment = _experiment_or_404(experiment_id)
    path = None
    if variant_id:
        variant = _variant_or_404(experiment, variant_id)
        path = variant.table_paths.get(table)
    path = path or experiment.table_paths.get(table)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"Table not found: {table}")
    return load_table(path, limit=limit, offset=offset)


@app.get("/api/experiments/{experiment_id}/series/{series}")
def series(experiment_id: str, series: str, variant_id: str = "") -> dict[str, object]:
    experiment = _experiment_or_404(experiment_id)
    variant = _variant_or_404(experiment, variant_id) if variant_id else (experiment.variants[0] if experiment.variants else None)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")
    return {"variant_id": variant.id, "series": series, "rows": load_variant_series(experiment, variant, series)}


@app.get("/api/experiments/{experiment_id}/plots/{plot_type}.png")
def plot(experiment_id: str, plot_type: str) -> FileResponse:
    experiment = _experiment_or_404(experiment_id)
    try:
        path = render_plot(experiment, plot_type, cache_dir=CACHE_ROOT)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="image/png")


def _experiment_or_404(experiment_id: str):
    experiment = index.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return experiment


def _variant_or_404(experiment, variant_id: str):
    for variant in experiment.variants:
        if variant.id == variant_id:
            return variant
    raise HTTPException(status_code=404, detail=f"Variant not found: {variant_id}")


def _sort_key(experiment, sort: str):
    if sort == "variant_count":
        return len(experiment.variants)
    if sort in {"name", "kind", "created_at"}:
        return getattr(experiment, sort) or ""
    best = index.best_variant(experiment)
    if best:
        return best.metrics.get(sort) or best.metrics.get("drawdown_magnitude" if sort == "max_drawdown" else sort) or 0
    return 0


def _metric_or_attr(variant, key: str):
    if key == "role":
        return variant.role or variant.kind
    if key == "rank":
        return variant.rank
    if key == "computed_score":
        return variant.computed_score
    return variant.metrics.get(key)
