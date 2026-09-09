"""Typed loading of `configs/<case>.yaml` files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from maads.paths import resolve_path
from maads.schema_inference import infer_source_paths, reconcile_data_paths


class DataSource(BaseModel):
    """One raw file in a source bundle (user cases) or an extra document."""

    path: str
    role: str | None = None
    original_filename: str | None = None


class DataPaths(BaseModel):
    train_csv: str | None = None
    test_csv: str | None = None
    sample_submission_csv: str | None = None
    sources: list[DataSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_train_or_sources(self) -> DataPaths:
        if not self.train_csv and not self.sources:
            raise ValueError("data.train_csv or data.sources is required")
        return self


class SuccessCriterion(BaseModel):
    metric: str
    threshold: float
    direction: Literal["minimize", "maximize"] | None = None


class CaseConfig(BaseModel):
    case_id: str
    kaggle_competition: str = ""
    problem_statement: str
    problem_type: str = ""  # e.g. binary_classification | classification | regression
    target_column: str = ""
    id_column: str = ""
    evaluation_metric: str = ""
    data: DataPaths
    feature_hints: dict[str, Any] = Field(default_factory=dict)
    class_labels: dict[str, str] = Field(default_factory=dict)
    success_criterion: SuccessCriterion


def primary_train_csv(data: DataPaths) -> str:
    """Path to the primary labelled (or only) table.

    Demos always set ``train_csv``. User source bundles may list files only
    under ``sources``; prefer a train-named CSV over a test-named one.
    """
    if data.train_csv:
        return data.train_csv
    inferred = infer_source_paths(data.sources)
    if inferred["train"]:
        return inferred["train"]
    for src in data.sources:
        if src.path.lower().endswith(".csv"):
            return src.path
    if data.sources:
        return data.sources[0].path
    raise ValueError("no tabular source path in case data")


def source_locations(data: DataPaths) -> list[str]:
    """Deduplicated list of configured source paths (empties omitted)."""
    seen: list[str] = []
    for src in data.sources:
        if src.path and src.path not in seen:
            seen.append(src.path)
    for path in (data.train_csv, data.test_csv, data.sample_submission_csv):
        if path and path not in seen:
            seen.append(path)
    return seen


def _resolve_optional(path: str | None) -> str | None:
    if not path:
        return None
    return str(resolve_path(path))


def load_case_config(path: Path) -> CaseConfig:
    """Load and validate a case config from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    cfg = CaseConfig.model_validate(raw)
    sources = [
        src.model_copy(update={"path": str(resolve_path(src.path))})
        for src in cfg.data.sources
    ]
    data = cfg.data.model_copy(
        update={
            "train_csv": _resolve_optional(cfg.data.train_csv),
            "test_csv": _resolve_optional(cfg.data.test_csv),
            "sample_submission_csv": _resolve_optional(cfg.data.sample_submission_csv),
            "sources": sources,
        }
    )
    data = reconcile_data_paths(data)
    return cfg.model_copy(update={"data": data})


def kickoff_inputs(config: CaseConfig) -> dict[str, str]:
    """Flat inputs dict for CrewAI ``{placeholder}`` substitution and run artefacts."""
    sc = config.success_criterion
    return {
        "case_id": config.case_id,
        "kaggle_competition": config.kaggle_competition,
        "problem_statement": config.problem_statement,
        "problem_type": config.problem_type,
        "target_column": config.target_column,
        "id_column": config.id_column,
        "evaluation_metric": config.evaluation_metric,
        "success_metric": sc.metric,
        "success_threshold": str(sc.threshold),
    }
