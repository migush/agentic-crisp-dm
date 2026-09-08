"""State-artifact validators run before CRISP-DM phase transitions.

These bridge "what the state claims" and "what the artifacts on disk actually
are". A non-empty result is a real, measured deficit — the orchestrator turns it
into a Loop B signal rather than advancing on a lie.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from maads.text_normalize import normalize_inclusion_rationale

if TYPE_CHECKING:
    from maads.state import CrispDMState

_CV_SCORE_CEILINGS: dict[str, float] = {
    "accuracy": 0.999,
    "auc": 0.999,
    "roc_auc": 0.999,
}


def validate_phase_3_artifacts(state: "CrispDMState") -> list[str]:
    """Check the prepared parquet matches the Data Preparation claims.

    Returns a list of human-readable deficits (empty == clean). Reads the parquet
    written by the Data Engineer's authored prep code at 3.5.
    """
    errors: list[str] = []
    target = state.resolved_target()
    id_col = state.config.id_column

    merged = state.dp.merged_data or {}
    cols_train = merged.get("columns_train")
    if isinstance(cols_train, list) and cols_train:
        predictors = _predictor_columns(cols_train, target, id_col)
        if not predictors:
            errors.append("merged_data.columns_train has no predictor columns")

    train_path = state.dp.dataset.get("train")
    test_path = state.dp.dataset.get("test")
    if not train_path:
        return errors + ["dp.dataset['train'] is missing"]
    if not test_path:
        errors.append("dp.dataset['test'] is missing")

    p = Path(train_path)
    if not p.exists():
        return errors + [f"train parquet does not exist: {train_path}"]
    if p.suffix != ".parquet":
        errors.append(f"train dataset is not parquet: {train_path}")

    try:
        import pandas as pd

        df = pd.read_parquet(train_path)
    except Exception as exc:  # unreadable parquet is itself a deficit
        return errors + [f"could not read train parquet: {exc}"]

    if not target:
        errors.append("target_column is unset and Data Understanding did not record a target")
    elif target not in df.columns:
        errors.append(f"target '{target}' not in prepared train parquet")
    elif bool(df[target].isna().any()):
        errors.append(f"target '{target}' has missing values after prep")
    else:
        state.adopt_target_if_blank(target)

    feature_cols = _predictor_columns(list(df.columns), target, id_col)
    if not feature_cols:
        errors.append("prepared train parquet has no feature columns")

    rationale = normalize_inclusion_rationale(
        state.dp.rationale_for_inclusion_exclusion,
    )
    included = rationale.get("included_columns")
    if isinstance(included, list):
        expected = _predictor_columns(included, target, id_col)
        missing = [c for c in expected if c not in df.columns]
        if missing:
            shown = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            errors.append(
                f"rationale included columns missing from parquet: {shown}{suffix}"
            )

    # Claimed derived features must actually be present as columns.
    derived = state.dp.derived_attributes or {}
    claimed = derived.get("items") if isinstance(derived, dict) else None
    for feat in _feature_names(claimed):
        if feat not in df.columns:
            errors.append(f"claimed derived feature '{feat}' absent from parquet")

    return errors


def validate_phase_4_models(state: "CrispDMState") -> list[str]:
    """Check the modeling state is internally consistent."""
    errors: list[str] = []
    if not state.md.models:
        return ["no models were produced in Phase 4"]
    metric = (state.config.evaluation_metric or "").lower()
    ceiling = _CV_SCORE_CEILINGS.get(metric)
    for i, run in enumerate(state.md.models):
        if run.cv_score is None:
            errors.append(f"model {i} ('{run.technique}') has no cv_score")
        elif ceiling is not None and run.cv_score >= ceiling:
            errors.append(
                f"model {i} ('{run.technique}') cv_score {run.cv_score} "
                f"exceeds sanity ceiling for {metric}"
            )
    if state.md.chosen_model is None:
        errors.append("models exist but chosen_model is not set")
    else:
        chosen = state.md.chosen_model
        if chosen.cv_score is None:
            errors.append("chosen_model has no cv_score")
        elif ceiling is not None and chosen.cv_score >= ceiling:
            errors.append(
                f"chosen_model cv_score {chosen.cv_score} exceeds sanity ceiling "
                f"for {metric}"
            )
    return errors


def _predictor_columns(
    columns: list[str],
    target: str,
    id_col: str | None,
) -> list[str]:
    reserved = {target}
    if id_col:
        reserved.add(id_col)
    return [c for c in columns if c not in reserved]


def _feature_names(claimed) -> list[str]:
    """Pull plausible feature-name strings out of the loosely-typed derived list."""
    names: list[str] = []
    if not isinstance(claimed, list):
        return names
    for item in claimed:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("feature") or item.get("field")
            if isinstance(name, str):
                names.append(name)
    return names
