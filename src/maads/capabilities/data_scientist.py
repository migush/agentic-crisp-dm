"""Data Scientist capabilities — orchestrate deterministic modeling tools."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from maads.capabilities import ml_tools
from maads.capabilities.shared import (
    abspath as _abspath,
    execution_or_llm,
    record_degraded,
)
from maads.deltas import StateDelta
from maads.schema_inference import apply_inferred_schema
from maads.state import CrispDMState, EvaluationBundle, ModelRun, coerce_evaluation_bundle
from maads.success_criterion import normalize_assessment
from maads.tools import inspect_dataset


def _train_schema_context(dataset_train: str, id_col: str) -> tuple[list[str], str]:
    import pandas as pd

    cols = list(pd.read_parquet(dataset_train).columns)
    id_present = id_col in cols
    note = (
        f"TRAIN_PARQUET columns (measured): {cols}. "
        f"ID_COL ({id_col!r}) is {'present' if id_present else 'absent'} in train "
        "(train parquet intentionally drops identifier columns after prep 3.5; "
        "ID_COL is kept in test only). "
        "Drop features with: "
        "X = train.drop(columns=[c for c in (TARGET, ID_COL) if c in train.columns])."
    )
    return cols, note


def _primary_text_column_name(state: CrispDMState, columns: list[str] | None = None) -> str | None:
    hints = (state.config.feature_hints or {}).get("text_free") or []
    cols = set(columns or [])
    for col in hints:
        if not cols or col in cols:
            return str(col)
    if not cols or "text" in cols:
        return "text"
    return None


def _text_modeling_hint(state: CrispDMState) -> str:
    if not _is_text_modeling_case(state):
        return ""
    return (
        " Text modeling: use the deterministic tfidf_logreg baseline "
        "(FunctionTransformer + TfidfVectorizer is handled inside ml_tools)."
    )


def _is_text_modeling_case(state: CrispDMState) -> bool:
    return ml_tools.is_nlp_primary(state.config.feature_hints or {})


def _run_text_model_baseline(pyexec, header_vars: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible wrapper used by older tests; prefers in-process tools."""
    del pyexec
    return ml_tools.run_experiment(
        header_vars["TRAIN_PARQUET"],
        technique="tfidf_logreg",
        target=header_vars["TARGET"],
        id_column=header_vars.get("ID_COL") or "",
        metric=header_vars.get("METRIC") or "f1",
        problem_type=header_vars.get("PROBLEM_TYPE") or "binary_classification",
        feature_hints={"text_free": [header_vars.get("PRIMARY_TEXT_COL") or "text"]},
    )


def _run_text_assess_baseline(pyexec, header_vars: dict[str, Any]) -> dict[str, Any]:
    del pyexec
    import json

    class_labels = header_vars.get("CLASS_LABELS") or {}
    if isinstance(class_labels, str):
        class_labels = json.loads(class_labels) if class_labels.strip() else {}
    return ml_tools.assess_fitted_experiment(
        header_vars["TRAIN_PARQUET"],
        technique=header_vars.get("MODEL_TECHNIQUE") or "tfidf_logreg",
        target=header_vars["TARGET"],
        id_column=header_vars.get("ID_COL") or "",
        metric="f1",
        problem_type=header_vars.get("PROBLEM_TYPE") or "binary_classification",
        feature_hints={"text_free": [header_vars.get("PRIMARY_TEXT_COL") or "text"]},
        figures_dir=header_vars.get("FIGURES_DIR"),
        class_labels=class_labels,
    )


def _model_technique_from_state(state: CrispDMState) -> str:
    if state.md.models:
        best = ml_tools.select_best_model(
            state.md.models,
            metric=state.config.evaluation_metric,
            direction=(state.config.success_criterion.direction if state.config.success_criterion else None),
        )
        return best.technique or state.md.modeling_technique or "tfidf_logreg"
    return state.md.modeling_technique or "logistic_regression"


def _artifact_from_state(state: CrispDMState) -> str | None:
    if state.md.chosen_model and state.md.chosen_model.artifact_path:
        return state.md.chosen_model.artifact_path
    if state.md.models:
        try:
            best = ml_tools.select_best_model(
                state.md.models,
                metric=state.config.evaluation_metric,
                direction=(state.config.success_criterion.direction if state.config.success_criterion else None),
            )
        except ValueError:
            best = state.md.models[-1]
        if best.artifact_path:
            return best.artifact_path
        path = (best.parameter_settings or {}).get("artifact_path")
        if path:
            return str(path)
    return None


def execution_evidence(
    pyexec,
    state: CrispDMState,
    substep: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Run deterministic DS tools for exploration and modeling substeps."""
    del pyexec
    train = _abspath(state.config.data.train_csv)
    target = state.resolved_target()
    hints = state.config.feature_hints or {}

    if substep == "2.3":
        summary = inspect_dataset(
            train,
            None,
            target_column=state.resolved_target() or None,
        )
        if summary.get("error"):
            raise RuntimeError(f"dataset inspect failed: {summary['error']}")
        profile = ml_tools.profile_dataset(
            train,
            target=target or None,
            id_column=state.config.id_column or None,
            na_means_absent=list(hints.get("na_means_absent") or []),
        )
        schema_fields = apply_inferred_schema(state, profile=profile)
        target = state.resolved_target()
        if target and not (
            isinstance(profile.get("target"), dict) and profile["target"].get("name") == target
        ):
            profile = ml_tools.profile_dataset(
                train,
                target=target,
                id_column=state.config.id_column or None,
                na_means_absent=list(hints.get("na_means_absent") or []),
            )
        return {
            "data_exploration_report": ml_tools.explore_report_from_profile(profile, target),
            "schema_fields": schema_fields,
        }

    if substep == "4.3":
        dataset_train = state.dp.dataset.get("train")
        if not dataset_train:
            return {}
        schema_fields: list[str] = []
        if not state.resolved_target():
            last_profile = ml_tools.profile_dataset(
                dataset_train,
                target=None,
                id_column=state.config.id_column or None,
            )
            schema_fields = apply_inferred_schema(state, profile=last_profile)
        target = state.resolved_target()
        if not target:
            raise RuntimeError(
                "target_column is unset; cannot run modeling baselines"
            )
        hints = state.config.feature_hints or {}
        findings = ml_tools.lint_prepared_features(
            dataset_train,
            target=target,
            id_column=state.config.id_column,
        )
        if findings:
            state.validator_findings = list(
                dict.fromkeys([*(state.validator_findings or []), *findings]),
            )
        techniques = ml_tools.baseline_techniques_for(
            problem_type=state.config.problem_type,
            feature_hints=hints,
            preferred=state.md.modeling_technique,
        )
        compared = ml_tools.compare_experiments(
            dataset_train,
            techniques,
            target=target,
            id_column=state.config.id_column,
            metric=state.config.evaluation_metric,
            problem_type=state.config.problem_type,
            feature_hints=hints,
            artifact_dir=artifact_dir,
            direction=(state.config.success_criterion.direction if state.config.success_criterion else None),
        )
        best = compared["best"]
        return {
            "model_run": {
                "technique": best.get("technique") or "unspecified",
                "cv_score": best.get("cv_score"),
                "cv_std": best.get("cv_std"),
                "description": (
                    f"{best.get('n_features', '?')} features, CV; "
                    f"ladder={techniques}; source=deterministic"
                ),
                "parameter_settings": best.get("parameter_settings") or {},
                "artifact_path": best.get("artifact_path") or "",
            },
            "candidate_runs": compared.get("candidates") or [],
            "schema_fields": schema_fields,
        }

    if substep == "4.4":
        dataset_train = state.dp.dataset.get("train")
        if not dataset_train:
            return {}
        figures_dir = str((artifact_dir / "figures").resolve())
        technique = _model_technique_from_state(state)
        artifact_path = _artifact_from_state(state)
        assessed = ml_tools.assess_fitted_experiment(
            dataset_train,
            technique=technique,
            target=target,
            id_column=state.config.id_column,
            metric=state.config.evaluation_metric,
            problem_type=state.config.problem_type,
            feature_hints=hints,
            artifact_path=artifact_path,
            figures_dir=figures_dir,
            class_labels=state.config.class_labels or {},
        )
        return {
            "evaluation_bundle": assessed.get("evaluation_bundle"),
            "assessment": assessed.get("assessment"),
        }
    return {}


_DS_EXECUTION_AUTHORITY_KEYS: dict[str, tuple[str, ...]] = {
    "2.3": ("data_exploration_report",),
    "4.3": ("model_run",),
    "4.4": ("evaluation_bundle",),
}


def _execution_authoritative(execution: dict[str, Any], substep: str) -> bool:
    return any(
        execution.get(key) is not None
        for key in _DS_EXECUTION_AUTHORITY_KEYS.get(substep, ())
    )


def apply_response(
    data: dict,
    state: CrispDMState,
    substep: str,
    execution: dict[str, Any],
) -> StateDelta:
    """Map data-scientist JSON (or execution evidence) into shared state."""
    from maads.output_contracts import validate_agent_output

    if not _execution_authoritative(execution, substep):
        schema_errors = validate_agent_output("data_scientist", data, substep=substep)
        if schema_errors:
            return StateDelta(
                notes=f"DS {substep}: schema-invalid response: {schema_errors[0]}",
                failed=True,
            )

    su = (data or {}).get("state_updates") or {}
    du = su.get("du") or {}
    md = su.get("md") or {}
    ev = su.get("ev") or {}
    fields: list[str] = list(execution.get("schema_fields") or [])

    if substep == "2.3":
        report = execution_or_llm(execution, du, "data_exploration_report")
        if not report:
            desc = state.du.data_description_report or {}
            report = {"n_rows": desc.get("n_rows"), "target": state.resolved_target()}
        state.du.data_exploration_report = report
        fields.append("du.data_exploration_report")
        inferred = (report or {}).get("target") if isinstance(report, dict) else None
        if state.adopt_target_if_blank(inferred if isinstance(inferred, str) else None):
            fields.append("config.target_column")
    elif substep == "4.1":
        preferred = md.get("modeling_technique")
        ladder = ml_tools.baseline_techniques_for(
            problem_type=state.config.problem_type,
            feature_hints=state.config.feature_hints or {},
            preferred=preferred,
        )
        state.md.modeling_technique = preferred or (ladder[0] if ladder else "to be chosen at 4.3")
        state.md.modeling_assumptions = md.get("modeling_assumptions") or [
            "tabular or text features from feature_hints",
            "no leakage (pipeline fit on train folds only)",
            f"baseline ladder: {ladder}",
        ]
        fields.extend(["md.modeling_technique", "md.modeling_assumptions"])
    elif substep == "4.2":
        state.md.test_design = md.get("test_design") or {
            "cv": "stratified_5fold" if state.config.problem_type != "regression" else "kfold_5",
            "metric": state.config.evaluation_metric,
            "direction": criterion_direction_safe(state),
        }
        fields.append("md.test_design")
    elif substep == "4.3":
        run = dict(execution.get("model_run") or {})
        llm_run = md.get("model_run") or {}
        if llm_run.get("description") and not run.get("description"):
            run["description"] = llm_run["description"]
        if not run:
            return StateDelta(notes="DS 4.3: no model execution evidence")
        technique = run.get("technique") or "unspecified"
        artifact_path = run.get("artifact_path") or (run.get("parameter_settings") or {}).get("artifact_path")
        state.md.models.append(ModelRun(
            technique=technique,
            cv_score=run.get("cv_score"),
            cv_std=run.get("cv_std"),
            description=run.get("description") or "model run",
            parameter_settings=run.get("parameter_settings") or {},
            artifact_path=str(artifact_path) if artifact_path else None,
        ))
        # Also record other ladder candidates that produced scores.
        for cand in execution.get("candidate_runs") or []:
            if not isinstance(cand, dict) or cand.get("technique") == technique:
                continue
            if not isinstance(cand.get("cv_score"), (int, float)):
                if cand.get("error"):
                    record_degraded(state, "4.3", "data_scientist", f"{cand.get('technique')}: {cand.get('error')}")
                continue
            state.md.models.append(ModelRun(
                technique=str(cand.get("technique") or "candidate"),
                cv_score=cand.get("cv_score"),
                cv_std=cand.get("cv_std"),
                description="ladder candidate",
                parameter_settings=cand.get("parameter_settings") or {},
                artifact_path=cand.get("artifact_path") or None,
            ))
        state.md.modeling_technique = technique
        fields.extend(["md.models", "md.modeling_technique"])
    elif substep == "4.4":
        bundle_raw = execution.get("evaluation_bundle")
        if state.md.models:
            try:
                best = ml_tools.select_best_model(
                    state.md.models,
                    metric=state.config.evaluation_metric,
                    direction=(state.config.success_criterion.direction if state.config.success_criterion else None),
                )
            except ValueError:
                best = state.md.models[-1]
            chosen = md.get("chosen_model_technique")
            if chosen:
                for m in state.md.models:
                    if m.technique == chosen:
                        best = m
                        break
            best.assessment = (
                execution.get("assessment")
                or md.get("assessment")
                or "selected: direction-aware best CV score"
            )
            if bundle_raw:
                try:
                    best.evaluation_bundle = EvaluationBundle.model_validate(
                        coerce_evaluation_bundle(
                            bundle_raw,
                            problem_type=state.config.problem_type,
                            class_labels=state.config.class_labels or {},
                        ),
                    )
                except ValidationError as exc:
                    return StateDelta(
                        notes=f"DS 4.4: invalid evaluation_bundle: {exc}",
                        failed=True,
                    )
            state.md.chosen_model = best
            fields.append("md.chosen_model")
        elif not bundle_raw:
            return StateDelta(notes="DS 4.4: no evaluation_bundle from execution", failed=True)
    elif substep == "5.1":
        cv = state.md.chosen_model.cv_score if state.md.chosen_model else None
        sc = state.config.success_criterion
        raw = ev.get("assessment_of_dm_results") or {
            "cv_score": cv,
            "threshold": sc.threshold,
        }
        state.ev.assessment_of_dm_results = normalize_assessment(
            raw,
            metric=sc.metric,
            threshold=sc.threshold,
            direction=sc.direction,
            cv_score=cv,
        )
        fields.append("ev.assessment_of_dm_results")
        if state.md.chosen_model:
            state.ev.approved_models = [state.md.chosen_model]
            fields.append("ev.approved_models")

    summary = (data or {}).get("summary", "")
    return StateDelta(fields, notes=summary or f"DS completed {substep}")


def criterion_direction_safe(state: CrispDMState) -> str:
    from maads.success_criterion import criterion_direction

    sc = state.config.success_criterion
    return criterion_direction(
        state.config.evaluation_metric,
        sc.direction if sc else None,
    )
