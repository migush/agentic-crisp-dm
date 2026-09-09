"""Data Engineer capabilities — orchestrate deterministic data tools."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from maads.capabilities import ml_tools
from maads.capabilities.shared import (
    abspath as _abspath,
    de_dataset_context as _de_dataset_context,
    measure_prep_artifacts as _measure_prep_artifacts,
    prep_inputs as _prep_inputs,
    prep_workdir as _prep_workdir,
    require_inspect_ok as _require_inspect_ok,
)
from maads.config import primary_train_csv, source_locations
from maads.deltas import StateDelta
from maads.schema_inference import apply_inferred_schema
from maads.state import CrispDMState
from maads.tools import PythonExec


def execution_evidence(
    pyexec: PythonExec,
    state: CrispDMState,
    substep: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Run deterministic DE tools for owned execution substeps.

    ``pyexec`` is retained for API compatibility with agents/tests; standard
    DU/DP paths do not author freeform Python.
    """
    del pyexec  # unused on the deterministic path
    schema_fields = apply_inferred_schema(state)
    train = _abspath(state.config.data.train_csv) or _abspath(primary_train_csv(state.config.data))
    test = _abspath(state.config.data.test_csv)
    target = state.resolved_target()
    idc = state.config.id_column
    hints = state.config.feature_hints or {}
    ds_ctx = _de_dataset_context(state, train, test)
    _require_inspect_ok(ds_ctx)
    sources = source_locations(state.config.data)
    na_absent = list(hints.get("na_means_absent") or [])
    high_missing = list(hints.get("high_missing") or [])

    if substep == "2.1":
        return {
            "initial_data_collection_report": ml_tools.collect_report(
                train, test, source_paths=sources or None,
            ),
            "schema_fields": schema_fields,
        }

    if substep == "2.2":
        profile = ml_tools.profile_dataset(
            train, test or None, target=target or None, id_column=idc or None,
            na_means_absent=na_absent, high_missing=high_missing,
        )
        schema_fields = list(dict.fromkeys([*schema_fields, *apply_inferred_schema(state, profile=profile)]))
        return {
            "data_description_report": ml_tools.describe_report_from_profile(profile),
            "schema_fields": schema_fields,
        }

    if substep == "2.4":
        profile = ml_tools.profile_dataset(
            train, test or None, target=target or None, id_column=idc or None,
            na_means_absent=na_absent, high_missing=high_missing,
        )
        return {
            "data_quality_report": ml_tools.quality_report_from_profile(
                profile, target=target, na_means_absent=na_absent,
                high_missing=high_missing,
            ),
        }

    prep_wd = str(_prep_workdir(artifact_dir).resolve())
    train_in, test_in = _prep_inputs(artifact_dir, state, substep)

    if substep == "3.2":
        payload = ml_tools.clean_tables(
            train_in, test_in, prep_wd, target=target, feature_hints=hints,
        )
        return {
            "data_cleaning_report": {
                "missing_before": payload.get("missing_before"),
                "missing_after_train": payload.get("missing_after"),
                "operations": payload.get("operations") or [],
                "train_out": payload.get("train_out"),
                "test_out": payload.get("test_out"),
                "source": payload.get("source") or "deterministic clean_tables",
            },
        }

    if substep == "3.3":
        payload = ml_tools.construct_tables(
            train_in, test_in, prep_wd, target=target, feature_hints=hints,
        )
        derived = payload.get("derived") or []
        return {
            "derived_attributes": {
                "items": [
                    item if isinstance(item, dict) else {"field": str(item), "source": "deterministic 3.3"}
                    for item in derived
                ],
            },
            "generated_records": {"count": 0, "source": "deterministic 3.3"},
        }

    if substep == "3.4":
        payload = ml_tools.integrate_tables(train_in, test_in, prep_wd, target=target)
        return {
            "merged_data": {
                "train_rows": payload.get("train_rows"),
                "test_rows": payload.get("test_rows"),
                "columns_train": payload.get("columns_train"),
                "columns_test": payload.get("columns_test"),
                "source": payload.get("source") or "deterministic integrate_tables",
            },
        }

    if substep == "3.5":
        if not state.resolved_target() and train_in:
            last_profile = ml_tools.profile_dataset(
                train_in, test_in or None,
                target=None, id_column=idc or None,
            )
            schema_fields = list(dict.fromkeys([
                *schema_fields, *apply_inferred_schema(state, profile=last_profile),
            ]))
        target = state.resolved_target()
        idc = state.config.id_column
        hints = state.config.feature_hints or {}
        outdir = str(artifact_dir.resolve())
        info = ml_tools.format_tables(
            train_in, test_in, outdir,
            target=target, id_column=idc, feature_hints=hints,
        )
        n_derived = len(info.get("derived") or [])
        measured = _measure_prep_artifacts(
            source_train=train,
            source_test=test,
            train_parquet=info["train"],
            test_parquet=info["test"],
            target=target,
            payload_derived=info.get("derived") or [],
            payload_dropped=info.get("dropped") or [],
        )
        findings = ml_tools.lint_prepared_features(
            info["train"], target=target, id_column=idc,
        )
        if findings:
            state.validator_findings = list(
                dict.fromkeys([*(state.validator_findings or []), *findings]),
            )
        return {
            "dataset": {"train": info["train"], "test": info["test"]},
            "dataset_description": (
                f"{info.get('n_train')} train / {info.get('n_test')} test rows (parquet); "
                f"{n_derived} derived feature(s); deterministic format_tables"
            ),
            "derived": info.get("derived") or [],
            "dropped": info.get("dropped") or [],
            "schema_fields": schema_fields,
            **measured,
        }
    return {}


_EXECUTION_AUTHORITY_KEYS: dict[str, tuple[str, ...]] = {
    "2.1": ("initial_data_collection_report",),
    "2.2": ("data_description_report",),
    "2.4": ("data_quality_report",),
    "3.2": ("data_cleaning_report",),
    "3.3": ("derived_attributes", "generated_records"),
    "3.4": ("merged_data",),
    "3.5": ("dataset", "dataset_description"),
}


def _execution_authoritative(execution: dict[str, Any], substep: str) -> bool:
    return any(
        execution.get(key) is not None
        for key in _EXECUTION_AUTHORITY_KEYS.get(substep, ())
    )


def apply_response(
    data: dict,
    state: CrispDMState,
    substep: str,
    execution: dict[str, Any],
) -> StateDelta:
    from maads.capabilities.shared import execution_or_llm
    from maads.output_contracts import validate_agent_output

    if not _execution_authoritative(execution, substep):
        schema_errors = validate_agent_output("data_engineer", data, substep=substep)
        if schema_errors:
            return StateDelta(
                notes=f"DE {substep}: schema-invalid response: {schema_errors[0]}",
                failed=True,
            )

    su = (data or {}).get("state_updates") or {}
    du = su.get("du") or {}
    dp = su.get("dp") or {}
    fields: list[str] = list(execution.get("schema_fields") or [])

    if substep == "2.1":
        report = execution_or_llm(execution, du, "initial_data_collection_report")
        if report:
            state.du.initial_data_collection_report = report
            fields.append("du.initial_data_collection_report")
    elif substep == "2.2":
        report = execution_or_llm(execution, du, "data_description_report")
        if report:
            state.du.data_description_report = report
            fields.append("du.data_description_report")
    elif substep == "2.4":
        report = execution_or_llm(execution, du, "data_quality_report")
        if report:
            state.du.data_quality_report = report
            fields.append("du.data_quality_report")
    elif substep == "3.1":
        rationale = dp.get("rationale_for_inclusion_exclusion")
        if rationale:
            from maads.text_normalize import normalize_inclusion_rationale

            state.dp.rationale_for_inclusion_exclusion = normalize_inclusion_rationale(
                rationale,
            )
            fields.append("dp.rationale_for_inclusion_exclusion")
    elif substep == "3.2":
        cleaning = execution_or_llm(execution, dp, "data_cleaning_report")
        if cleaning:
            state.dp.data_cleaning_report = cleaning
            fields.append("dp.data_cleaning_report")
    elif substep == "3.3":
        derived = execution_or_llm(execution, dp, "derived_attributes")
        if derived:
            state.dp.derived_attributes = derived
            fields.append("dp.derived_attributes")
        generated = execution_or_llm(execution, dp, "generated_records")
        if generated:
            state.dp.generated_records = generated
            fields.append("dp.generated_records")
    elif substep == "3.4":
        merged = execution_or_llm(execution, dp, "merged_data")
        if merged:
            state.dp.merged_data = merged
            fields.append("dp.merged_data")
    elif substep == "3.5":
        dataset = execution.get("dataset")
        description = execution.get("dataset_description")
        if dataset:
            state.dp.dataset = dataset
            fields.append("dp.dataset")
        if description:
            state.dp.dataset_description = description
            fields.append("dp.dataset_description")
        cleaning = execution.get("data_cleaning_report")
        if cleaning:
            state.dp.data_cleaning_report = cleaning
            fields.append("dp.data_cleaning_report")
        derived = execution.get("derived_attributes")
        if derived:
            state.dp.derived_attributes = derived
            fields.append("dp.derived_attributes")
        merged = execution.get("merged_data")
        if merged:
            state.dp.merged_data = merged
            fields.append("dp.merged_data")

    summary = (data or {}).get("summary", "")
    return StateDelta(fields, notes=summary or f"DE completed {substep}")
