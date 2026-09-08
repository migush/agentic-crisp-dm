"""Data Engineer capabilities — CRISP-DM-independent execution API."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from maads.codegen import run_authored_code
from maads.deltas import StateDelta
from maads.state import CrispDMState
from maads.tools import PythonExec

from maads.capabilities.shared import (
    abspath as _abspath,
    de_dataset_context as _de_dataset_context,
    has_keys as _has_keys,
    prep_inputs as _prep_inputs,
    prep_workdir as _prep_workdir,
    describe_data_contract as _describe_data_contract,
    measure_prep_artifacts as _measure_prep_artifacts,
    require_inspect_ok as _require_inspect_ok,
    target_preserved as _target_preserved,
)
from maads.config import primary_train_csv, source_locations

_QUALITY_24_INSTRUCTION = (
    "CRISP-DM 2.4 Verify Data Quality: inspect the training data and classify "
    "genuine quality BLOCKERS vs tolerable issues. Compute from the data. "
    "Parse DATASET_INSPECT_JSON for na_means_absent: columns listed there use "
    "NA to mean feature absence (not missing data) — high missingness on those "
    "columns is NOT a blocker; record as tolerable with note 'structural absence "
    "(no feature)'. Blockers are reserved for: missing target, constant columns, "
    "duplicate-ID issues, undocumented high missingness, and schema contradictions."
)


def execution_evidence(
    pyexec: PythonExec,
    state: CrispDMState,
    substep: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Have the Data Engineer author and run the code for its owned substep.

    Returns measured evidence in the shape `_apply_data_engineer_response`
    expects. Each authored attempt self-debugs via codegen.run_authored_code
    (with Developer DEBUG on exhaustion); no baseline fallback.
    """
    train = _abspath(state.config.data.train_csv) or _abspath(primary_train_csv(state.config.data))
    test = _abspath(state.config.data.test_csv)
    target = state.resolved_target()
    idc = state.config.id_column
    ds_ctx = _de_dataset_context(state, train, test)
    _require_inspect_ok(ds_ctx)
    sources_json = json.dumps(source_locations(state.config.data))
    missing_note = (
        " Missing TEST_CSV or sample submission is work for this phase, not an upload error: "
        "inventory SOURCE_PATHS, decide file roles from evidence, and create a modelling "
        "holdout or submission schema only when the goal requires it."
    )

    if substep == "2.1":
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_engineer", state=state,
            instruction="CRISP-DM 2.1 Collect Initial Data: load the available source "
                        "CSVs (TRAIN_CSV, TEST_CSV when non-empty, and every path in "
                        "SOURCE_PATHS) and report a brief collection summary. "
                        "Use DATASET_INSPECT_JSON for column alignment hints."
                        + missing_note,
            header_vars={
                "TRAIN_CSV": train, "TEST_CSV": test, "SOURCE_PATHS": sources_json, **ds_ctx,
            },
            contract=lambda p: _has_keys(p, "train_rows", "columns"),
            contract_hint="Required keys: train_rows (int), columns (list). Include test_rows when a test file exists.",
            artifact_dir=artifact_dir,
        )
        return {"initial_data_collection_report": res.payload}

    if substep == "2.2":
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_engineer", state=state,
            instruction="CRISP-DM 2.2 Describe Data: profile the training data — "
                        "row/column counts, dtypes, missing counts, cardinality.",
            header_vars={"TRAIN_CSV": train, "SOURCE_PATHS": sources_json, **ds_ctx},
            contract=_describe_data_contract,
            contract_hint=(
                "Required keys: n_rows (int), n_cols (int), columns (list of str), "
                "dtypes (dict column name -> dtype string), "
                "missing (dict column name -> int count). "
                "Do not emit parallel lists for dtypes or missing."
            ),
            artifact_dir=artifact_dir,
        )
        return {"data_description_report": res.payload}

    if substep == "2.4":
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_engineer", state=state,
            instruction=_QUALITY_24_INSTRUCTION,
            header_vars={"TRAIN_CSV": train, "TARGET": target, "SOURCE_PATHS": sources_json, **ds_ctx},
            contract=lambda p: _has_keys(p, "blockers", "tolerable"),
            contract_hint="Required keys: blockers (list of strings), tolerable (list of strings).",
            artifact_dir=artifact_dir,
        )
        return {"data_quality_report": res.payload}

    prep_wd = str(_prep_workdir(artifact_dir).resolve())
    train_in, test_in = _prep_inputs(artifact_dir, state, substep)

    if substep == "3.2":
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_engineer", state=state,
            instruction="CRISP-DM 3.2 Clean Data: read TRAIN_IN and TEST_IN (TEST_IN may "
                        "be empty — then decide whether to split or leave holdout for later), "
                        "apply leakage-safe cleaning (impute missing, fix invalid values), "
                        "write train_clean.parquet and test_clean.parquet under OUTDIR, "
                        "and report missing counts before and after. Missing test/submission/"
                        "integer labels are work for this phase, not upload errors.",
            header_vars={
                "TRAIN_IN": train_in, "TEST_IN": test_in, "OUTDIR": prep_wd, "TARGET": target,
                **ds_ctx,
            },
            contract=lambda p: (
                _has_keys(p, "train_out", "test_out")
                or _has_keys(p, "missing_before", "missing_after")
            ) + _target_preserved(p, target),
            contract_hint="Required keys: train_out, test_out (paths), missing_before, "
                          "missing_after (per-column int counts).",
            artifact_dir=artifact_dir,
        )
        payload = res.payload
        return {
            "data_cleaning_report": {
                "missing_before": payload.get("missing_before"),
                "missing_after_train": payload.get("missing_after"),
                "operations": payload.get("operations") or [],
                "train_out": payload.get("train_out"),
                "test_out": payload.get("test_out"),
                "source": "executed at 3.2",
            },
        }

    if substep == "3.3":
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_engineer", state=state,
            instruction="CRISP-DM 3.3 Construct Data: read TRAIN_IN and TEST_IN, add "
                        "justified derived features available at prediction time, write "
                        "train_constructed.parquet and test_constructed.parquet under OUTDIR, "
                        "and list derived feature names.",
            header_vars={
                "TRAIN_IN": train_in, "TEST_IN": test_in, "OUTDIR": prep_wd,
                "TARGET": target, **ds_ctx,
            },
            contract=lambda p: _has_keys(p, "train_out", "test_out", "derived")
            + _target_preserved(p, target),
            contract_hint="Required keys: train_out, test_out (paths), derived (list of names).",
            artifact_dir=artifact_dir,
        )
        payload = res.payload
        derived = payload.get("derived") or []
        return {
            "derived_attributes": {
                "items": [
                    item if isinstance(item, dict) else {"field": str(item), "source": "executed at 3.3"}
                    for item in derived
                ],
            },
            "generated_records": {"count": 0, "source": "executed at 3.3"},
        }

    if substep == "3.4":
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_engineer", state=state,
            instruction="CRISP-DM 3.4 Integrate Data: read TRAIN_IN and TEST_IN, validate "
                        "schema compatibility and row granularity, write "
                        "train_integrated.parquet and test_integrated.parquet under OUTDIR, "
                        "and report merged row/column counts. The TARGET column exists only "
                        "in train (not test) — this is expected; never drop TARGET to make "
                        "the schemas match. Keep all train columns; only align shared "
                        "feature columns.",
            header_vars={
                "TRAIN_IN": train_in, "TEST_IN": test_in, "OUTDIR": prep_wd,
                "TARGET": target, **ds_ctx,
            },
            contract=lambda p: _has_keys(p, "train_out", "test_out", "train_rows", "test_rows")
            + _target_preserved(p, target),
            contract_hint="Required keys: train_out, test_out, train_rows, test_rows, "
                          "columns_train, columns_test.",
            artifact_dir=artifact_dir,
        )
        payload = res.payload
        return {
            "merged_data": {
                "train_rows": payload.get("train_rows"),
                "test_rows": payload.get("test_rows"),
                "columns_train": payload.get("columns_train"),
                "columns_test": payload.get("columns_test"),
                "source": "executed at 3.4",
            },
        }

    if substep == "3.5":
        outdir = str(artifact_dir.resolve())
        source_train = train
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_engineer", state=state,
            instruction="CRISP-DM 3.5 Format Data: read TRAIN_IN and TEST_IN (create a "
                        "modelling holdout if only one table exists), drop identifier/"
                        "leakage columns, keep TARGET in train when it is known, keep "
                        "ID_COL in test when it is known, and write final train.parquet "
                        "and test.parquet into OUTDIR. A labelled second file may be "
                        "holdout, leakage, or something else — decide from evidence. "
                        "Invent a submission schema only if the goal needs predictions.",
            header_vars={
                "TRAIN_IN": train_in, "TEST_IN": test_in, "OUTDIR": outdir,
                "SOURCE_TRAIN": source_train, "TARGET": target, "ID_COL": idc,
                **ds_ctx,
            },
            contract=lambda p: (
                _has_keys(p, "train", "test", "n_train", "n_test")
                or ([] if int(p.get("n_train", 0)) > 0 else ["n_train must be > 0"])
            ) + _target_preserved(p, target, path_key="train"),
            contract_hint="Required keys: train (parquet path), test (parquet path), "
                          "n_train (int>0), n_test (int), derived (list), dropped (list).",
            artifact_dir=artifact_dir,
        )
        info = res.payload
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
        return {
            "dataset": {"train": info["train"], "test": info["test"]},
            "dataset_description": (
                f"{info.get('n_train')} train / {info.get('n_test')} test rows (parquet); "
                f"{n_derived} derived feature(s) reported by code"
            ),
            "derived": info.get("derived") or [],
            "dropped": info.get("dropped") or [],
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
    fields: list[str] = []

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
