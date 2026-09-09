"""Senior Data Engineer — task formatting (persona in backstories/data_engineer.md)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maads.output_contracts import schema_hint_for_agent
from maads.state import CrispDMState, SUBSTEP_NAMES

_SUBSTEP_ASSIGNMENTS: dict[str, dict[str, Any]] = {
    "2.1": {
        "objective": "Collect initial data and produce the Initial Data Collection Report",
        "requested_outputs": ["du.initial_data_collection_report"],
        "completion_criteria": [
            "Source files inventoried and readable",
            "Row counts recorded for every available table (test may be absent)",
        ],
        "constraints": ["Do not mutate raw source files"],
    },
    "2.2": {
        "objective": "Describe the data and produce the Data Description Report",
        "requested_outputs": ["du.data_description_report"],
        "completion_criteria": [
            "Column names, dtypes, missingness, and cardinality documented",
        ],
        "constraints": ["Ground claims in executed profiling only"],
    },
    "2.4": {
        "objective": "Verify data quality and produce the Data Quality Report",
        "requested_outputs": ["du.data_quality_report"],
        "completion_criteria": [
            "Blockers and tolerable issues classified with evidence",
            "na_means_absent and high_missing columns treated as documented absence, not blockers",
        ],
        "constraints": [
            "Parse na_means_absent and high_missing from DATASET_INSPECT_JSON / feature_hints; high NA on those columns is tolerable",
            "Reserve blockers for missing target, constants, undocumented corruption",
        ],
    },
    "3.1": {
        "objective": "Select data and document inclusion/exclusion rationale",
        "requested_outputs": ["dp.rationale_for_inclusion_exclusion"],
        "completion_criteria": ["Every retained or excluded field justified"],
        "constraints": ["Respect prediction-time availability"],
    },
    "3.2": {
        "objective": "Clean data and produce the Data Cleaning Report",
        "requested_outputs": ["dp.data_cleaning_report"],
        "completion_criteria": ["Cleaning strategy documented and leakage-safe"],
        "constraints": ["Classify operations as DETERMINISTIC or LEARNED"],
    },
    "3.3": {
        "objective": "Construct derived attributes when justified",
        "requested_outputs": ["dp.derived_attributes", "dp.generated_records"],
        "completion_criteria": ["Derived fields traceable to sources"],
        "constraints": ["No target leakage"],
    },
    "3.4": {
        "objective": "Integrate data sources when applicable",
        "requested_outputs": ["dp.merged_data"],
        "completion_criteria": ["Join cardinality validated"],
        "constraints": ["Document row-count effects"],
    },
    "3.5": {
        "objective": "Format data and produce downstream train/test datasets",
        "requested_outputs": ["dp.dataset", "dp.dataset_description"],
        "completion_criteria": [
            "Prepared train/test artifacts exist and are validated",
        ],
        "constraints": ["Preserve identifiers separately from features"],
    },
}

def _domain_evidence(state: CrispDMState) -> list[Any]:
    bu = state.bu
    inv = bu.inventory_of_resources or {}
    artifacts = inv.get("domain_artifacts") or {}
    evidence: list[Any] = []
    if bu.business_objectives:
        evidence.append({"kind": "business_objectives", "value": bu.business_objectives})
    if bu.data_mining_goals:
        evidence.append({"kind": "data_mining_goals", "value": bu.data_mining_goals})
    if artifacts:
        evidence.append({"kind": "domain_artifacts", "value": artifacts})
    if bu.terminology:
        evidence.append({"kind": "terminology", "value": bu.terminology})
    return evidence


def _assignment_for_substep(substep: str, state: CrispDMState) -> dict[str, Any]:
    meta = _SUBSTEP_ASSIGNMENTS.get(substep, {})
    return {
        "assignment_id": substep,
        "objective": meta.get("objective", f"Complete CRISP-DM substep {substep}"),
        "crisp_dm_phase": substep.split(".")[0],
        "crisp_dm_substeps": [substep],
        "requested_outputs": meta.get("requested_outputs", []),
        "completion_criteria": meta.get("completion_criteria", []),
        "constraints": meta.get("constraints", []),
        "substep_name": SUBSTEP_NAMES.get(substep, "?"),
        "case_id": state.case_id,
    }


def _inputs_for_task(
    state: CrispDMState,
    artifact_dir: Path,
    *,
    execution_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = state.config
    inputs: dict[str, Any] = {
        "source_locations": [p for p in [
            *([s.path for s in cfg.data.sources] if cfg.data.sources else []),
            cfg.data.train_csv,
            cfg.data.test_csv,
            cfg.data.sample_submission_csv,
        ] if p],
        "config_path": None,
        "metadata_paths": [],
        "upstream_artifacts": [],
        "sample_output_path": cfg.data.sample_submission_csv,
        "accepted_decisions": {
            "target_column": cfg.target_column,
            "id_column": cfg.id_column,
            "evaluation_metric": cfg.evaluation_metric,
            "problem_type": cfg.problem_type,
        },
        "domain_evidence": _domain_evidence(state),
        "revision_feedback": None,
        "data_mining_goals": state.bu.data_mining_goals,
    }
    if execution_evidence:
        inputs["execution_evidence"] = execution_evidence
    inputs["artifact_directory"] = str(artifact_dir.resolve())
    return inputs


def format_data_engineer_task(
    state: CrispDMState,
    artifact_dir: Path,
    *,
    execution_evidence: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Build the data-engineer assignment instruction and JSON schema hint."""
    assignment = _assignment_for_substep(state.substep, state)
    inputs = _inputs_for_task(state, artifact_dir, execution_evidence=execution_evidence)
    runtime_input = {
        "assignment": assignment,
        "inputs": inputs,
    }
    instruction = (
        "Complete the assigned CRISP-DM substep using the runtime input below. "
        "Ground every claim in execution_evidence when present. "
        "Return exactly one JSON object matching the output schema in your instructions.\n\n"
        f"Runtime input:\n{json.dumps(runtime_input, indent=2, default=str)}"
    )
    return instruction, schema_hint_for_agent("data_engineer", substep=state.substep)
