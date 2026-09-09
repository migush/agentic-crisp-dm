"""Senior Developer — task formatting (persona in backstories/developer.md)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maads.output_contracts import schema_hint_for_agent
from maads.state import CrispDMState, SUBSTEP_NAMES

MAX_DEBUG_RETRIES = 3

DEVELOPER_OUTPUT_SCHEMA_HINT = schema_hint_for_agent("developer")

_SUBSTEP_ASSIGNMENTS: dict[str, dict[str, Any]] = {
    "6.1": {
        "mode": "DEPLOY",
        "objective": "Build a schema-valid submission.csv",
        "requested_outputs": [
            "dep.deployment_plan",
            "dep.submission_path"
        ],
        "completion_criteria": [
            "submission written; validated against sample_submission_csv when that file exists",
            "dep.submission_path set to verified artifact"
        ],
        "constraints": [
            "Never treat sample submission as ground-truth labels"
        ]
    },
    "6.4": {
        "mode": "DEPLOY",
        "objective": "Review the project and document experience",
        "requested_outputs": [
            "dep.experience_documentation"
        ],
        "completion_criteria": [
            "Honest review of what worked, broke, and lessons for next runs"
        ],
        "constraints": []
    }
}


def _assignment_for_substep(substep: str, state: CrispDMState) -> dict[str, Any]:
    meta = _SUBSTEP_ASSIGNMENTS.get(substep, {})
    return {
        "assignment_id": substep,
        "mode": meta.get("mode", "DEPLOY"),
        "objective": meta.get("objective", f"Complete CRISP-DM substep {substep}"),
        "crisp_dm_substeps": [substep],
        "requesting_agent": None,
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
        "dataset": state.dp.dataset,
        "chosen_model": (
            state.md.chosen_model.model_dump() if state.md.chosen_model else None
        ),
        "sample_submission_csv": cfg.data.sample_submission_csv,
        "retry_budget": 3,
        "upstream_artifacts": [],
    }
    if execution_evidence:
        inputs["execution_evidence"] = execution_evidence
    inputs["artifact_directory"] = str(artifact_dir.resolve())
    return inputs


def format_developer_debug_task(
    state: CrispDMState,
    artifact_dir: Path,
    *,
    failure_kind: str,
    requesting_agent: str,
    error_class: str,
    last_error: str,
    stderr_excerpt: str,
    failing_code: str,
    header_var_names: list[str],
    contract_hint: str,
    schema_columns: list[str],
    attempt: int = 1,
    max_retries: int = MAX_DEBUG_RETRIES,
    malformed_json: str = "",
    task_instruction: str = "",
    schema_errors: list[str] | None = None,
) -> str:
    """Build a DEBUG-mode instruction for the Developer LLM."""
    runtime_input = {
        "assignment": {
            "assignment_id": f"debug-{state.substep}",
            "mode": "DEBUG",
            "objective": f"Repair {failure_kind} failure from {requesting_agent}",
            "requesting_agent": requesting_agent,
            "crisp_dm_substep": state.substep,
            "case_id": state.case_id,
            "expected_assignment_id": state.substep,
            "expected_agent": requesting_agent,
        },
        "failure": {
            "kind": failure_kind,
            "error_class": error_class,
            "last_error": last_error,
            "stderr_excerpt": stderr_excerpt,
            "attempt": attempt,
            "retry_budget": max_retries,
            "schema_errors": schema_errors or [],
        },
        "inputs": {
            "failing_code": failing_code or None,
            "header_variables": header_var_names,
            "contract_hint": contract_hint,
            "schema_columns": schema_columns,
            "malformed_json": malformed_json or None,
            "original_task_instruction": task_instruction or None,
        },
        "artifact_directory": str(artifact_dir.resolve()),
    }
    if failure_kind == "python_exec":
        deliverable = (
            "Return ONLY a ```python ...``` code block containing the SMALLEST fix. "
            "Do not redefine injected header variables. The code must print exactly "
            "one JSON object line to stdout when executed."
        )
    else:
        schema_fix = ""
        if schema_errors:
            schema_fix = (
                " Fix these validation errors: "
                + "; ".join(schema_errors[:8])
                + ". assumptions/risks/blockers/handoffs must be arrays of objects "
                '(e.g. [{"statement": "..."}]), not bare strings.'
            )
        deliverable = (
            f"Return ONLY the repaired {requesting_agent} substep JSON for CRISP-DM "
            f"substep {state.substep}. assignment_id must be '{state.substep}' and "
            f"agent must be '{requesting_agent}'. Do NOT return debug metadata "
            "(mode, diagnosis, status=FIXED). List fields must be JSON arrays [], "
            "never empty strings. state_updates must be an object."
            f"{schema_fix} "
            "Your response must begin with '{' and end with '}'. "
            "Do not include markdown wraps."
        )
    return (
        "DEBUG mode — you are the on-call Developer. Diagnose the failure below, "
        "apply the smallest correct fix, and return executable output.\n"
        f"{deliverable}\n\n"
        f"Runtime input:\n{json.dumps(runtime_input, indent=2, default=str)}"
    )


def format_developer_task(
    state: CrispDMState,
    artifact_dir: Path,
    *,
    execution_evidence: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Build the developer assignment instruction and JSON schema hint."""
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
    return instruction, DEVELOPER_OUTPUT_SCHEMA_HINT
