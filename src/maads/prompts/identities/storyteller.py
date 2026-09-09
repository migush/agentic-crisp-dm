"""Storyteller — task formatting."""
from __future__ import annotations

import json
from pathlib import Path

from maads.output_contracts import schema_hint_for_agent
from maads.state import CrispDMState, SUBSTEP_NAMES

_SUBSTEP_ASSIGNMENTS: dict[str, dict] = {
    "6.2": {
        "objective": "Generate report evidence and storytelling specification from evaluation_bundle",
        "requested_outputs": ["story_spec", "dep.story_spec_path"],
        "completion_criteria": [
            "story_spec references only metrics present in evaluation_bundle",
            "interpretations use real class names from class_labels",
            "methodological warnings flag imbalance or degraded steps when present",
        ],
        "constraints": [
            "Do not invent metrics or figures not in evaluation_bundle",
        ],
    },
    "6.3": {
        "objective": "Produce final_report.md (handled deterministically by capability layer)",
        "requested_outputs": ["dep.final_report_path"],
        "completion_criteria": ["Report path recorded after render"],
        "constraints": [],
    },
}


def _assignment_for_substep(substep: str, state: CrispDMState) -> dict:
    meta = _SUBSTEP_ASSIGNMENTS.get(substep, {})
    return {
        "assignment_id": substep,
        "objective": meta.get("objective", f"Complete CRISP-DM substep {substep}"),
        "crisp_dm_phase": substep.split(".")[0] if "." in substep else substep,
        "crisp_dm_substeps": [substep],
        "requested_outputs": meta.get("requested_outputs", []),
        "completion_criteria": meta.get("completion_criteria", []),
        "constraints": meta.get("constraints", []),
        "substep_name": SUBSTEP_NAMES.get(substep, "?"),
        "case_id": state.case_id,
    }


def format_storyteller_task(
    state: CrispDMState,
    artifact_dir: Path,
    *,
    execution_evidence: dict | None = None,
) -> tuple[str, str]:
    substep = state.substep
    assignment = _assignment_for_substep(substep, state)
    inputs: dict = {"artifact_directory": str(artifact_dir.resolve())}
    if execution_evidence:
        inputs["execution_evidence"] = execution_evidence
    runtime_input = {
        "assignment": assignment,
        "inputs": inputs,
    }
    instruction = (
        "Complete the assigned CRISP-DM substep using the runtime input below. "
        "assignment_id must be exactly the CRISP-DM substep id in assignment "
        f"(currently '{substep}'), not a run-id or agent-qualified composite. "
        "Return exactly one JSON object matching the output schema in your instructions.\n\n"
        f"Runtime input:\n{json.dumps(runtime_input, indent=2, default=str)}"
    )
    if substep == "6.2":
        instruction += (
            " Build story_spec from evaluation_bundle in the state view. "
            "Every interpretation must cite a metric that exists in the bundle."
        )
    return instruction, schema_hint_for_agent("storyteller", substep=substep)
