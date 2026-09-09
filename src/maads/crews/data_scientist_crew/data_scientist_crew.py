"""Data scientist crew — exploration, modeling, and evaluation substeps."""
from __future__ import annotations

from pathlib import Path

from maads.crews.kickoff import kickoff_json
from maads.prompts.identities.data_scientist import format_data_scientist_task
from maads.state import CrispDMState

_OWNED = frozenset({"2.3", "4.1", "4.2", "4.3", "4.4", "5.1"})


class DataScientistCrew:
    def kickoff_substep(
        self,
        substep: str,
        state: CrispDMState,
        artifact_dir: Path,
        *,
        execution_evidence: dict | None = None,
    ) -> dict | None:
        if substep not in _OWNED:
            raise ValueError(f"data scientist crew does not own substep {substep}")
        instruction, schema_hint = format_data_scientist_task(
            state, artifact_dir, execution_evidence=execution_evidence,
        )
        return kickoff_json(
            "data_scientist",
            instruction,
            state,
            schema_hint=schema_hint,
            artifact_dir=artifact_dir,
        )
