"""Storyteller crew — report evidence and storytelling substeps."""
from __future__ import annotations

from pathlib import Path

from maads.crews.kickoff import kickoff_json
from maads.prompts.identities.storyteller import format_storyteller_task
from maads.state import CrispDMState

_OWNED = frozenset({"6.2", "6.3"})


class StorytellerCrew:
    def kickoff_substep(
        self,
        substep: str,
        state: CrispDMState,
        artifact_dir: Path,
        *,
        execution_evidence: dict | None = None,
    ) -> dict | None:
        if substep not in _OWNED:
            raise ValueError(f"storyteller crew does not own substep {substep}")
        instruction, schema_hint = format_storyteller_task(
            state, artifact_dir, execution_evidence=execution_evidence,
        )
        return kickoff_json(
            "storyteller",
            instruction,
            state,
            schema_hint=schema_hint,
            artifact_dir=artifact_dir,
        )
