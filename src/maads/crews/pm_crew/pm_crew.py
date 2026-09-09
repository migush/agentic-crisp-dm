"""PM crew — checkpoint decisions and phase-5 substeps."""
from __future__ import annotations

from pathlib import Path

from maads.crews.kickoff import kickoff_json
from maads.prompts import (
    PM_DECISION_INSTRUCTION,
    PM_NEXT_STEPS_INSTRUCTION,
    PM_REVIEW_INSTRUCTION,
)
from maads.state import CrispDMState


class PMCrew:
    """Project manager crew for decisions and PM-owned substeps."""

    def kickoff_decision(
        self,
        state: CrispDMState,
        artifact_dir: Path,
    ) -> dict | None:
        return kickoff_json(
            "pm",
            PM_DECISION_INSTRUCTION,
            state,
            artifact_dir=artifact_dir,
        )

    def kickoff_substep(
        self,
        substep: str,
        state: CrispDMState,
        artifact_dir: Path,
    ) -> dict | None:
        if substep == "5.2":
            return kickoff_json(
                "pm",
                PM_REVIEW_INSTRUCTION,
                state,
                artifact_dir=artifact_dir,
            )
        if substep == "5.3":
            return kickoff_json(
                "pm",
                PM_NEXT_STEPS_INSTRUCTION,
                state,
                artifact_dir=artifact_dir,
            )
        raise ValueError(f"pm crew does not own substep {substep}")
