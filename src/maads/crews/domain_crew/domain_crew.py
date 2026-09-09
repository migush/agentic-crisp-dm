"""Domain expert crew — Business Understanding substeps 1.1–1.3."""
from __future__ import annotations

from pathlib import Path

from maads.crews.kickoff import kickoff_json
from maads.prompts.identities.domain import (
    format_domain_refine_goals_task,
    format_domain_situation_task,
    format_domain_understanding_task,
)
from maads.state import CrispDMState

_SUBSTEP_TASKS = {
    "1.1": format_domain_understanding_task,
    "1.2": format_domain_situation_task,
    "1.3": format_domain_refine_goals_task,
}


class DomainCrew:
    """Domain knowledge crew for phase 1 substeps."""

    def kickoff_substep(
        self,
        substep: str,
        state: CrispDMState,
        artifact_dir: Path,
    ) -> dict | None:
        """Run one domain substep via the shared JSON kickoff seam."""
        formatter = _SUBSTEP_TASKS.get(substep)
        if formatter is None:
            raise ValueError(f"domain crew does not own substep {substep}")
        instruction, schema_hint = formatter(state)
        return kickoff_json(
            "domain",
            instruction,
            state,
            schema_hint=schema_hint,
            artifact_dir=artifact_dir,
        )
