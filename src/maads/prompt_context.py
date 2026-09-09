"""Normalize and consolidate prompt context before agent payload compilation."""
from __future__ import annotations

import json
from typing import Any

from maads.state import SUBSTEP_NAMES

# Re-export normalization helpers for callers that imported them here.
from maads.text_normalize import (  # noqa: F401
    dedupe_nested_dict,
    dedupe_passages,
    normalize_passage_text,
    strip_markdown_headers,
)


def compile_task_payload(
    *,
    agent_name: str,
    instruction: str,
    state_view: dict[str, Any],
    schema_hint: str = "",
    template_kind: str = "substep_json",
    substep: str = "",
) -> str:
    """Assemble the final CrewAI task description with a single state_view block."""
    from maads.prompts.loader import task_scaffold

    state_view_json = json.dumps(state_view, default=str, ensure_ascii=False)
    if template_kind == "state_only":
        return task_scaffold("state_only")["description"].format(
            state_view=state_view_json,
            instruction=instruction,
        )
    if template_kind == "authored_code":
        return task_scaffold("authored_code")["description"].format(
            instruction=instruction,
            state_view=state_view_json,
        )
    return task_scaffold("substep_json")["description"].format(
        substep=substep,
        substep_name=SUBSTEP_NAMES.get(substep, "?"),
        instruction=instruction,
        state_view=state_view_json,
        schema_hint=schema_hint,
    )
