"""CrewAI task assembly + kickoff: run one agent on one task and capture the output.

The agents themselves (personas, LLM tiering) are defined the idiomatic CrewAI way in
`maads.crew_base` (`@CrewBase MaadsCrew`, driven by `config/agents.yaml`). This module
is the per-call seam: `build_task_description` fetches the substep's agent via
`crew_base.agent_for` and renders the task description from the `config/tasks.yaml`
scaffolds; `_kickoff` wraps that agent+task in a one-agent `Crew` and kicks it off,
folding token usage into `state.token_spend`. The agent *wrappers* in `agents.py` call
`run_json_task(...)` / `run_text_task(...)`; the deterministic orchestrator drives the
cycle.

Model selection — CrewAI best practice: one dedicated ``LLM`` per ``Agent`` at
construction time (see ``agent_for``). Resolution order:

    1. ``MODEL_<AGENT>`` per-role override (e.g. ``MODEL_DEVELOPER``)
    2. ``MODEL_CODE`` / ``OPENAI_MODEL_CODE`` for code-authoring roles
    3. ``MODEL_JSON`` for structured-JSON roles (Ollama path)
    4. ``MODEL`` default (Ollama) or OpenAI tiering (``agents.yaml`` tier field)
"""
from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Task

from maads.crew_base import agent_for, build_llm, reset_llm_caches, resolve_model_for_agent
from maads.prompt_context import compile_task_payload
from maads.prompts import (
    AGENT_TASK_TEMPLATES,
    JSON_EXPECTED_OUTPUT,
)
from maads.state import CrispDMState
from maads.token_budget import (
    TokenBudgetExceeded,
    assert_can_spend,
    check_after_spend,
    repairs_allowed,
)

__all__ = [
    "build_llm",
    "make_agent",
    "resolve_model_for_agent",
    "reset_llm_caches",
    "build_task_description",
    "pop_last_json_task_meta",
    "pop_last_kickoff_output",
    "run_json_task",
    "run_text_task",
    "CrewKickoffError",
]

_last_crew_output: ContextVar[str | None] = ContextVar("_last_crew_output", default=None)
_last_crew_tokens: ContextVar[int | None] = ContextVar("_last_crew_tokens", default=None)
_last_json_task_meta: ContextVar[dict[str, Any] | None] = ContextVar(
    "_last_json_task_meta", default=None,
)


def pop_last_kickoff_output() -> tuple[str | None, int | None]:
    """Return (raw_output, total_tokens) from the most recent kickoff in this context."""
    raw = _last_crew_output.get()
    tokens = _last_crew_tokens.get()
    _last_crew_output.set(None)
    _last_crew_tokens.set(None)
    return raw, tokens


def pop_last_json_task_meta() -> dict[str, Any] | None:
    """Return validation/repair metadata from the most recent ``run_json_task``."""
    meta = _last_json_task_meta.get()
    _last_json_task_meta.set(None)
    return meta


class CrewKickoffError(RuntimeError):
    """CrewAI kickoff failed (LLM timeout, provider error, etc.)."""


def make_agent(agent_name: str, dataset_name: str = "") -> Agent:
    """Back-compat shim — agents are now built by ``maads.crew_base.agent_for``."""
    return agent_for(agent_name, dataset_name)


def _strip_markdown_wrappers(text: str) -> str:
    """Remove common LLM wrappers: fences, horizontal rules, leading/trailing noise."""
    text = text.strip()
    fence = re.match(
        r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
        text,
        re.DOTALL,
    )
    if fence:
        text = fence.group(1).strip()
    else:
        text = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    lines = [
        ln
        for ln in text.splitlines()
        if not re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", ln)
    ]
    return "\n".join(lines).strip()


def _find_balanced_json(text: str) -> str | None:
    """Extract the first top-level ``{...}`` object with balanced braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_json(text: str) -> str:
    """Fix common LLM JSON mistakes (trailing commas, line comments)."""
    without_comments = re.sub(r"//[^\n\"]*", "", text)
    return re.sub(r",(\s*[}\]])", r"\1", without_comments)


def _try_parse_json(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json(text: str) -> dict | None:
    """Parse JSON, cleaning up common LLM formatting before retrying.

    Reasoning models (e.g. minimax-m3) prepend a ``<think>...</think>`` block whose
    prose contains braces, which would otherwise derail the balanced-object search
    below. Strip such blocks first so only the real answer is parsed.
    """
    if not text or not str(text).strip():
        return None

    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", str(text),
                  flags=re.DOTALL | re.IGNORECASE).strip()
    if not text:
        return None

    stripped = _strip_markdown_wrappers(str(text))
    candidates: list[str] = []
    for candidate in (str(text).strip(), stripped):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        for body in (candidate, _repair_json(candidate)):
            parsed = _try_parse_json(body)
            if parsed is not None:
                return parsed

    for candidate in candidates:
        fragment = _find_balanced_json(candidate)
        if not fragment:
            continue
        for body in (fragment, _repair_json(fragment)):
            parsed = _try_parse_json(body)
            if parsed is not None:
                return parsed

    return None


def build_task_description(
    agent_name: str,
    instruction: str,
    state: CrispDMState,
    schema_hint: str = "",
    *,
    json_enforced: bool = True,
) -> tuple[str, str, Agent]:
    """Assemble the CrewAI Task description; returns (description, state_view_json, agent)."""
    view = state.view_for(agent_name)
    dataset_name = state.case_id if agent_name == "domain" else ""
    agent = agent_for(agent_name, dataset_name, json_enforced=json_enforced)
    state_view = json.dumps(view, default=str, ensure_ascii=False)
    if not json_enforced:
        template_kind = "authored_code"
    else:
        template_kind = AGENT_TASK_TEMPLATES.get(agent_name, "substep_json")
    description = compile_task_payload(
        agent_name=agent_name,
        instruction=instruction,
        state_view=view,
        schema_hint=schema_hint,
        template_kind=template_kind,
        substep=state.substep,
    )
    return description, state_view, agent


def _resolve_llm_provider(agent_name: str) -> str:
    """Infer provider label for token accounting."""
    model = resolve_model_for_agent(agent_name)
    if model.startswith("ollama/"):
        return "ollama"
    if "deepseek" in model.lower():
        return "deepseek"
    if os.getenv("OPENAI_API_KEY") or model.startswith("gpt-"):
        return "openai"
    return "other"


def _kickoff(
    agent_name: str,
    instruction: str,
    state: CrispDMState,
    schema_hint: str,
    expected_output: str,
    *,
    json_enforced: bool = True,
) -> str:
    """Run one CrewAI task and return its raw string output.

    Sends only `state.view_for(agent_name)` as context (token discipline) and
    folds the reported token usage into `state.token_spend`.

    Raises:
        CrewKickoffError: when CrewAI kickoff fails.
        TokenBudgetExceeded: when MAX_TOKENS_PER_RUN or per-agent cap is exceeded.
    """
    assert_can_spend(state, agent_name)
    description, _state_view, agent = build_task_description(
        agent_name, instruction, state, schema_hint, json_enforced=json_enforced,
    )
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    try:
        output = crew.kickoff()
    except Exception as exc:
        raise CrewKickoffError(f"CrewAI kickoff failed for {agent_name}: {exc}") from exc

    raw_output = str(output)
    _last_crew_output.set(raw_output)
    total_tokens = None
    n_input = n_output = None
    try:
        total_tokens = int(output.token_usage.total_tokens)
        n_input = int(output.token_usage.prompt_tokens)
        n_output = int(output.token_usage.completion_tokens)
    except (AttributeError, TypeError, ValueError):
        pass
    _last_crew_tokens.set(total_tokens)

    # Token accounting — record CrewAI usage on shared state.
    try:
        if total_tokens is not None:
            provider = _resolve_llm_provider(agent_name)
            state.add_tokens(agent_name, total_tokens, provider=provider, n_input=n_input, n_output=n_output)
    except (AttributeError, TypeError, ValueError):
        pass
    check_after_spend(state, agent_name)
    return raw_output


def run_json_task(
    agent_name: str,
    instruction: str,
    state: CrispDMState,
    schema_hint: str = "",
    *,
    artifact_dir: Path | None = None,
) -> dict | None:
    """Run one CrewAI task for `agent_name` and return schema-valid JSON (or None).

    Raises:
        CrewKickoffError: when kickoff fails or the output is not JSON/schema-valid.
        TokenBudgetExceeded: when MAX_TOKENS_PER_RUN or per-agent cap is exceeded.
    """
    from maads.output_contracts import validate_agent_output

    _last_json_task_meta.set({
        "json_valid": False,
        "schema_ok": False,
        "schema_errors": [],
        "repair": {"kind": "none", "requesting_agent": agent_name, "succeeded": True},
    })

    raw_output = _kickoff(
        agent_name, instruction, state, schema_hint,
        expected_output=JSON_EXPECTED_OUTPUT,
        json_enforced=True,
    )
    parsed = _extract_json(raw_output)
    json_valid = parsed is not None

    def _set_meta(
        *,
        payload: dict | None,
        repair_kind: str = "none",
        repair_succeeded: bool = True,
    ) -> list[str]:
        errors = (
            validate_agent_output(agent_name, payload, substep=state.substep)
            if payload is not None
            else ["output is not a JSON object"]
        )
        _last_json_task_meta.set({
            "json_valid": payload is not None,
            "schema_ok": not errors,
            "schema_errors": errors,
            "repair": {
                "kind": repair_kind,
                "requesting_agent": agent_name,
                "succeeded": repair_succeeded and not errors,
            },
        })
        return errors

    if parsed is not None:
        errors = _set_meta(payload=parsed)
        if not errors:
            return parsed
        if artifact_dir is not None and repairs_allowed(state):
            from maads.debug import debug_json_parse

            outcome = debug_json_parse(
                state=state,
                artifact_dir=artifact_dir,
                requesting_agent=agent_name,
                raw_text=raw_output,
                schema_hint=schema_hint,
                instruction=instruction,
                failure_kind="json_schema",
                invalid_payload=parsed,
            )
            if outcome.status == "FIXED" and outcome.payload is not None:
                repair_errors = _set_meta(
                    payload=outcome.payload,
                    repair_kind=outcome.repair_kind,
                    repair_succeeded=True,
                )
                if not repair_errors:
                    return outcome.payload
        elif artifact_dir is not None:
            raise CrewKickoffError(
                f"CrewAI returned schema-invalid JSON for {agent_name} "
                f"at substep {state.substep}; DEBUG skipped (token budget soft limit or cap)"
            )
        schema_errors = _last_json_task_meta.get() or {}
        detail = "; ".join(schema_errors.get("schema_errors", errors)[:5])
        raise CrewKickoffError(
            f"CrewAI returned schema-invalid JSON for {agent_name} "
            f"at substep {state.substep}: {detail}"
        )

    if raw_output.strip():
        if artifact_dir is not None and repairs_allowed(state):
            from maads.debug import debug_json_parse

            outcome = debug_json_parse(
                state=state,
                artifact_dir=artifact_dir,
                requesting_agent=agent_name,
                raw_text=raw_output,
                schema_hint=schema_hint,
                instruction=instruction,
                failure_kind="json_parse",
            )
            if outcome.status == "FIXED" and outcome.payload is not None:
                repair_errors = _set_meta(
                    payload=outcome.payload,
                    repair_kind=outcome.repair_kind,
                    repair_succeeded=True,
                )
                if not repair_errors:
                    return outcome.payload
        elif artifact_dir is not None:
            raise CrewKickoffError(
                f"CrewAI returned non-JSON output for {agent_name} at substep "
                f"{state.substep}; DEBUG skipped (token budget soft limit or cap)"
            )
        _set_meta(payload=None, repair_succeeded=False)
        raise CrewKickoffError(
            f"CrewAI returned non-JSON output for {agent_name} at substep {state.substep}"
        )

    _set_meta(payload=None, repair_succeeded=False)
    return parsed


def run_text_task(
    agent_name: str,
    instruction: str,
    state: CrispDMState,
    expected_output: str = "The requested output.",
) -> str:
    """Run one CrewAI task and return the raw text output (no JSON parsing).

    Used for code authoring, where demanding JSON-wrapped code would force the
    model to escape quotes/newlines inside a string — exactly what small models
    get wrong. Instead the agent returns a fenced code block we extract verbatim.

    Raises:
        CrewKickoffError: when CrewAI kickoff fails.
        TokenBudgetExceeded: when MAX_TOKENS_PER_RUN or per-agent cap is exceeded.
    """
    return _kickoff(
        agent_name, instruction, state, "", expected_output, json_enforced=False,
    )
