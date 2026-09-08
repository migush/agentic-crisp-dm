"""Per-agent LLM wire-params (reasoning_effort) resolved before kickoff."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

AGENT_NAMES = (
    "pm",
    "domain",
    "data_engineer",
    "data_scientist",
    "developer",
    "storyteller",
)

# Bootstrap seed defaults (write-only ledger will inform future revisions).
_ROLE_DEFAULT_EFFORT: dict[str, str] = {
    "pm": "medium",
    "domain": "medium",
    "storyteller": "medium",
    "data_engineer": "high",
    "data_scientist": "high",
    "developer": "high",
}

_ASTRA_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})
_GPT56_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
_GENERIC_REASONING_EFFORTS = frozenset({"low", "medium", "high"})

_crewai_wire_patch_installed = False


def _resolved_effort(llm: Any) -> str | None:
    effort = getattr(llm, "reasoning_effort", None)
    if effort and effort != "none":
        return str(effort)
    extra = getattr(llm, "additional_params", None) or {}
    effort = extra.get("reasoning_effort")
    if effort and effort != "none":
        return str(effort)
    return None


def install_crewai_reasoning_effort_wire_patch() -> None:
    """Normalize ``reasoning_effort`` for CrewAI OpenAI Completions *and* Responses.

    Completions: CrewAI only forwards ``reasoning_effort`` when ``is_o1_model``
    (``"o1" in model``). Omitting it for ``gpt-6-astra`` makes the API default
    to ``none`` → HTTP 400.

    Responses: effort must be ``reasoning={"effort": ...}``. A top-level
    ``reasoning_effort`` kwarg (e.g. leaked via ``additional_params``) makes
    ``Responses.create()`` raise ``unexpected keyword argument``.

    Idempotent. Safe no-op if the CrewAI layout is older/missing.
    """
    global _crewai_wire_patch_installed
    if _crewai_wire_patch_installed:
        return
    try:
        from crewai.llms.providers.openai.completion import OpenAICompletion
    except ImportError:
        return

    if getattr(OpenAICompletion, "_maads_reasoning_effort_patch_v2", False):
        _crewai_wire_patch_installed = True
        return

    original_completions = OpenAICompletion._prepare_completion_params
    original_responses = OpenAICompletion._prepare_responses_params

    def _prepare_completion_params(self, messages, tools=None):  # type: ignore[no-untyped-def]
        params = original_completions(self, messages, tools)
        effort = _resolved_effort(self)
        if effort:
            params["reasoning_effort"] = effort
        elif params.get("reasoning_effort") == "none":
            params.pop("reasoning_effort", None)
        return params

    def _prepare_responses_params(  # type: ignore[no-untyped-def]
        self, messages, tools=None, response_model=None
    ):
        params = original_responses(
            self, messages, tools=tools, response_model=response_model
        )
        # Responses API rejects top-level reasoning_effort=.
        params.pop("reasoning_effort", None)
        effort = _resolved_effort(self)
        if effort:
            reasoning = params.get("reasoning")
            if isinstance(reasoning, dict):
                reasoning = {**reasoning, "effort": effort}
            else:
                reasoning = {"effort": effort}
            params["reasoning"] = reasoning
        else:
            reasoning = params.get("reasoning")
            if isinstance(reasoning, dict) and reasoning.get("effort") == "none":
                params.pop("reasoning", None)
        return params

    OpenAICompletion._prepare_completion_params = _prepare_completion_params  # type: ignore[method-assign]
    OpenAICompletion._prepare_responses_params = _prepare_responses_params  # type: ignore[method-assign]
    OpenAICompletion._maads_reasoning_effort_patch_v2 = True
    _crewai_wire_patch_installed = True


def _model_leaf(model: str) -> str:
    return model.strip().lower().rsplit("/", 1)[-1]


def model_uses_reasoning_effort(model: str) -> bool:
    """True when the provider expects an explicit ``reasoning_effort`` value.

    Ollama and classic chat models (gpt-4o, …) do not. Reasoning-family ids
    (gpt-5*, gpt-6*, o1/o3/o4*, *astra*) do — CrewAI may otherwise send
    ``none``, which some models reject with HTTP 400.
    """
    if model.startswith("ollama/"):
        return False
    leaf = _model_leaf(model)
    if "astra" in leaf:
        return True
    if leaf.startswith(("gpt-5", "gpt-6")):
        return True
    if leaf.startswith(("o1", "o3", "o4")):
        return True
    return False


def allowed_reasoning_efforts(model: str) -> frozenset[str] | None:
    """Allowed effort values for ``model``, or None if the param is unused."""
    if not model_uses_reasoning_effort(model):
        return None
    leaf = _model_leaf(model)
    if "astra" in leaf or leaf.startswith("gpt-6"):
        return _ASTRA_EFFORTS
    if leaf.startswith("gpt-5.6") or leaf.startswith("gpt-5"):
        return _GPT56_EFFORTS
    return _GENERIC_REASONING_EFFORTS


def role_default_effort(agent_name: str) -> str:
    return _ROLE_DEFAULT_EFFORT.get(agent_name, "medium")


def _env_effort(agent_name: str) -> str | None:
    specific = os.getenv(f"REASONING_EFFORT_{agent_name.upper()}")
    if specific is not None and specific.strip():
        return specific.strip().lower()
    global_val = os.getenv("REASONING_EFFORT")
    if global_val is not None and global_val.strip():
        return global_val.strip().lower()
    return None


def clamp_effort(desired: str, allowed: frozenset[str]) -> str:
    """Map ``desired`` onto ``allowed``, never returning ``none`` for agent work."""
    desired = desired.strip().lower()
    usable = allowed - {"none"}
    if not usable:
        raise ValueError(f"no usable reasoning_effort values in {sorted(allowed)}")
    if desired in usable:
        return desired
    for candidate in ("high", "xhigh", "medium", "low"):
        if candidate in usable:
            return candidate
    return sorted(usable)[0]


@dataclass(frozen=True)
class AgentLlmParams:
    agent: str
    model: str
    reasoning_effort: str | None


class LlmParamError(ValueError):
    """Illegal reasoning_effort (or similar) for a resolved model."""


def resolve_agent_llm_params(agent_name: str, model: str | None = None) -> AgentLlmParams:
    """Resolve wire params for one agent; raise :class:`LlmParamError` if illegal."""
    if model is None:
        from maads.crew_base import resolve_model_for_agent

        model_id = resolve_model_for_agent(agent_name)
    else:
        model_id = model
    allowed = allowed_reasoning_efforts(model_id)
    if allowed is None:
        return AgentLlmParams(agent=agent_name, model=model_id, reasoning_effort=None)

    override = _env_effort(agent_name)
    if override is not None:
        if override == "none" or override not in allowed:
            raise LlmParamError(
                f"REASONING_EFFORT for {agent_name}={override!r} is not allowed "
                f"for model {model_id!r}; allowed: {sorted(allowed - {'none'})}"
            )
        effort = override
    else:
        effort = clamp_effort(role_default_effort(agent_name), allowed)

    return AgentLlmParams(agent=agent_name, model=model_id, reasoning_effort=effort)


def resolve_all_agent_llm_params() -> dict[str, AgentLlmParams]:
    """Resolve params for all six agents (models from env / override)."""
    return {name: resolve_agent_llm_params(name) for name in AGENT_NAMES}


def preflight_llm_params() -> list[str]:
    """Return errors for illegal per-agent LLM params; empty = ok to start."""
    install_crewai_reasoning_effort_wire_patch()
    errors: list[str] = []
    for name in AGENT_NAMES:
        try:
            resolve_agent_llm_params(name)
        except LlmParamError as exc:
            errors.append(str(exc))
    return errors
