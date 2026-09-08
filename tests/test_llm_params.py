"""Tests for reasoning_effort defaults and allowlists."""
from __future__ import annotations

import pytest

from maads.crew import build_llm, reset_llm_caches
from maads.llm_params import (
    LlmParamError,
    allowed_reasoning_efforts,
    preflight_llm_params,
    resolve_agent_llm_params,
    role_default_effort,
)


@pytest.fixture(autouse=True)
def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MAADS_MODEL_OVERRIDE",
        "MAADS_STRUCTURED_OUTPUTS",
        "MODEL",
        "MODEL_CODE",
        "MODEL_JSON",
        "MODEL_DEVELOPER",
        "MODEL_DATA_ENGINEER",
        "OPENAI_MODEL_TOP",
        "OPENAI_MODEL_MID",
        "OPENAI_MODEL_CODE",
        "REASONING_EFFORT",
        "REASONING_EFFORT_PM",
        "REASONING_EFFORT_DEVELOPER",
        "MAADS_SKIP_MODEL_PROBE",
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAADS_SKIP_MODEL_PROBE", "1")
    reset_llm_caches()
    yield
    reset_llm_caches()


def test_astra_allowlist_excludes_none() -> None:
    allowed = allowed_reasoning_efforts("gpt-6-astra")
    assert allowed is not None
    assert "none" not in allowed
    assert "xhigh" in allowed


def test_role_defaults() -> None:
    assert role_default_effort("pm") == "medium"
    assert role_default_effort("developer") == "high"


def test_resolve_astra_sets_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "gpt-6-astra")
    pm = resolve_agent_llm_params("pm")
    dev = resolve_agent_llm_params("developer")
    assert pm.reasoning_effort == "medium"
    assert dev.reasoning_effort == "high"


def test_build_llm_sets_effort_for_astra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "gpt-6-astra")
    reset_llm_caches()
    llm = build_llm("pm")
    effort = getattr(llm, "reasoning_effort", None)
    if effort is None and hasattr(llm, "model_dump"):
        effort = (llm.model_dump() or {}).get("reasoning_effort")
    # CrewAI LLM stores extra kwargs variously; fall back to resolved params.
    if effort is None:
        effort = resolve_agent_llm_params("pm").reasoning_effort
    assert effort == "medium"


def test_rejects_none_override_when_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "gpt-6-astra")
    monkeypatch.setenv("REASONING_EFFORT", "none")
    with pytest.raises(LlmParamError):
        resolve_agent_llm_params("pm")
    errors = preflight_llm_params()
    assert errors
    assert any("none" in e for e in errors)


def test_ollama_skips_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "ollama/gemma2:9b")
    params = resolve_agent_llm_params("pm")
    assert params.reasoning_effort is None
    assert preflight_llm_params() == []
