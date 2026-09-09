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
    assert role_default_effort("developer") == "medium"
    assert role_default_effort("data_engineer") == "medium"
    assert role_default_effort("data_scientist") == "medium"


def test_effort_allowlists_by_model_family() -> None:
    assert "none" not in (allowed_reasoning_efforts("gpt-6-astra") or set())
    gpt56 = allowed_reasoning_efforts("gpt-5.6-sol")
    assert gpt56 is not None and "none" in gpt56
    gpt55 = allowed_reasoning_efforts("gpt-5.5-pro")
    assert gpt55 is not None and "medium" in gpt55
    assert allowed_reasoning_efforts("ollama/gemma2:9b") is None


def test_resolve_astra_sets_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "gpt-6-astra")
    pm = resolve_agent_llm_params("pm")
    dev = resolve_agent_llm_params("developer")
    assert pm.reasoning_effort == "medium"
    assert dev.reasoning_effort == "medium"


def test_build_llm_sets_effort_for_astra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "gpt-6-astra")
    reset_llm_caches()
    llm = build_llm("pm")
    assert getattr(llm, "reasoning_effort", None) == "medium"
    # Must NOT leak into additional_params — that breaks Responses.create().
    assert "reasoning_effort" not in (getattr(llm, "additional_params", None) or {})


def test_build_llm_puts_effort_on_wire_for_non_o1(monkeypatch: pytest.MonkeyPatch) -> None:
    """CrewAI only auto-forwards reasoning_effort for o1*; astra must still send it."""
    monkeypatch.setenv("MODEL", "gpt-6-astra")
    reset_llm_caches()
    llm = build_llm("pm")
    assert getattr(llm, "is_o1_model", False) is False
    prepare = getattr(llm, "_prepare_completion_params", None)
    assert prepare is not None, "expected OpenAICompletion._prepare_completion_params"
    params = prepare([{"role": "user", "content": "ping"}])
    assert params.get("reasoning_effort") == "medium"
    assert params.get("reasoning_effort") != "none"


def test_wire_patch_forwards_attribute_without_additional_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if additional_params is stripped, the class patch still sends effort."""
    from crewai import LLM

    from maads.llm_params import install_crewai_reasoning_effort_wire_patch

    install_crewai_reasoning_effort_wire_patch()
    llm = LLM(model="gpt-6-astra", reasoning_effort="high")
    llm.additional_params = {}
    assert llm.is_o1_model is False
    params = llm._prepare_completion_params([{"role": "user", "content": "ping"}])
    assert params.get("reasoning_effort") == "high"


def test_responses_api_uses_reasoning_effort_dict_not_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpt-5.5-pro Responses path must not pass reasoning_effort= to create()."""
    from crewai import LLM

    from maads.llm_params import install_crewai_reasoning_effort_wire_patch

    install_crewai_reasoning_effort_wire_patch()
    llm = LLM(
        model="gpt-5.5-pro",
        api="responses",
        reasoning_effort="medium",
        # Simulate the PR #13 leak that caused TypeError on Responses.create.
        additional_params={"reasoning_effort": "medium"},
    )
    params = llm._prepare_responses_params([{"role": "user", "content": "ping"}])
    assert "reasoning_effort" not in params
    assert params.get("reasoning") == {"effort": "medium"}


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
