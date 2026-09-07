"""Tests for plain LLM config on codegen text tasks."""
from __future__ import annotations

import pytest

from maads.crew_base import build_llm, reset_llm_caches


@pytest.fixture(autouse=True)
def _isolate_llm_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MAADS_MODEL_OVERRIDE",
        "MODEL_CODE",
        "MODEL_JSON",
        "MODEL_PM",
        "MODEL_DEVELOPER",
        "OPENAI_MODEL_CODE",
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAADS_SKIP_MODEL_PROBE", "1")
    monkeypatch.setenv("MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    reset_llm_caches()
    yield
    reset_llm_caches()


def test_build_llm_plain_text_omits_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "maads.crew_base._json_response_format_for_agent",
        lambda agent_name, model: {"type": "json_object"},
    )
    llm = build_llm("data_scientist", json_enforced=False)
    assert getattr(llm, "response_format", None) is None

    llm_json = build_llm("data_scientist", json_enforced=True)
    assert getattr(llm_json, "response_format", None) == {"type": "json_object"}


def test_build_llm_ollama_cloud_passes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "ollama/gpt-oss:120b")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr("maads.crew_base.LLM", FakeLLM)
    reset_llm_caches()
    llm = build_llm("pm")
    assert llm.model == "ollama/gpt-oss:120b"
    assert captured["base_url"] == "https://ollama.com"
    assert captured["api_key"] == "ollama-secret"


def test_build_llm_ollama_local_omits_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL", "ollama/gemma2:9b")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr("maads.crew_base.LLM", FakeLLM)
    reset_llm_caches()
    build_llm("pm")
    assert captured["base_url"] == "http://localhost:11434"
    assert "api_key" not in captured
