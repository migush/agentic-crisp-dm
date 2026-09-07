"""Tests for local vs Ollama Cloud host/auth resolution."""
from __future__ import annotations

from maads.ollama_runtime import (
    OLLAMA_CLOUD_HOST,
    OLLAMA_LOCAL_HOST,
    is_ollama_cloud_host,
    ollama_api_key,
    ollama_auth_headers,
    ollama_base_url,
    ollama_client_kwargs,
)


def test_defaults_to_localhost_without_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert ollama_api_key() is None
    assert ollama_base_url() == OLLAMA_LOCAL_HOST
    assert ollama_auth_headers() == {}
    assert ollama_client_kwargs() == {"host": OLLAMA_LOCAL_HOST}
    assert is_ollama_cloud_host() is False


def test_key_without_base_url_defaults_to_ollama_cloud(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert ollama_base_url() == OLLAMA_CLOUD_HOST
    assert ollama_auth_headers() == {"Authorization": "Bearer secret"}
    assert ollama_client_kwargs() == {
        "host": OLLAMA_CLOUD_HOST,
        "headers": {"Authorization": "Bearer secret"},
    }
    assert is_ollama_cloud_host() is True


def test_explicit_base_url_wins_over_cloud_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    assert ollama_base_url() == "http://localhost:11434"
    assert is_ollama_cloud_host() is False
    assert ollama_client_kwargs()["host"] == "http://localhost:11434"
    assert ollama_client_kwargs()["headers"]["Authorization"] == "Bearer secret"
