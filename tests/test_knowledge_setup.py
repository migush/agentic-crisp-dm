"""Tests for knowledge / embedder setup."""
from __future__ import annotations

from maads.knowledge_setup import resolve_embedder_config


def test_resolve_embedder_uses_ollama_when_model_is_ollama(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MODEL", "ollama/gemma2:9b")
    cfg = resolve_embedder_config()
    assert cfg is not None
    assert cfg["provider"] == "ollama"
    assert cfg["config"]["model_name"] == "nomic-embed-text"
    assert cfg["config"]["url"].endswith("/api/embeddings")


def test_resolve_embedder_cloud_url_when_ollama_key_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("MODEL", "ollama/gpt-oss:120b")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
    cfg = resolve_embedder_config()
    assert cfg is not None
    assert cfg["config"]["url"] == "https://ollama.com/api/embeddings"


def test_resolve_embedder_defaults_to_openai_when_key_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "ollama/gemma2:9b")
    assert resolve_embedder_config() is None


def test_resolve_embedder_forced_ollama(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDINGS_OLLAMA_MODEL_NAME", "mxbai-embed-large")
    cfg = resolve_embedder_config()
    assert cfg is not None
    assert cfg["config"]["model_name"] == "mxbai-embed-large"
