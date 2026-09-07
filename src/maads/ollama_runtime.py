"""Resolve Ollama host and auth for local vs cloud.

When ``OLLAMA_API_KEY`` is set and ``OLLAMA_BASE_URL`` is unset, clients talk
to ``https://ollama.com`` instead of localhost. Callers must not ``ollama pull``
against the cloud host.
"""
from __future__ import annotations

import os
from typing import Any

OLLAMA_CLOUD_HOST = "https://ollama.com"
OLLAMA_LOCAL_HOST = "http://localhost:11434"


def ollama_api_key() -> str | None:
    key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    return key or None


def ollama_base_url() -> str:
    explicit = (os.getenv("OLLAMA_BASE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    if ollama_api_key():
        return OLLAMA_CLOUD_HOST
    return OLLAMA_LOCAL_HOST


def is_ollama_cloud_host(host: str | None = None) -> bool:
    url = (host or ollama_base_url()).rstrip("/")
    return url == OLLAMA_CLOUD_HOST or url.startswith("https://ollama.com")


def ollama_auth_headers() -> dict[str, str]:
    key = ollama_api_key()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def ollama_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"host": ollama_base_url()}
    headers = ollama_auth_headers()
    if headers:
        kwargs["headers"] = headers
    return kwargs
