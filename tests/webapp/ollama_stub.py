"""Shared Ollama Cloud /api/tags stub for webapp tests (no live network)."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import EmailMessage

DEFAULT_OLLAMA_CLOUD_NAMES = ("gpt-oss:120b", "gpt-oss:20b")


class FakeOllamaTags:
    """Stand-in for urllib.request.urlopen against ollama.com/api/tags."""

    models: list[dict[str, str]] = [{"name": name} for name in DEFAULT_OLLAMA_CLOUD_NAMES]
    extra_models: list[dict[str, str]] = []
    error: BaseException | None = None
    status_code: int | None = None
    last_authorization: str | None = None
    last_url: str | None = None

    @classmethod
    def reset(cls) -> None:
        cls.models = [{"name": name} for name in DEFAULT_OLLAMA_CLOUD_NAMES]
        cls.extra_models = [{"name": "nomic-embed-text"}]
        cls.error = None
        cls.status_code = None
        cls.last_authorization = None
        cls.last_url = None

    @classmethod
    def urlopen(cls, req, timeout=None):  # noqa: ARG003 — match urllib signature
        cls.last_url = getattr(req, "full_url", None) or str(req)
        if hasattr(req, "get_header"):
            cls.last_authorization = req.get_header("Authorization")
        else:
            headers = getattr(req, "headers", {})
            cls.last_authorization = headers.get("Authorization") or headers.get("authorization")
        if cls.error is not None:
            raise cls.error
        if cls.status_code in (401, 403, 500):
            raise urllib.error.HTTPError(
                cls.last_url,
                cls.status_code,
                "error",
                EmailMessage(),
                io.BytesIO(b""),
            )
        payload = json.dumps({"models": list(cls.models) + list(cls.extra_models)}).encode("utf-8")
        return io.BytesIO(payload)
