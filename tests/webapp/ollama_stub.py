"""Shared Ollama Cloud /api/tags and /api/chat stub for webapp tests (no live network)."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import EmailMessage

DEFAULT_OLLAMA_CLOUD_NAMES = ("gpt-oss:120b", "gpt-oss:20b")


class FakeOllamaTags:
    """Stand-in for urllib.request.urlopen against ollama.com."""

    models: list[dict[str, str]] = [{"name": name} for name in DEFAULT_OLLAMA_CLOUD_NAMES]
    extra_models: list[dict[str, str]] = []
    error: BaseException | None = None
    status_code: int | None = None
    chat_denied: dict[str, int] = {}
    chat_error: BaseException | None = None
    last_authorization: str | None = None
    last_url: str | None = None
    last_chat_models: list[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.models = [{"name": name} for name in DEFAULT_OLLAMA_CLOUD_NAMES]
        cls.extra_models = [{"name": "nomic-embed-text"}]
        cls.error = None
        cls.status_code = None
        cls.chat_denied = {}
        cls.chat_error = None
        cls.last_authorization = None
        cls.last_url = None
        cls.last_chat_models = []

    @classmethod
    def _record_request(cls, req) -> str:
        url = getattr(req, "full_url", None) or str(req)
        cls.last_url = url
        if hasattr(req, "get_header"):
            cls.last_authorization = req.get_header("Authorization")
        else:
            headers = getattr(req, "headers", {})
            cls.last_authorization = headers.get("Authorization") or headers.get("authorization")
        return url

    @classmethod
    def _http_error(cls, url: str, status_code: int) -> None:
        raise urllib.error.HTTPError(
            url,
            status_code,
            "error",
            EmailMessage(),
            io.BytesIO(b""),
        )

    @classmethod
    def _open_chat(cls, req, url: str):
        if cls.chat_error is not None:
            raise cls.chat_error
        raw = getattr(req, "data", None) or b""
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        model = payload.get("model") if isinstance(payload, dict) else None
        name = model if isinstance(model, str) else ""
        cls.last_chat_models.append(name)
        denied = cls.chat_denied.get(name)
        if denied is not None:
            cls._http_error(url, denied)
        return io.BytesIO(b'{"message":{"content":"."},"done":true}')

    @classmethod
    def urlopen(cls, req, timeout=None):  # noqa: ARG003 — match urllib signature
        url = cls._record_request(req)
        if "/api/chat" in url:
            return cls._open_chat(req, url)
        if cls.error is not None:
            raise cls.error
        if cls.status_code in (401, 403, 500):
            cls._http_error(url, cls.status_code)
        payload = json.dumps({"models": list(cls.models) + list(cls.extra_models)}).encode("utf-8")
        return io.BytesIO(payload)
