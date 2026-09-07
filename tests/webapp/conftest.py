"""Webapp tests: allow the published JWT fallback and stub OpenAI Models API."""

from __future__ import annotations

import os

import pytest

from tests.webapp.ollama_stub import FakeOllamaTags
from tests.webapp.openai_stub import DEFAULT_LIVE_MODEL_IDS, FakeOpenAI

# Must be set before webapp.backend.app is imported (create_app asserts this).
os.environ.setdefault("WEBAPP_ALLOW_DEV_SECRET", "1")
os.environ.setdefault("WEBAPP_INSECURE_COOKIES", "1")


@pytest.fixture(autouse=True)
def stub_openai_models(monkeypatch):
    FakeOpenAI.live_ids = list(DEFAULT_LIVE_MODEL_IDS)
    FakeOpenAI.extra_ids = [
        "text-embedding-3-small",
        "whisper-1",
        "tts-1",
        "gpt-image-1",
        "omni-moderation-latest",
        "gpt-4o-realtime-preview",
        "gpt-4o-transcribe",
    ]
    FakeOpenAI.error = None
    FakeOpenAI.last_api_key = None
    monkeypatch.setattr("webapp.backend.openai_models.OpenAI", FakeOpenAI)
    return FakeOpenAI


@pytest.fixture(autouse=True)
def stub_ollama_tags(monkeypatch):
    FakeOllamaTags.reset()
    monkeypatch.setattr("webapp.backend.ollama_models.urllib.request.urlopen", FakeOllamaTags.urlopen)
    return FakeOllamaTags
