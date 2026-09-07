"""Shared OpenAI Models API stub for webapp tests (no live network)."""

from __future__ import annotations

DEFAULT_LIVE_MODEL_IDS = ("gpt-4o", "gpt-5.4-mini", "gpt-5.4")


class FakeModel:
    def __init__(self, id: str, display_name: str | None = None):
        self.id = id
        if display_name is not None:
            self.display_name = display_name


class FakePage:
    def __init__(self, models: list[FakeModel]):
        self.data = models

    def __iter__(self):
        return iter(self.data)


class FakeOpenAI:
    """Stand-in for openai.OpenAI; tests configure .live_ids / .error."""

    live_ids: list[str] = list(DEFAULT_LIVE_MODEL_IDS)
    extra_ids: list[str] = []
    error: BaseException | None = None
    last_api_key: str | None = None

    def __init__(self, api_key: str | None = None, **_kwargs):
        type(self).last_api_key = api_key
        self.models = self

    def list(self):
        if type(self).error is not None:
            raise type(self).error
        ids = list(type(self).live_ids) + list(type(self).extra_ids)
        return FakePage([FakeModel(model_id) for model_id in ids])
