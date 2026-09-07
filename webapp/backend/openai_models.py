"""Live OpenAI model listing for the hosted account product.

The browser cannot call api.openai.com (no CORS), so the webapp proxies
``models.list()`` with the just-decrypted key — the same trust boundary as
``POST /api/tasks``. The key is never logged or persisted.
"""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
)
from fastapi import HTTPException

# Capability filter, not a curated catalog: keep chat/LLM ids the pipeline can
# run, drop embeddings / audio / image / moderation / realtime / transcribe.
_KEEP_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_DROP_SUBSTRINGS = ("embedding", "whisper", "tts", "image", "moderation", "realtime", "transcribe")


def is_chat_llm_id(model_id: str) -> bool:
    lowered = model_id.lower()
    if any(token in lowered for token in _DROP_SUBSTRINGS):
        return False
    return any(lowered.startswith(prefix) for prefix in _KEEP_PREFIXES)


def _model_label(model: object) -> str:
    display = getattr(model, "display_name", None)
    if isinstance(display, str) and display.strip():
        return display
    return str(getattr(model, "id", ""))


def list_openai_chat_models(api_key: str) -> list[dict[str, str]]:
    """Return ``{id, label}`` chat models visible to ``api_key``.

    OpenAI 401/403 become HTTP 400 (invalid API key). Network failures become
    HTTP 502. Does not fall back to the static ``OPENAI_MODELS`` catalog.
    """
    try:
        page = OpenAI(api_key=api_key).models.list()
    except (AuthenticationError, PermissionDeniedError):
        raise HTTPException(status_code=400, detail="invalid API key") from None
    except (APIConnectionError, APITimeoutError):
        raise HTTPException(status_code=502, detail="Failed to reach OpenAI.") from None
    except APIStatusError as exc:
        if getattr(exc, "status_code", None) in (401, 403):
            raise HTTPException(status_code=400, detail="invalid API key") from None
        raise HTTPException(status_code=502, detail="OpenAI models request failed.") from None

    models: list[dict[str, str]] = []
    seen: set[str] = set()
    for model in page:
        model_id = getattr(model, "id", None)
        if not isinstance(model_id, str) or model_id in seen or not is_chat_llm_id(model_id):
            continue
        seen.add(model_id)
        models.append({"id": model_id, "label": _model_label(model)})
    models.sort(key=lambda entry: entry["id"])
    return models
