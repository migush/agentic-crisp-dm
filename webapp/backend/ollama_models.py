"""Live Ollama Cloud model listing for the hosted account product.

The browser cannot call ollama.com (no CORS), so the webapp proxies
``GET /api/tags`` with the just-decrypted key — the same trust boundary as
``POST /api/tasks``. The key is never logged or persisted.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from fastapi import HTTPException

OLLAMA_CLOUD_TAGS_URL = "https://ollama.com/api/tags"

# Capability filter: keep chat/LLM ids the pipeline can run, drop embeddings.
_DROP_SUBSTRINGS = ("embed",)


def is_chat_llm_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(token in lowered for token in _DROP_SUBSTRINGS)


def _prefixed_id(name: str) -> str:
    if name.startswith("ollama/"):
        return name
    return f"ollama/{name}"


def list_ollama_cloud_chat_models(api_key: str) -> list[dict[str, str]]:
    """Return ``{id, label}`` chat models visible to ``api_key``.

    Ollama 401/403 become HTTP 400 (invalid API key). Network failures become
    HTTP 502. Ids are prefixed with ``ollama/`` so ``crew_base.build_llm``
    routes correctly. Does not fall back to the static ``OLLAMA_CLOUD_MODELS``
    catalog.
    """
    request = urllib.request.Request(
        OLLAMA_CLOUD_TAGS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HTTPException(status_code=400, detail="invalid API key") from None
        raise HTTPException(status_code=502, detail="Ollama models request failed.") from None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=502, detail="Failed to reach Ollama.") from None

    entries = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise HTTPException(status_code=502, detail="Ollama models request failed.") from None

    models: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            raw_name = entry.get("name") or entry.get("model")
            name = raw_name if isinstance(raw_name, str) else ""
        else:
            continue
        name = name.strip()
        if not name or name in seen or not is_chat_llm_id(name):
            continue
        seen.add(name)
        models.append({"id": _prefixed_id(name), "label": name.removeprefix("ollama/")})
    models.sort(key=lambda entry: entry["id"])
    return models
