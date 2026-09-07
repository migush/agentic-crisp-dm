"""Live Ollama Cloud model listing for the hosted account product.

The browser cannot call ollama.com (no CORS), so the webapp proxies
``GET /api/tags`` with the just-decrypted key — the same trust boundary as
``POST /api/tasks``. ``/api/tags`` lists every cloud tag, including models
that 402/403 for the current plan, so each chat id is then probed with a
1-token ``POST /api/chat``. The key is never logged or persisted.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException

OLLAMA_CLOUD_TAGS_URL = "https://ollama.com/api/tags"
OLLAMA_CLOUD_CHAT_URL = "https://ollama.com/api/chat"

# Capability filter: keep chat/LLM ids the pipeline can run, drop embeddings.
_DROP_SUBSTRINGS = ("embed",)
_PLAN_DENIED_CODES = frozenset({402, 403})
_PROBE_TIMEOUT_S = 8
_PROBE_WORKERS = 4


def is_chat_llm_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(token in lowered for token in _DROP_SUBSTRINGS)


def _prefixed_id(name: str) -> str:
    if name.startswith("ollama/"):
        return name
    return f"ollama/{name}"


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _chat_access_ok(api_key: str, name: str) -> bool:
    """Return False only when Ollama says this key cannot run ``name``.

    402 (subscription / extra usage) and 403 (plan denied) are dropped.
    Timeouts, 429, and 5xx fail open so a flaky probe does not empty the
    Tasks dropdown.
    """
    body = json.dumps(
        {
            "model": name,
            "messages": [{"role": "user", "content": "."}],
            "stream": False,
            "think": False,
            "keep_alive": 0,
            "options": {"num_predict": 1},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_CLOUD_CHAT_URL,
        data=body,
        headers={**_auth_headers(api_key), "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_S) as response:
            response.read()
        return True
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return exc.code not in _PLAN_DENIED_CODES
    except (urllib.error.URLError, TimeoutError, OSError):
        return True


def _filter_usable_chat_names(api_key: str, names: list[str]) -> list[str]:
    if not names:
        return []
    workers = min(_PROBE_WORKERS, len(names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = list(pool.map(lambda name: _chat_access_ok(api_key, name), names))
    return [name for name, ok in zip(names, flags) if ok]


def list_ollama_cloud_chat_models(api_key: str) -> list[dict[str, str]]:
    """Return ``{id, label}`` chat models this ``api_key`` can actually run.

    Ollama 401/403 on ``/api/tags`` become HTTP 400 (invalid API key).
    Network failures become HTTP 502. Ids are prefixed with ``ollama/`` so
    ``crew_base.build_llm`` routes correctly. Does not fall back to the
    static ``OLLAMA_CLOUD_MODELS`` catalog. Per-model 402/403 from
    ``/api/chat`` omit that id; they do not fail the whole list.
    """
    request = urllib.request.Request(
        OLLAMA_CLOUD_TAGS_URL,
        headers=_auth_headers(api_key),
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

    names: list[str] = []
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
        names.append(name)

    models = [
        {"id": _prefixed_id(name), "label": name.removeprefix("ollama/")}
        for name in _filter_usable_chat_names(api_key, names)
    ]
    models.sort(key=lambda entry: entry["id"])
    return models
