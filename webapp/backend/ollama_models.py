"""Live Ollama Cloud model listing for the hosted account product.

The browser cannot call ollama.com (no CORS), so the webapp proxies
``GET /api/tags`` with the just-decrypted key — the same trust boundary as
``POST /api/tasks``. ``/api/tags`` includes models the key can *see* but not
*run* (HTTP 402 subscription / extra-usage). Those are dropped by a cheap
per-model chat probe so the Tasks dropdown matches what kickoff can use.
The key is never logged or persisted.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import HTTPException

OLLAMA_CLOUD_TAGS_URL = "https://ollama.com/api/tags"
OLLAMA_CLOUD_CHAT_URL = "https://ollama.com/api/chat"

# Capability filter: keep chat/LLM ids the pipeline can run, drop embeddings.
_DROP_SUBSTRINGS = ("embed",)

_PROBE_TIMEOUT_SEC = 15
_PROBE_WORKERS = 8


def is_chat_llm_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(token in lowered for token in _DROP_SUBSTRINGS)


def _prefixed_id(name: str) -> str:
    if name.startswith("ollama/"):
        return name
    return f"ollama/{name}"


def _unprefixed_name(model_id: str) -> str:
    return model_id.removeprefix("ollama/")


def list_ollama_cloud_chat_models(api_key: str) -> list[dict[str, str]]:
    """Return ``{id, label}`` chat models this key can actually run.

    Ollama 401/403 on the tags request become HTTP 400 (invalid API key).
    Network failures become HTTP 502. Models that return 402/403 on a probe
    chat (subscription or extra usage required) are omitted. Ids are prefixed
    with ``ollama/`` so ``crew_base.build_llm`` routes correctly.
    """
    names = _chat_names_from_tags(api_key)
    usable = _filter_usable_names(api_key, names)
    models = [{"id": _prefixed_id(name), "label": name.removeprefix("ollama/")} for name in usable]
    models.sort(key=lambda entry: entry["id"])
    return models


def ollama_cloud_model_available(api_key: str, model_id: str) -> bool:
    """True when ``model_id`` is a tagged chat model this key can invoke."""
    name = _unprefixed_name(model_id.strip())
    if not name or not is_chat_llm_id(name):
        return False
    tagged = set(_chat_names_from_tags(api_key))
    if name not in tagged:
        return False
    return _probe_model_access(api_key, name)


def _chat_names_from_tags(api_key: str) -> list[str]:
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
        name = name.strip().removeprefix("ollama/")
        if not name or name in seen or not is_chat_llm_id(name):
            continue
        seen.add(name)
        names.append(name)
    return names


def _filter_usable_names(api_key: str, names: list[str]) -> list[str]:
    if not names:
        return []
    usable: list[str] = []
    workers = min(_PROBE_WORKERS, len(names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe_model_access, api_key, name): name for name in names}
        for fut in as_completed(futures):
            if fut.result():
                usable.append(futures[fut])
    return usable


def _probe_model_access(api_key: str, name: str) -> bool:
    """Return False when Ollama rejects the model for this key (402/403).

    A 1-token chat is enough for the billing gate to fire without a real
    completion. Timeouts and transport errors treat the model as unusable.
    """
    payload = json.dumps(
        {
            "model": name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "options": {"num_predict": 1},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_CLOUD_CHAT_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_SEC) as response:
            response.read()
        return True
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return False
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
