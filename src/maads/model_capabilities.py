"""Probe configured LLM models for JSON mode vs Structured Outputs support."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)

_AGENT_NAMES = (
    "pm",
    "domain",
    "data_engineer",
    "data_scientist",
    "developer",
    "storyteller",
)

_PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}

_capabilities_cache: dict[str, ModelJsonCapabilities] | None = None


class JsonFormatMode(str, Enum):
    """Best JSON enforcement mode available for a model."""

    NONE = "none"
    JSON_MODE = "json_mode"
    STRUCTURED_OUTPUTS = "structured_outputs"


@dataclass(frozen=True)
class ModelJsonCapabilities:
    """Detected JSON formatting support for one model id."""

    model: str
    structured_outputs: bool
    json_mode: bool

    @property
    def mode(self) -> JsonFormatMode:
        if self.structured_outputs:
            return JsonFormatMode.STRUCTURED_OUTPUTS
        if self.json_mode:
            return JsonFormatMode.JSON_MODE
        return JsonFormatMode.NONE


def _skip_probe() -> bool:
    return os.getenv("MAADS_SKIP_MODEL_PROBE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configured_models() -> dict[str, str]:
    """Return unique configured model ids mapped to an example agent slug."""
    from maads.crew_base import resolve_model_for_agent

    seen: dict[str, str] = {}
    for agent in _AGENT_NAMES:
        model = resolve_model_for_agent(agent)
        seen.setdefault(model, agent)
    return seen


def probe_model(model: str) -> ModelJsonCapabilities:
    """Live-probe a single model for Structured Outputs and JSON mode."""
    if model.startswith("ollama/"):
        return _probe_ollama(model)
    return _probe_openai_compatible(model)


def cached_model_capabilities(model: str) -> ModelJsonCapabilities | None:
    """Return probe results for ``model`` when already cached, else None."""
    if _capabilities_cache is None:
        return None
    return _capabilities_cache.get(model)


def get_model_capabilities(model: str) -> ModelJsonCapabilities:
    """Return cached capabilities for ``model``, probing on demand if needed."""
    global _capabilities_cache
    if _capabilities_cache is None:
        _capabilities_cache = {}
    cached = _capabilities_cache.get(model)
    if cached is not None:
        return cached
    if _skip_probe():
        cached = ModelJsonCapabilities(model, False, False)
    else:
        cached = probe_model(model)
    _capabilities_cache[model] = cached
    return cached


def ensure_model_capabilities_probed() -> dict[str, ModelJsonCapabilities]:
    """Probe every configured model once; safe to call at CLI startup."""
    global _capabilities_cache
    if _capabilities_cache is not None:
        return dict(_capabilities_cache)

    _capabilities_cache = {}
    if _skip_probe():
        for model in configured_models():
            _capabilities_cache[model] = ModelJsonCapabilities(model, False, False)
        return dict(_capabilities_cache)

    for model in configured_models():
        caps = probe_model(model)
        _capabilities_cache[model] = caps
        _log.info(
            "Model %s: %s (structured_outputs=%s json_mode=%s)",
            model,
            caps.mode.value,
            caps.structured_outputs,
            caps.json_mode,
        )
    return dict(_capabilities_cache)


def log_model_capabilities(*, stream: Any | None = None) -> None:
    """Print a one-line summary per configured model (for CLI startup)."""
    import sys

    out = stream or sys.stderr
    caps = ensure_model_capabilities_probed()
    if not caps:
        print("Model capabilities: no models configured", file=out)
        return
    print("Model capabilities:", file=out)
    for model in sorted(caps):
        mode = caps[model].mode.value
        print(f"  {model}: {mode}", file=out)


def reset_model_capabilities_cache() -> None:
    """Clear cached probe results (tests)."""
    global _capabilities_cache
    _capabilities_cache = None


def _probe_openai_compatible(model: str) -> ModelJsonCapabilities:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return ModelJsonCapabilities(model, False, False)

    from openai import OpenAI

    client = OpenAI()
    structured = _probe_openai_structured(client, model)
    json_mode = False if structured else _probe_openai_json_mode(client, model)
    return ModelJsonCapabilities(model, structured, json_mode)


def _probe_openai_structured(client: Any, model: str) -> bool:
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Return ok=true."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "maads_probe",
                    "strict": True,
                    "schema": _PROBE_SCHEMA,
                },
            },
            max_completion_tokens=32,
        )
        return True
    except Exception as exc:
        _log.debug("Structured Outputs probe failed for %s: %s", model, exc)
        return False


def _probe_openai_json_mode(client: Any, model: str) -> bool:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Respond with JSON only."},
                {"role": "user", "content": 'Return {"ok": true}.'},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=32,
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        return isinstance(parsed, dict)
    except Exception as exc:
        _log.debug("JSON mode probe failed for %s: %s", model, exc)
        return False


def _probe_ollama(model: str) -> ModelJsonCapabilities:
    model_name = model.removeprefix("ollama/")
    structured = _probe_ollama_structured(model_name)
    json_mode = False if structured else _probe_ollama_json_mode(model_name)
    return ModelJsonCapabilities(model, structured, json_mode)


def _ollama_probe_client() -> Any:
    import ollama
    from maads.ollama_runtime import ollama_client_kwargs

    return ollama.Client(**ollama_client_kwargs())


def _probe_ollama_json_mode(model_name: str) -> bool:
    try:
        response = _ollama_probe_client().chat(
            model=model_name,
            messages=[{"role": "user", "content": 'Return {"ok": true} as JSON only.'}],
            format="json",
            options={"num_predict": 32},
        )
        parsed = json.loads(response.message.content or "")
        return isinstance(parsed, dict)
    except Exception as exc:
        _log.debug("Ollama JSON mode probe failed for %s: %s", model_name, exc)
        return False


def _probe_ollama_structured(model_name: str) -> bool:
    try:
        response = _ollama_probe_client().chat(
            model=model_name,
            messages=[{"role": "user", "content": "Return ok=true."}],
            format=_PROBE_SCHEMA,
            options={"num_predict": 32},
        )
        parsed = json.loads(response.message.content or "")
        return isinstance(parsed, dict) and "ok" in parsed
    except Exception as exc:
        _log.debug("Ollama structured probe failed for %s: %s", model_name, exc)
        return False
