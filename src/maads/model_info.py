"""Fetch model metadata from Ollama and OpenAI provider APIs."""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from maads.model_capabilities import (
    ModelJsonCapabilities,
    cached_model_capabilities,
    get_model_capabilities,
)

_log = logging.getLogger(__name__)


def resolve_default_model() -> str:
    """Return the configured default chat model id (``MODEL`` env)."""
    return (os.getenv("MODEL") or "").strip()


def fetch_model_info(model: str, *, probe: bool = False) -> dict[str, Any]:
    """Return normalized provider metadata for ``model``.

    When ``probe`` is True, live-probe JSON capabilities (may incur API cost).
    Otherwise include cached probe results only when already available.
    """
    model = (model or "").strip()
    if not model:
        model = resolve_default_model()
    if not model:
        return {
            "model": "",
            "provider": None,
            "available": False,
            "error": "No model configured (set MODEL in .env or pick one in the UI).",
            "details": {},
            "json_capabilities": None,
        }

    from maads.model_catalog import model_label

    label = model_label(model)

    if model.startswith("ollama/"):
        payload = _fetch_ollama_info(model)
    else:
        payload = _fetch_openai_info(model)

    if label:
        payload["label"] = label

    caps = _resolve_json_capabilities(model, probe=probe)
    if caps is not None:
        payload["json_capabilities"] = _caps_dict(caps)
    else:
        payload["json_capabilities"] = None

    return payload


def log_selected_model_info(model: str, *, stream: Any | None = None) -> None:
    """Print provider metadata for an explicitly selected model (CLI/UI runs)."""
    import sys

    out = stream or sys.stderr
    info = fetch_model_info(model, probe=False)
    label = info.get("label")
    heading = label or info.get("model") or model
    print(f"Model: {heading}", file=out)
    if not info.get("available"):
        err = info.get("error") or "unavailable"
        print(f"  Provider check failed: {err}", file=out)
        return
    details = info.get("details") or {}
    bits: list[str] = []
    if info.get("provider"):
        bits.append(str(info["provider"]))
    for key, fmt in (
        ("family", "{}"),
        ("parameter_size", "{}"),
        ("quantization", "{}"),
        ("context_length", "{} ctx"),
    ):
        val = details.get(key)
        if val is not None:
            bits.append(fmt.format(val))
    if bits:
        print(f"  {' · '.join(bits)}", file=out)
    caps = info.get("json_capabilities")
    if caps:
        print(f"  JSON: {caps.get('mode', 'unknown')}", file=out)


def _resolve_json_capabilities(
    model: str,
    *,
    probe: bool,
) -> ModelJsonCapabilities | None:
    if probe:
        try:
            return get_model_capabilities(model)
        except Exception as exc:
            _log.debug("JSON capability probe failed for %s: %s", model, exc)
            return None
    return cached_model_capabilities(model)


def _caps_dict(caps: ModelJsonCapabilities) -> dict[str, Any]:
    return {
        "mode": caps.mode.value,
        "structured_outputs": caps.structured_outputs,
        "json_mode": caps.json_mode,
    }


def _fetch_ollama_info(model: str) -> dict[str, Any]:
    from maads.ollama_runtime import ollama_base_url, ollama_client_kwargs

    model_name = model.removeprefix("ollama/")
    host = ollama_base_url()
    details: dict[str, Any] = {"host": host, "model_name": model_name}
    provider_raw: dict[str, Any] = {}

    try:
        import ollama

        client = ollama.Client(**ollama_client_kwargs())
        show = client.show(model_name)
        show_dict = show.model_dump() if hasattr(show, "model_dump") else dict(show)
        provider_raw["show"] = _trim_ollama_show(show_dict)

        if show_dict.get("details"):
            d = show_dict["details"]
            if hasattr(d, "model_dump"):
                d = d.model_dump()
            details.update(
                {
                    "family": d.get("family"),
                    "families": d.get("families"),
                    "format": d.get("format"),
                    "parameter_size": d.get("parameter_size"),
                    "quantization": d.get("quantization_level"),
                    "parent_model": d.get("parent_model") or None,
                }
            )

        if show_dict.get("capabilities"):
            details["capabilities"] = list(show_dict["capabilities"])

        if show_dict.get("parameters"):
            details["parameters"] = _parse_ollama_parameters(show_dict["parameters"])

        if show_dict.get("modified_at"):
            details["modified_at"] = _iso_timestamp(show_dict["modified_at"])

        modelinfo = show_dict.get("modelinfo") or {}
        if modelinfo:
            provider_raw["modelinfo_keys"] = sorted(modelinfo.keys())
            details["architecture"] = modelinfo.get("general.architecture")
            param_count = modelinfo.get("general.parameter_count")
            if param_count is not None:
                details["parameter_count"] = int(param_count)
            ctx = _ollama_context_length(modelinfo)
            if ctx is not None:
                details["context_length"] = ctx
            license_name = modelinfo.get("general.license")
            if license_name and isinstance(license_name, str):
                details["license"] = license_name.split("\n", 1)[0].strip()

        list_entry = _ollama_list_entry(client, model_name)
        if list_entry:
            provider_raw["list"] = list_entry
            if list_entry.get("size") is not None:
                details["size_bytes"] = list_entry["size"]
            if list_entry.get("modified_at") and "modified_at" not in details:
                details["modified_at"] = _iso_timestamp(list_entry["modified_at"])

        return {
            "model": model,
            "provider": "ollama",
            "available": True,
            "error": None,
            "details": details,
            "provider_raw": provider_raw,
        }
    except Exception as exc:
        _log.debug("Ollama model info failed for %s: %s", model_name, exc)
        return {
            "model": model,
            "provider": "ollama",
            "available": False,
            "error": str(exc),
            "details": details,
            "provider_raw": provider_raw,
        }


def _fetch_openai_info(model: str) -> dict[str, Any]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return {
            "model": model,
            "provider": "openai",
            "available": False,
            "error": "OPENAI_API_KEY is not set.",
            "details": {},
            "provider_raw": {},
        }

    from openai import OpenAI

    client = OpenAI()
    details: dict[str, Any] = {}
    provider_raw: dict[str, Any] = {}

    try:
        retrieved = client.models.retrieve(model)
        row = retrieved.model_dump() if hasattr(retrieved, "model_dump") else dict(retrieved)
        provider_raw["retrieve"] = row
        details["owned_by"] = row.get("owned_by")
        if row.get("created") is not None:
            details["created"] = int(row["created"])
            details["created_at"] = datetime.fromtimestamp(int(row["created"]), UTC).isoformat(
                timespec="seconds"
            )
    except Exception as exc:
        _log.debug("OpenAI model retrieve failed for %s: %s", model, exc)
        return {
            "model": model,
            "provider": "openai",
            "available": False,
            "error": str(exc),
            "details": details,
            "provider_raw": provider_raw,
        }

    # Best-effort: confirm the id appears in the account's model list.
    try:
        listed = [m.id for m in client.models.list().data]
        provider_raw["listed"] = model in listed
        details["listed_in_account"] = model in listed
    except Exception as exc:
        _log.debug("OpenAI model list failed: %s", exc)

    return {
        "model": model,
        "provider": "openai",
        "available": True,
        "error": None,
        "details": details,
        "provider_raw": provider_raw,
    }


def _ollama_list_entry(client: Any, model_name: str) -> dict[str, Any] | None:
    try:
        listed = client.list()
        models = listed.models if hasattr(listed, "models") else listed.get("models", [])
        for entry in models:
            name = entry.model if hasattr(entry, "model") else entry.get("model")
            if name == model_name or (name and name.split(":", 1)[0] == model_name.split(":", 1)[0]):
                if hasattr(entry, "model_dump"):
                    return entry.model_dump()
                return dict(entry)
    except Exception as exc:
        _log.debug("Ollama list failed: %s", exc)
    return None


def _ollama_context_length(modelinfo: dict[str, Any]) -> int | None:
    best: int | None = None
    for key, val in modelinfo.items():
        if not key.endswith(".context_length"):
            continue
        if isinstance(val, (int, float)):
            ival = int(val)
            best = ival if best is None else max(best, ival)
    return best


def _parse_ollama_parameters(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def _trim_ollama_show(show: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(show)
    for heavy in ("modelfile", "template", "license"):
        if heavy in trimmed:
            val = trimmed[heavy]
            if isinstance(val, str) and len(val) > 200:
                trimmed[heavy] = val[:200] + "…"
    if "modelinfo" in trimmed:
        trimmed.pop("modelinfo", None)
    return trimmed


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
