"""Hosted BYO-key providers (OpenAI and Ollama Cloud)."""

from __future__ import annotations

from fastapi import HTTPException

from .ollama_models import list_ollama_cloud_chat_models, ollama_cloud_model_available
from .openai_models import list_openai_chat_models

HOSTED_PROVIDERS = frozenset({"openai", "ollama_cloud"})

OLLAMA_CLOUD_BASE_URL = "https://ollama.com"

PROVIDER_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "ollama_cloud": "OLLAMA_API_KEY",
}

# Strip the parent's own provider env so a VPS .env cannot hijack a user run.
PARENT_ENV_DENYLIST = frozenset({"OPENAI_API_KEY", "OLLAMA_API_KEY", "OLLAMA_BASE_URL"})


def require_hosted_provider(provider: str, *, kind: str = "keys") -> str:
    if provider not in HOSTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Hosted {kind} require provider=openai or provider=ollama_cloud.",
        )
    return provider


def list_live_chat_models(provider: str, api_key: str) -> list[dict[str, str]]:
    require_hosted_provider(provider, kind="tasks")
    if provider == "openai":
        return list_openai_chat_models(api_key)
    return list_ollama_cloud_chat_models(api_key)


def model_available_for_key(provider: str, api_key: str, model_id: str) -> bool:
    require_hosted_provider(provider, kind="tasks")
    if provider == "openai":
        live_ids = {entry["id"] for entry in list_openai_chat_models(api_key)}
        return model_id in live_ids
    return ollama_cloud_model_available(api_key, model_id)
