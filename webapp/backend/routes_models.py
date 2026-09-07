"""Authenticated proxy for the OpenAI Models API."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .openai_models import list_openai_chat_models
from .routes_auth import require_user

router = APIRouter(prefix="/api/models", tags=["models"])


class ListModelsRequest(BaseModel):
    decrypted_api_key: str = Field(min_length=1)


@router.post("")
def list_models(body: ListModelsRequest, _user_id: int = Depends(require_user)) -> list[dict[str, str]]:
    """Return chat/LLM ids visible to this key. Never logs or persists the key."""
    return list_openai_chat_models(body.decrypted_api_key)
