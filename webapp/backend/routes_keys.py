"""Stores and returns encrypted-API-key blobs. Never decrypts anything.

The browser derives an encryption key from the user's passphrase (Web Crypto
API, PBKDF2 + AES-GCM — see webapp/frontend/src/lib/crypto.ts) and sends only
ciphertext + IV + KDF params here. This backend has no way to read the
plaintext key; decryption only happens client-side, or transiently in the
run-launcher's memory at the moment a task actually calls the provider.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .db import get_conn
from .routes_auth import require_user

router = APIRouter(prefix="/api/keys", tags=["keys"])

_HOSTED_PROVIDER = "openai"


class StoredKey(BaseModel):
    provider: str
    selected_model: str
    ciphertext_b64: str
    iv_b64: str
    kdf_salt_b64: str
    kdf_params_json: str
    updated_at: str


class StoreKeyRequest(BaseModel):
    provider: str
    selected_model: str = ""
    ciphertext_b64: str
    iv_b64: str
    kdf_salt_b64: str
    kdf_params_json: str


@router.get("", response_model=list[StoredKey])
def list_keys(user_id: int = Depends(require_user)) -> list[StoredKey]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT provider, selected_model, ciphertext_b64, iv_b64, kdf_salt_b64, kdf_params_json, updated_at "
            "FROM api_keys WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [StoredKey(**dict(row)) for row in rows]


@router.put("", response_model=StoredKey)
def upsert_key(body: StoreKeyRequest, user_id: int = Depends(require_user)) -> StoredKey:
    if body.provider != _HOSTED_PROVIDER:
        raise HTTPException(status_code=400, detail="Hosted keys must use provider=openai.")
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO api_keys
                (user_id, provider, selected_model, ciphertext_b64, iv_b64, kdf_salt_b64, kdf_params_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider) DO UPDATE SET
                selected_model = excluded.selected_model,
                ciphertext_b64 = excluded.ciphertext_b64,
                iv_b64 = excluded.iv_b64,
                kdf_salt_b64 = excluded.kdf_salt_b64,
                kdf_params_json = excluded.kdf_params_json,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                body.provider,
                body.selected_model,
                body.ciphertext_b64,
                body.iv_b64,
                body.kdf_salt_b64,
                body.kdf_params_json,
                now,
                now,
            ),
        )
    return StoredKey(**body.model_dump(), updated_at=now)


@router.delete("/{provider}", status_code=204)
def delete_key(provider: str, user_id: int = Depends(require_user)) -> None:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM api_keys WHERE user_id = ? AND provider = ?", (user_id, provider))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="No stored key for this provider.")
