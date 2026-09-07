"""Password hashing and JWT session tokens for the account product.

Both pieces are free/OSS with no external service: argon2-cffi for password
hashing, PyJWT for short-lived bearer tokens. This module only ever handles
the user's *login* password — the LLM provider API key is a separate secret,
encrypted client-side, and never passes through here (see routes_keys.py).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_hasher = PasswordHasher()

# MVP: a fixed secret from the environment. Rotate by setting a new
# WEBAPP_JWT_SECRET (invalidates all existing sessions). Must be set to a
# real random value (e.g. `openssl rand -hex 32`) outside local dev.
_JWT_SECRET = os.environ.get("WEBAPP_JWT_SECRET", "dev-secret-change-me-dev-secret-change-me")
_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_TTL = timedelta(hours=12)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def issue_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + _ACCESS_TOKEN_TTL}
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_access_token(token: str) -> int | None:
    """Return the user id encoded in ``token``, or ``None`` if invalid/expired."""
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    return int(payload["sub"])
