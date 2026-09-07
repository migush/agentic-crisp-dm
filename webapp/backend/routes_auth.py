from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import os

from fastapi import APIRouter, Cookie, Header, HTTPException, Response
from pydantic import BaseModel, EmailStr, Field

from .auth import decode_access_token, hash_password, issue_access_token, verify_password
from .db import get_conn

router = APIRouter(prefix="/api/auth", tags=["auth"])

# The account SPA sends the token as an Authorization header, but the trace
# dashboard also downloads files through plain <a href> links and <img> URLs,
# which cannot carry a header. Mirroring the token into an HttpOnly cookie lets
# those work without exposing it to JS. Same token, same TTL, same secret.
SESSION_COOKIE = "maads_session"
_COOKIE_MAX_AGE = 12 * 60 * 60  # matches _ACCESS_TOKEN_TTL in auth.py


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        # Off only for plain-HTTP local dev; the deployed site is HTTPS-only.
        secure=os.environ.get("WEBAPP_INSECURE_COOKIES") != "1",
        samesite="lax",
        path="/",
    )


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, response: Response) -> TokenResponse:
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (body.email, hash_password(body.password), datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="An account with this email already exists.") from None
        user_id = cur.lastrowid
    token = issue_access_token(user_id)
    _set_session_cookie(response, token)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, response: Response) -> TokenResponse:
    with get_conn() as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE email = ?", (body.email,)).fetchone()
    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = issue_access_token(row["id"])
    _set_session_cookie(response, token)
    return TokenResponse(access_token=token)


@router.post("/logout", status_code=204)
def logout(response: Response) -> None:
    """Clear the session cookie. The bearer token in localStorage is dropped by
    the client; JWTs stay valid until they expire (there is no revocation list)."""
    response.delete_cookie(SESSION_COOKIE, path="/")


def require_user(authorization: str | None = Header(default=None)) -> int:
    """FastAPI dependency: extract and validate the bearer token, return the user id."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(" ", 1)[1]
    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user_id


def require_user_flexible(
    authorization: str | None = Header(default=None),
    maads_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> int:
    """Like :func:`require_user`, but also accepts the session cookie.

    Used by the mounted trace dashboard, whose file downloads and figure URLs
    are plain browser navigations that cannot set an Authorization header.
    """
    if authorization and authorization.lower().startswith("bearer "):
        user_id = decode_access_token(authorization.split(" ", 1)[1])
        if user_id is not None:
            return user_id
    if maads_session:
        user_id = decode_access_token(maads_session)
        if user_id is not None:
            return user_id
    raise HTTPException(status_code=401, detail="Not authenticated.")
