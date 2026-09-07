"""End-to-end API tests for username auth, ciphertext-only keys, and live models."""

from __future__ import annotations

import sqlite3

import httpx
import webapp.backend.db as db_module
from fastapi.testclient import TestClient
from maads.model_catalog import model_catalog
from openai import AuthenticationError

from tests.webapp.openai_stub import FakeOpenAI


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAPP_ALLOW_DEV_SECRET", "1")
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", tmp_path / "webapp.db")
    from webapp.backend.app import create_app

    return TestClient(create_app())


def test_register_then_login_returns_tokens(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    resp = client.post("/api/auth/register", json={"username": "alice"})
    assert resp.status_code == 201
    assert "access_token" in resp.json()

    resp = client.post("/api/auth/login", json={"username": "alice"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_unknown_username_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    resp = client.post("/api/auth/login", json={"username": "nobody"})
    assert resp.status_code == 401


def test_duplicate_registration_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    client.post("/api/auth/register", json={"username": "carol"})
    resp = client.post("/api/auth/register", json={"username": "carol"})
    assert resp.status_code == 409


def test_register_rejects_invalid_username(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    resp = client.post("/api/auth/register", json={"username": "bad name"})
    assert resp.status_code == 422


def test_keys_endpoint_requires_auth(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    resp = client.get("/api/keys")
    assert resp.status_code == 401


def test_store_and_list_key_only_ever_holds_ciphertext(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    token = client.post("/api/auth/register", json={"username": "dana"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    body = {
        "provider": "openai",
        "selected_model": "gpt-5.4-mini",
        "ciphertext_b64": "c2VjcmV0LWNpcGhlcnRleHQ=",
        "iv_b64": "aXY=",
        "kdf_salt_b64": "c2FsdA==",
        "kdf_params_json": '{"iterations": 210000}',
    }
    resp = client.put("/api/keys", json=body, headers=headers)
    assert resp.status_code == 200

    resp = client.get("/api/keys", headers=headers)
    assert resp.status_code == 200
    [stored] = resp.json()
    assert stored["ciphertext_b64"] == body["ciphertext_b64"]
    assert "api_key" not in stored and "plaintext" not in stored


def test_store_key_rejects_non_openai_provider(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    token = client.post("/api/auth/register", json={"username": "erin"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.put(
        "/api/keys",
        json={
            "provider": "ollama_cloud",
            "selected_model": "ollama/gpt-oss:20b-cloud",
            "ciphertext_b64": "c2VjcmV0LWNpcGhlcnRleHQ=",
            "iv_b64": "aXY=",
            "kdf_salt_b64": "c2FsdA==",
            "kdf_params_json": '{"iterations": 210000}',
        },
        headers=headers,
    )
    assert resp.status_code == 400


def test_models_requires_auth(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    resp = client.post("/api/models", json={"decrypted_api_key": "sk-x"})
    assert resp.status_code == 401


def test_models_returns_filtered_live_list_not_catalog(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    token = client.post("/api/auth/register", json={"username": "frank"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/api/models", json={"decrypted_api_key": "sk-live"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    ids = [entry["id"] for entry in body]
    assert ids == ["gpt-4o", "gpt-5.4", "gpt-5.4-mini"]
    assert all(entry["label"] == entry["id"] for entry in body)
    assert FakeOpenAI.last_api_key == "sk-live"
    assert body != model_catalog()
    assert "openai" not in body and "ollama_cloud" not in body
    assert "text-embedding-3-small" not in ids
    assert "whisper-1" not in ids
    assert "gpt-image-1" not in ids


def test_models_invalid_openai_key_returns_400(tmp_path, monkeypatch):
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    FakeOpenAI.error = AuthenticationError(
        "Invalid API key",
        response=httpx.Response(401, request=request),
        body=None,
    )
    client = make_client(tmp_path, monkeypatch)
    token = client.post("/api/auth/register", json={"username": "gina"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/api/models", json={"decrypted_api_key": "sk-bad"}, headers=headers)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "invalid API key"


def _seed_legacy_db(db_path):
    """Write a pre-username schema without going through init_db."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                provider TEXT NOT NULL,
                selected_model TEXT NOT NULL,
                ciphertext_b64 TEXT NOT NULL,
                iv_b64 TEXT NOT NULL,
                kdf_salt_b64 TEXT NOT NULL,
                kdf_params_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, provider)
            );
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                case_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                started_at TEXT,
                finished_at TEXT,
                run_artifact_path TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (7, "miro.georgiev@gmail.com", "argon2-hash-must-be-dropped", "2026-01-01T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO api_keys
                (user_id, provider, selected_model, ciphertext_b64, iv_b64, kdf_salt_b64, kdf_params_json,
                 created_at, updated_at)
            VALUES (7, 'openai', 'gpt-5.4', 'cipher', 'iv', 'salt', '{}', '2026-01-02T00:00:00+00:00',
                    '2026-01-02T00:00:00+00:00')
            """
        )
        conn.execute(
            "INSERT INTO tasks (user_id, case_name, provider, model_id, status) VALUES (7, 'titanic', 'openai', 'gpt-5.4', 'failed')"
        )


def test_init_db_migrates_legacy_users_in_place(tmp_path, monkeypatch):
    db_path = tmp_path / "webapp.db"
    _seed_legacy_db(db_path)
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setenv("WEBAPP_ALLOW_DEV_SECRET", "1")

    db_module.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        assert cols == {"id", "username", "created_at"}
        row = conn.execute("SELECT id, username FROM users").fetchone()
        assert row == (7, "miro.georgiev@gmail.com")
        key = conn.execute("SELECT user_id, ciphertext_b64 FROM api_keys").fetchone()
        assert key == (7, "cipher")
        task = conn.execute("SELECT user_id, case_name FROM tasks").fetchone()
        assert task == (7, "titanic")

    from webapp.backend.app import create_app

    client = TestClient(create_app())
    resp = client.post("/api/auth/login", json={"username": "miro.georgiev@gmail.com"})
    assert resp.status_code == 200
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    stored = client.get("/api/keys", headers=headers).json()
    assert stored[0]["ciphertext_b64"] == "cipher"
    assert stored[0]["selected_model"] == "gpt-5.4"
