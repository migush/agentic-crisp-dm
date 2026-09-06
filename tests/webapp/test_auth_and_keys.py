"""End-to-end API tests for the account backend: register/login, and
ciphertext-only key storage (the backend must never see a plaintext key)."""

from __future__ import annotations

import webapp.backend.db as db_module
from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    db_path = tmp_path / "webapp.db"
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", db_path)

    from webapp.backend.app import create_app

    app = create_app()
    return TestClient(app)


def test_register_then_login_returns_tokens(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    resp = client.post("/api/auth/register", json={"email": "a@example.com", "password": "correct horse battery"})
    assert resp.status_code == 201
    assert "access_token" in resp.json()

    resp = client.post("/api/auth/login", json={"email": "a@example.com", "password": "correct horse battery"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    client.post("/api/auth/register", json={"email": "b@example.com", "password": "correct horse battery"})

    resp = client.post("/api/auth/login", json={"email": "b@example.com", "password": "wrong password"})
    assert resp.status_code == 401


def test_duplicate_registration_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    client.post("/api/auth/register", json={"email": "c@example.com", "password": "correct horse battery"})

    resp = client.post("/api/auth/register", json={"email": "c@example.com", "password": "another password"})
    assert resp.status_code == 409


def test_keys_endpoint_requires_auth(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    resp = client.get("/api/keys")
    assert resp.status_code == 401


def test_store_and_list_key_only_ever_holds_ciphertext(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    token = client.post(
        "/api/auth/register", json={"email": "d@example.com", "password": "correct horse battery"}
    ).json()["access_token"]
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


def test_models_endpoint_reuses_model_catalog():
    from maads.model_catalog import model_catalog
    from webapp.backend.app import create_app

    client = TestClient(create_app())
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == model_catalog()
