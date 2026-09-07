"""Task launch/list/spend API tests.

run_launcher.run_task actually shells out to the maads pipeline, so these
tests monkeypatch it to a no-op and instead exercise the DB-facing surface:
task creation, ownership checks, and spend reporting.
"""

from __future__ import annotations

import webapp.backend.db as db_module
from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", tmp_path / "webapp.db")
    from webapp.backend.app import create_app

    return TestClient(create_app())


def register(client, email="a@example.com") -> dict:
    resp = client.post("/api/auth/register", json={"email": email, "password": "correct horse battery"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_launch_task_rejects_unknown_provider(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={"case_name": "titanic", "provider": "not-a-provider", "model_id": "x", "decrypted_api_key": "sk-x"},
        headers=headers,
    )
    assert resp.status_code == 400


def test_launch_task_queues_and_never_echoes_key(tmp_path, monkeypatch):
    calls = []
    import webapp.backend.run_launcher as run_launcher

    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: calls.append(kw))

    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={
            "case_name": "titanic",
            "provider": "openai",
            "model_id": "gpt-5.4-mini",
            "decrypted_api_key": "sk-super-secret",
        },
        headers=headers,
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert "decrypted_api_key" not in body and "sk-super-secret" not in resp.text


def test_list_tasks_only_returns_own_tasks(tmp_path, monkeypatch):
    import webapp.backend.run_launcher as run_launcher

    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: None)

    client = make_client(tmp_path, monkeypatch)
    headers_a = register(client, "a@example.com")
    headers_b = register(client, "b@example.com")

    client.post(
        "/api/tasks",
        json={"case_name": "titanic", "provider": "openai", "model_id": "gpt-5.4-mini", "decrypted_api_key": "sk-a"},
        headers=headers_a,
    )

    resp_a = client.get("/api/tasks", headers=headers_a)
    resp_b = client.get("/api/tasks", headers=headers_b)
    assert len(resp_a.json()) == 1
    assert len(resp_b.json()) == 0


def test_spend_endpoint_rejects_other_users_task(tmp_path, monkeypatch):
    import webapp.backend.run_launcher as run_launcher

    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: None)

    client = make_client(tmp_path, monkeypatch)
    headers_a = register(client, "a@example.com")
    headers_b = register(client, "b@example.com")

    resp = client.post(
        "/api/tasks",
        json={"case_name": "titanic", "provider": "openai", "model_id": "gpt-5.4-mini", "decrypted_api_key": "sk-a"},
        headers=headers_a,
    )
    task_id = resp.json()["id"]

    resp = client.get(f"/api/tasks/{task_id}/spend", headers=headers_b)
    assert resp.status_code == 404
