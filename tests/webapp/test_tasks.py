"""Task launch/list/spend API tests.

run_launcher.run_task actually shells out to the maads pipeline, so these
tests monkeypatch it to a no-op and instead exercise the DB-facing surface:
task creation, ownership checks, and spend reporting. OpenAI Models API is
stubbed in tests/webapp/conftest.py (no live network).
"""

from __future__ import annotations

import pytest
import webapp.backend.db as db_module
from fastapi.testclient import TestClient

from tests.webapp.openai_stub import FakeOpenAI
from tests.webapp.ollama_stub import FakeOllamaTags


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAPP_ALLOW_DEV_SECRET", "1")
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", tmp_path / "webapp.db")
    from webapp.backend.app import create_app

    return TestClient(create_app())


def register(client, username="alice") -> dict:
    resp = client.post("/api/auth/register", json={"username": username})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_launch_task_accepts_ollama_cloud(tmp_path, monkeypatch):
    calls = []
    import webapp.backend.run_launcher as run_launcher

    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: calls.append(kw))

    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={
            "case_name": "titanic",
            "provider": "ollama_cloud",
            "model_id": "ollama/gpt-oss:120b",
            "decrypted_api_key": "ollama-secret",
        },
        headers=headers,
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["provider"] == "ollama_cloud"
    assert body["model_id"] == "ollama/gpt-oss:120b"
    assert "ollama-secret" not in resp.text
    assert FakeOllamaTags.last_authorization == "Bearer ollama-secret"
    assert calls[0]["provider"] == "ollama_cloud"
    assert calls[0]["decrypted_api_key"] == "ollama-secret"


def test_launch_task_rejects_ollama_model_not_in_live_list(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={
            "case_name": "titanic",
            "provider": "ollama_cloud",
            "model_id": "ollama/does-not-exist",
            "decrypted_api_key": "ollama-secret",
        },
        headers=headers,
    )
    assert resp.status_code == 400


def test_launch_task_rejects_unknown_provider(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={"case_name": "titanic", "provider": "not-a-provider", "model_id": "x", "decrypted_api_key": "sk-x"},
        headers=headers,
    )
    assert resp.status_code == 400


@pytest.mark.parametrize("case_name", ["../../etc/passwd", "not_a_case", "", "titanic/../x"])
def test_launch_task_rejects_unknown_case(tmp_path, monkeypatch, case_name):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={"case_name": case_name, "provider": "openai", "model_id": "gpt-4o", "decrypted_api_key": "sk-x"},
        headers=headers,
    )
    assert resp.status_code == 400


def test_launch_ready_user_case_uses_config_path(tmp_path, monkeypatch):
    import webapp.backend.paths as paths_module
    import webapp.backend.run_launcher as run_launcher

    calls = []
    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(
        paths_module,
        "user_cases_root",
        lambda uid: tmp_path / "users" / str(uid) / "cases",
    )

    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    created = client.post(
        "/api/cases",
        data={"title": "Widget labels", "problem_statement": "Predict the class of each widget."},
        files=[("files", ("widgets.csv", b"rec_id,label\n1,a\n2,b\n", "text/csv"))],
        headers=headers,
    )
    assert created.status_code == 201
    cid = created.json()["case_id"]
    ready = client.put(
        f"/api/cases/{cid}",
        json={"problem_type": "classification", "target_column": "label", "mark_ready": True},
        headers=headers,
    )
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    resp = client.post(
        "/api/tasks",
        json={
            "case_name": cid,
            "provider": "openai",
            "model_id": "gpt-4o",
            "decrypted_api_key": "sk-x",
        },
        headers=headers,
    )
    assert resp.status_code == 202, resp.text
    assert calls[0]["case_name"] == cid
    assert calls[0]["config_path"]
    assert calls[0]["config_path"].endswith("case.yaml")


def test_launch_draft_user_case_rejected(tmp_path, monkeypatch):
    import webapp.backend.paths as paths_module

    monkeypatch.setattr(
        paths_module,
        "user_cases_root",
        lambda uid: tmp_path / "users" / str(uid) / "cases",
    )
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    created = client.post(
        "/api/cases",
        data={"title": "Drafty", "problem_statement": "Predict label."},
        files=[("files", ("ok.csv", b"rec_id,label\n1,a\n", "text/csv"))],
        headers=headers,
    )
    cid = created.json()["case_id"]
    resp = client.post(
        "/api/tasks",
        json={"case_name": cid, "provider": "openai", "model_id": "gpt-4o", "decrypted_api_key": "sk-x"},
        headers=headers,
    )
    assert resp.status_code == 400


def test_launch_task_rejects_model_not_in_live_list(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={
            "case_name": "titanic",
            "provider": "openai",
            "model_id": "gpt-does-not-exist",
            "decrypted_api_key": "sk-x",
        },
        headers=headers,
    )
    assert resp.status_code == 400


def test_launch_task_rejects_blank_model_id(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={"case_name": "titanic", "provider": "openai", "model_id": "   ", "decrypted_api_key": "sk-x"},
        headers=headers,
    )
    assert resp.status_code == 400


def test_launch_task_rejects_preflight_failure(tmp_path, monkeypatch):
    """HTTP 400 when demo data paths fail preflight (no subprocess)."""
    import webapp.backend.run_launcher as run_launcher

    calls = []
    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(
        "maads.preflight.preflight_case",
        lambda _cfg: ["train not found or not a file: /missing/train.csv"],
    )

    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/tasks",
        json={
            "case_name": "titanic",
            "provider": "openai",
            "model_id": "gpt-5.4-mini",
            "decrypted_api_key": "sk-x",
        },
        headers=headers,
    )
    assert resp.status_code == 400
    assert "train not found" in resp.json()["detail"]
    assert calls == []


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
    assert body["model_id"] == "gpt-5.4-mini"
    assert "decrypted_api_key" not in body and "sk-super-secret" not in resp.text
    assert FakeOpenAI.last_api_key == "sk-super-secret"


def test_launch_may_use_model_different_from_stored_selected_model(tmp_path, monkeypatch):
    import webapp.backend.run_launcher as run_launcher

    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: None)

    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    client.put(
        "/api/keys",
        json={
            "provider": "openai",
            "selected_model": "gpt-5.4",
            "ciphertext_b64": "c2VjcmV0LWNpcGhlcnRleHQ=",
            "iv_b64": "aXY=",
            "kdf_salt_b64": "c2FsdA==",
            "kdf_params_json": '{"iterations": 210000}',
        },
        headers=headers,
    )
    resp = client.post(
        "/api/tasks",
        json={
            "case_name": "titanic",
            "provider": "openai",
            "model_id": "gpt-4o",
            "decrypted_api_key": "sk-a",
        },
        headers=headers,
    )
    assert resp.status_code == 202
    assert resp.json()["model_id"] == "gpt-4o"
    stored = client.get("/api/keys", headers=headers).json()
    assert stored[0]["selected_model"] == "gpt-5.4"


def test_list_tasks_only_returns_own_tasks(tmp_path, monkeypatch):
    import webapp.backend.run_launcher as run_launcher

    monkeypatch.setattr(run_launcher, "run_task", lambda *a, **kw: None)

    client = make_client(tmp_path, monkeypatch)
    headers_a = register(client, "alice")
    headers_b = register(client, "bob")

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
    headers_a = register(client, "alice")
    headers_b = register(client, "bob")

    resp = client.post(
        "/api/tasks",
        json={"case_name": "titanic", "provider": "openai", "model_id": "gpt-5.4-mini", "decrypted_api_key": "sk-a"},
        headers=headers_a,
    )
    task_id = resp.json()["id"]

    resp = client.get(f"/api/tasks/{task_id}/spend", headers=headers_b)
    assert resp.status_code == 404
