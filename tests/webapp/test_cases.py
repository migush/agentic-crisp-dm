"""User case create/list/inspect/confirm API tests.

Uses generic messy mini CSVs — not a named demo schema.
"""

from __future__ import annotations

import webapp.backend.db as db_module
import webapp.backend.paths as paths_module
from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBAPP_ALLOW_DEV_SECRET", "1")
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", tmp_path / "webapp.db")
    monkeypatch.setattr(
        paths_module,
        "user_cases_root",
        lambda uid: tmp_path / "users" / str(uid) / "cases",
    )
    from webapp.backend.app import create_app

    return TestClient(create_app())


def register(client, username="alice") -> dict:
    resp = client.post("/api/auth/register", json={"username": username})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _widgets_csv() -> bytes:
    lines = ["rec_id,note,label"]
    for i, label in enumerate(["alpha", "beta", "gamma"] * 8):
        lines.append(f"{i+1},item {i} café,{label}")
    return "\n".join(lines).encode("latin-1")


def _holdout_csv() -> bytes:
    lines = ["rec_id,note,label"]
    for i, label in enumerate(["alpha", "beta", "gamma"] * 2):
        lines.append(f"{100+i},holdout {i},{label}")
    return "\n".join(lines).encode("utf-8")


def test_create_list_inspect_confirm_and_isolation(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    alice = register(client, "alice")
    bob = register(client, "bob")

    resp = client.post(
        "/api/cases",
        data={
            "title": "Widget labels",
            "problem_statement": "Predict the class of each widget from a short note.",
        },
        files=[
            ("files", ("widgets.csv", _widgets_csv(), "text/csv")),
            ("files", ("holdout.csv", _holdout_csv(), "text/csv")),
        ],
        headers=alice,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["kind"] == "user"
    assert body["status"] == "draft"
    assert body["case_id"] == "widget_labels"
    assert "UserName" not in resp.text and "Sentiment" not in resp.text
    files = body["inspect"]["files"]
    assert len(files) == 2
    encodings = {f["encoding"] for f in files}
    assert encodings & {"latin-1", "cp1252", "utf-8", "utf-8-sig"}
    cols = {c["name"] for c in files[0]["columns"]}
    assert cols == {"rec_id", "note", "label"}
    assert files[0]["n_rows"] > 0
    assert any(c["n_unique"] > 1 for c in files[0]["columns"])

    listed = client.get("/api/cases", headers=alice).json()
    demo_ids = {c["case_id"] for c in listed if c["kind"] == "demo"}
    mine = [c for c in listed if c["kind"] == "user"]
    assert "titanic" in demo_ids
    assert "titanic_loopdemo" not in demo_ids
    assert len(mine) == 1
    assert mine[0]["case_id"] == "widget_labels"

    bob_list = client.get("/api/cases", headers=bob).json()
    assert all(c["kind"] == "demo" for c in bob_list)

    got = client.get("/api/cases/widget_labels", headers=alice).json()
    assert got["inspect"]["runnable"] is True

    bob_get = client.get("/api/cases/widget_labels", headers=bob)
    assert bob_get.status_code == 404

    confirmed = client.put(
        "/api/cases/widget_labels",
        json={
            "problem_type": "classification",
            "target_column": "label",
            "id_column": "rec_id",
            "file_roles": {"holdout.csv": "holdout"},
            "mark_ready": True,
        },
        headers=alice,
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["status"] == "ready"
    yaml_text = next((tmp_path / "users").rglob("case.yaml")).read_text()
    assert "sample_submission_csv" not in yaml_text
    assert "holdout" in yaml_text
    assert "widgets.csv" in yaml_text


def test_path_traversal_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/cases",
        data={"title": "x", "case_id": "../../etc/passwd", "problem_statement": "no"},
        files=[("files", ("ok.csv", b"a,b\n1,2\n", "text/csv"))],
        headers=headers,
    )
    assert resp.status_code == 400

    resp = client.post(
        "/api/cases",
        data={"title": "Safe", "problem_statement": "Predict a from b."},
        files=[("files", ("../../etc/passwd.csv", b"a,b\n1,2\n", "text/csv"))],
        headers=headers,
    )
    assert resp.status_code == 201
    stored = list((tmp_path / "users").rglob("*.csv"))
    assert stored
    assert all(".." not in p.parts for p in stored)
    assert all(p.name.endswith(".csv") for p in stored)


def test_reserved_demo_id_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    resp = client.post(
        "/api/cases",
        data={"title": "Titanic clone", "case_id": "titanic", "problem_statement": "no"},
        files=[("files", ("ok.csv", b"a,b\n1,2\n", "text/csv"))],
        headers=headers,
    )
    assert resp.status_code == 400


def test_clustering_marked_unsupported(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    created = client.post(
        "/api/cases",
        data={"title": "Segments", "problem_statement": "Find groups of widgets."},
        files=[("files", ("ok.csv", b"rec_id,note\n1,x\n", "text/csv"))],
        headers=headers,
    )
    assert created.status_code == 201
    cid = created.json()["case_id"]
    resp = client.put(
        f"/api/cases/{cid}",
        json={"problem_type": "clustering", "mark_ready": True},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "unsupported"
    assert resp.json().get("unsupported_reason")


def test_delete_is_owner_only(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    alice = register(client, "alice")
    bob = register(client, "bob")
    created = client.post(
        "/api/cases",
        data={"title": "Mine", "problem_statement": "Predict label."},
        files=[("files", ("ok.csv", b"rec_id,label\n1,a\n", "text/csv"))],
        headers=alice,
    )
    cid = created.json()["case_id"]
    assert client.delete(f"/api/cases/{cid}", headers=bob).status_code == 404
    assert client.delete(f"/api/cases/{cid}", headers=alice).status_code == 204
    assert client.get(f"/api/cases/{cid}", headers=alice).status_code == 404


def test_reinspect_after_replace(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    created = client.post(
        "/api/cases",
        data={"title": "Swap", "problem_statement": "Predict label."},
        files=[("files", ("old.csv", b"rec_id,label\n1,a\n", "text/csv"))],
        headers=headers,
    )
    cid = created.json()["case_id"]
    resp = client.post(
        f"/api/cases/{cid}/inspect",
        files=[("files", ("new.csv", b"rec_id,note,label\n1,hello,a\n2,world,b\n", "text/csv"))],
        headers=headers,
    )
    assert resp.status_code == 200
    names = [f["original_filename"] for f in resp.json()["inspect"]["files"]]
    assert names == ["new.csv"]


def test_create_assigns_train_named_file_not_alphabetical_test(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    headers = register(client)
    train_csv = "UserName,OriginalTweet,Sentiment\n1,hello,Positive\n"
    test_csv = "UserName,OriginalTweet,Sentiment\n9,holdout,Negative\n"
    resp = client.post(
        "/api/cases",
        data={
            "title": "covid",
            "problem_statement": "Classify sentiment expressed in a status.",
        },
        files=[
            ("files", ("Corona_NLP_test.csv", test_csv.encode("utf-8"), "text/csv")),
            ("files", ("Corona_NLP_train.csv", train_csv.encode("utf-8"), "text/csv")),
        ],
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    yaml_text = next((tmp_path / "users").rglob("case.yaml")).read_text()
    assert "Corona_NLP_train.csv" in yaml_text
    train_line = [ln for ln in yaml_text.splitlines() if ln.strip().startswith("train_csv:")]
    assert train_line
    assert "Corona_NLP_train.csv" in train_line[0]
    assert "Corona_NLP_test.csv" not in train_line[0]
