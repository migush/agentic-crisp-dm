"""The trace dashboard, mounted inside the account app.

Two things are under test here: that the dashboard is reachable at all (it was
not, once the account product claimed /api and / in the reverse proxy), and
that every one of its routes now serves only the calling account's runs.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest
import webapp.backend.db as db_module
from fastapi.testclient import TestClient

import webapp.backend.paths as paths_module


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("WEBAPP_ALLOW_DEV_SECRET", "1")
    monkeypatch.setenv("WEBAPP_INSECURE_COOKIES", "1")
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", tmp_path / "webapp.db")
    monkeypatch.setattr(
        paths_module, "user_artifact_root", lambda uid: tmp_path / "users" / str(uid) / "artifacts"
    )
    monkeypatch.setattr(paths_module, "demo_artifact_root", lambda: tmp_path / "demo")
    from webapp.backend.app import create_app

    return TestClient(create_app())


def register(client: TestClient, username: str) -> dict[str, str]:
    resp = client.post("/api/auth/register", json={"username": username})
    assert resp.status_code == 201
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def make_run(artifact_root, case_id: str, run_id: str = "r1", *, marker: str = "x") -> None:
    """Write the minimal on-disk shape the dashboard treats as a real run."""
    run_dir = artifact_root / case_id / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"case_id": case_id, "run_id": run_id, "phase": "complete", "marker": marker}),
        encoding="utf-8",
    )
    # `current` is a text file naming the run id, not a symlink to a directory.
    (artifact_root / case_id / "current").write_text(run_id, encoding="utf-8")


def test_dashboard_spa_and_api_are_reachable(client: TestClient) -> None:
    # /dashboard normalises to /dashboard/ so the SPA's relative asset URLs resolve.
    assert client.get("/dashboard", follow_redirects=False).status_code == 307
    assert client.get("/dashboard/api/openapi.json").status_code == 200


def test_dashboard_api_requires_authentication(client: TestClient) -> None:
    assert client.get("/dashboard/api/cases").status_code == 401
    assert client.get("/dashboard/api/health").status_code == 401


def test_session_cookie_authenticates_downloads(client: TestClient) -> None:
    # Set by /login and /register; the only way <a download> links can authenticate.
    resp = client.post("/api/auth/register", json={"username": "carol"})
    assert resp.status_code == 201
    assert client.get("/dashboard/api/cases").status_code == 200

    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/dashboard/api/cases").status_code == 401


def test_each_account_sees_only_its_own_runs(client: TestClient, tmp_path) -> None:
    alice = register(client, "alice")
    bob = register(client, "bob")

    make_run(paths_module.user_artifact_root(1), "titanic", marker="alice")
    make_run(paths_module.user_artifact_root(2), "house_prices", marker="bob")

    alice_cases = {c["case_id"] for c in client.get("/dashboard/api/cases", headers=alice).json()}
    bob_cases = {c["case_id"] for c in client.get("/dashboard/api/cases", headers=bob).json()}
    assert alice_cases == {"titanic"}
    assert bob_cases == {"house_prices"}

    # Bob's case is not merely hidden from the listing — it is unreachable.
    assert client.get("/dashboard/api/cases/house_prices/status", headers=alice).status_code == 404
    assert client.get("/dashboard/api/cases/house_prices/state", headers=alice).status_code == 404

    # And a same-named case resolves to each account's own copy, not a shared one.
    make_run(paths_module.user_artifact_root(2), "titanic", marker="bob")
    assert client.get("/dashboard/api/cases/titanic/status", headers=alice).json()["marker"] == "alice"
    assert client.get("/dashboard/api/cases/titanic/status", headers=bob).json()["marker"] == "bob"


def test_demo_cases_are_shared_and_flagged_read_only(client: TestClient) -> None:
    headers = register(client, "dana")
    make_run(paths_module.demo_artifact_root(), "titanic", marker="demo")
    make_run(paths_module.user_artifact_root(1), "house_prices", marker="mine")

    cases = {c["case_id"]: c["read_only"] for c in client.get("/dashboard/api/cases", headers=headers).json()}
    assert cases == {"titanic": True, "house_prices": False}


def test_own_case_shadows_a_demo_of_the_same_name(client: TestClient) -> None:
    headers = register(client, "erin")
    make_run(paths_module.demo_artifact_root(), "titanic", marker="demo")
    make_run(paths_module.user_artifact_root(1), "titanic", marker="mine")

    cases = client.get("/dashboard/api/cases", headers=headers).json()
    assert [(c["case_id"], c["read_only"]) for c in cases] == [("titanic", False)]
    assert client.get("/dashboard/api/cases/titanic/status", headers=headers).json()["marker"] == "mine"


# Encoded so the HTTP client can't normalise the traversal away before it is
# sent — an attacker's client wouldn't normalise it either.
@pytest.mark.parametrize("case_id", ["..%2F..%2Fetc", "%2E%2E", "a%2Fb"])
def test_case_id_cannot_escape_the_account_root(client: TestClient, case_id: str) -> None:
    headers = register(client, "frank")
    resp = client.get(f"/dashboard/api/cases/{case_id}/state", headers=headers)
    assert resp.status_code == 404


def test_case_scope_rejects_traversal_ids() -> None:
    from fastapi import HTTPException

    from maads.dashboard.deps import CaseScope

    scope = CaseScope(write_root=paths_module.user_artifact_root(1))
    for bad in ["..", ".", "", "../other", "a/b", "a\\b"]:
        with pytest.raises(HTTPException) as exc:
            scope.case_path(bad)
        assert exc.value.status_code == 404


def test_dashboard_launch_is_refused_when_hosted(client: TestClient) -> None:
    headers = register(client, "gina")
    resp = client.post("/dashboard/api/run", json={"case_id": "titanic"}, headers=headers)
    # A hosted run needs the caller's provider key, which only exists decrypted
    # in their browser — launches go through POST /api/tasks instead.
    assert resp.status_code == 403

    client.cookies.clear()  # register() also set a session cookie
    assert client.post("/dashboard/api/run", json={"case_id": "titanic"}).status_code == 401


def _write_user_run_with_state(artifact_root, case_id: str, run_id: str) -> None:
    """Write a completed run with final_state.json but no prebuilt reports."""
    from maads.artifact_paths import ensure_run_layout
    from maads.config import load_case_config
    from maads.paths import resolve_path
    from maads.state import CrispDMState, ModelRun, Phase

    run_dir = artifact_root / case_id / "runs" / run_id
    ensure_run_layout(run_dir, run_id=run_id, case_id=case_id)
    (run_dir / "status.json").write_text(
        json.dumps({"case_id": case_id, "run_id": run_id, "phase": 6, "halted": True}),
        encoding="utf-8",
    )
    (artifact_root / case_id / "current").write_text(run_id, encoding="utf-8")
    cfg = load_case_config(resolve_path(f"configs/{case_id}.yaml"))
    state = CrispDMState.from_config(cfg)
    state.halted = True
    state.phase = Phase.DEPLOYMENT
    state.substep = "6.4"
    state.halt_reason = "completed phase 6"
    state.md.chosen_model = ModelRun(
        technique="test_model",
        cv_score=0.9,
        cv_std=0.01,
        assessment="selected",
    )
    metric = cfg.success_criterion.metric
    state.ev.assessment_of_dm_results = {
        "metric": metric,
        "achieved_score": 0.9,
        "threshold": cfg.success_criterion.threshold,
        "success_criterion_met": True,
    }
    state.ev.decision = "deploy"
    sub = run_dir / "submission.csv"
    sub.write_text("id,pred\n1,0\n", encoding="utf-8")
    state.dep.submission_path = str(sub)
    (run_dir / "final_report.md").write_text("# report\n", encoding="utf-8")
    state.dep.final_report_path = str(run_dir / "final_report.md")
    (run_dir / "final_state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")


def test_cookie_downloads_user_workbook_and_handoff(client: TestClient) -> None:
    # <a download> cannot send Authorization; hosted links rely on maads_session.
    resp = client.post("/api/auth/register", json={"username": "hank"})
    assert resp.status_code == 201

    run_id = "b47ea0aa-e249-4f15-9d8f-0c65b2055b84"
    _write_user_run_with_state(paths_module.user_artifact_root(1), "titanic", run_id)

    notebook = client.get(
        f"/dashboard/api/cases/titanic/reports/case_workbook.ipynb?run_id={run_id}",
    )
    assert notebook.status_code == 200
    assert "nbformat" in notebook.text

    handoff = client.get(
        f"/dashboard/api/cases/titanic/reports/handoff_standard.zip?run_id={run_id}",
    )
    assert handoff.status_code == 200
    assert handoff.headers["content-type"] == "application/zip"
    assert zipfile.is_zipfile(BytesIO(handoff.content))

    client.cookies.clear()
    assert client.get(
        f"/dashboard/api/cases/titanic/reports/case_workbook.ipynb?run_id={run_id}",
    ).status_code == 401
    assert client.get(
        f"/dashboard/api/cases/titanic/reports/handoff_standard.zip?run_id={run_id}",
    ).status_code == 401
