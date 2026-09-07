"""run_launcher writes into the launching account's own artifact root, and
reads back the run it just produced.

Both were broken: runs went to the shared repo-wide artifacts/ tree regardless
of who started them, and completion looked for state.json under
``<case>/current/state.json`` — but ``current`` is a text file holding the run
id, not a directory, so that path never existed and no spend was ever recorded.
"""

from __future__ import annotations

import json
import subprocess

import pytest
import webapp.backend.db as db_module
import webapp.backend.paths as paths_module
import webapp.backend.run_launcher as run_launcher


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "webapp.db"
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", path)
    db_module.init_db(path)
    return path


def make_user(db_path, user_id: int) -> None:
    """tasks.user_id is a foreign key, so the account row has to exist first."""
    with db_module.get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO users (id, username, created_at) VALUES (?, ?, '2026-01-01')",
            (user_id, f"user{user_id}"),
        )


def seed_finished_run(artifact_root, case_id: str, run_id: str, state: dict) -> None:
    """Lay out a completed run exactly as `maads run --artifact-dir` leaves it."""
    run_dir = artifact_root / case_id / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (artifact_root / case_id / "current").write_text(run_id, encoding="utf-8")


def test_run_uses_the_launching_users_artifact_root(db_path, tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    root = tmp_path / "u7"
    monkeypatch.setattr(paths_module, "user_artifact_root", lambda uid: _mkdir(root))

    make_user(db_path, 7)
    task_id = run_launcher.create_task(7, "titanic", "openai", "gpt-4o")
    run_launcher.run_task(
        task_id, user_id=7, case_name="titanic", provider="openai",
        model_id="gpt-4o", decrypted_api_key="sk-secret",
    )

    cmd = captured["cmd"]
    assert "--artifact-dir" in cmd
    assert cmd[cmd.index("--artifact-dir") + 1] == str(root)
    # The key travels only in the child env, never in the argv other processes can read.
    assert "sk-secret" not in " ".join(cmd)


def test_completion_records_spend_from_the_real_run_layout(db_path, tmp_path, monkeypatch):
    root = tmp_path / "u3"
    seed_finished_run(
        _mkdir(root),
        "titanic",
        "20260101T000000Z",
        {
            "token_spend": {"data_scientist": 300, "developer": 100},
            "total_input_tokens": 320,
            "total_output_tokens": 80,
        },
    )
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(paths_module, "user_artifact_root", lambda uid: root)

    make_user(db_path, 3)
    task_id = run_launcher.create_task(3, "titanic", "openai", "gpt-4o")
    run_launcher.run_task(
        task_id, user_id=3, case_name="titanic", provider="openai",
        model_id="gpt-4o", decrypted_api_key="sk-x",
    )

    with db_module.get_conn(db_path) as conn:
        task = conn.execute("SELECT status, run_artifact_path FROM tasks WHERE id = ?", (task_id,)).fetchone()
        events = conn.execute(
            "SELECT agent, input_tokens, output_tokens FROM token_spend_events WHERE task_id = ?",
            (task_id,),
        ).fetchall()

    assert task["status"] == "completed"
    # The concrete run dir, not the `current` pointer a later run would move.
    assert task["run_artifact_path"] == str(root / "titanic" / "runs" / "20260101T000000Z")

    by_agent = {row["agent"]: (row["input_tokens"], row["output_tokens"]) for row in events}
    assert by_agent == {"data_scientist": (240, 60), "developer": (80, 20)}


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
