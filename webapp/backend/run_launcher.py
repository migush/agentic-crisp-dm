"""Launches an existing maads CRISP-DM pipeline run on behalf of a logged-in user.

Security-critical: the user's decrypted provider API key must reach the
subprocess as an env var and NEVER touch disk, logs, or the tasks/DB tables.
Decryption happens just before spawning the subprocess; the plaintext lives
only in this process's memory and the child's env block, both torn down when
the run finishes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from maads.artifact_runs import resolve_active_run_dir
from maads.pricing import estimate_cost_usd

from . import paths
from .db import get_conn
from .hosted import OLLAMA_CLOUD_BASE_URL, PARENT_ENV_DENYLIST, PROVIDER_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_dir_for_case(artifact_root: Path, case_name: str) -> Path | None:
    """The run directory the just-finished subprocess wrote.

    ``<case>/current`` is a plain text file holding the run id, not a symlink to
    a directory — treating it as one (the previous behaviour) yielded a path
    that never exists, so no spend was ever recorded. resolve_active_run_dir
    reads it correctly and falls back to the newest run dir.
    """
    return resolve_active_run_dir(artifact_root / case_name)


def create_task(user_id: int, case_name: str, provider: str, model_id: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (user_id, case_name, provider, model_id, status) VALUES (?, ?, ?, ?, 'queued')",
            (user_id, case_name, provider, model_id),
        )
        return cur.lastrowid


def run_task(
    task_id: int,
    *,
    user_id: int,
    case_name: str,
    provider: str,
    model_id: str,
    decrypted_api_key: str,
    config_path: str | None = None,
) -> None:
    """Run synchronously (call from a FastAPI BackgroundTask). Never raises past logging."""
    env_var = PROVIDER_ENV_VAR.get(provider)
    if env_var is None:
        _mark_failed(task_id, f"Unknown provider: {provider}")
        return

    artifact_root = paths.user_artifact_root(user_id)
    started_at = datetime.now(timezone.utc).isoformat()
    _update_status(task_id, "running", started_at=started_at)

    child_env = {**_subprocess_base_env(), env_var: decrypted_api_key, "MODEL": model_id}
    if provider == "ollama_cloud":
        child_env["OLLAMA_BASE_URL"] = OLLAMA_CLOUD_BASE_URL
    argv = [sys.executable, "-m", "maads", "run"]
    if config_path:
        argv.extend(["--config", config_path])
    else:
        argv.extend(["--case", case_name])
    argv.extend(["--model", model_id, "--artifact-dir", str(artifact_root)])
    try:
        result = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60 * 60,
        )
    except subprocess.TimeoutExpired:
        _mark_failed(task_id, "Run timed out after 1 hour.")
        return
    finally:
        # Best-effort scrub; child_env goes out of scope regardless, but make
        # the "never persisted" guarantee explicit rather than incidental.
        child_env.clear()

    finished_at = datetime.now(timezone.utc).isoformat()
    if result.returncode != 0:
        _mark_failed(task_id, f"maads run exited {result.returncode}")
        return

    _record_completion(
        task_id,
        artifact_root=artifact_root,
        case_name=case_name,
        provider=provider,
        model_id=model_id,
        finished_at=finished_at,
    )


def _subprocess_base_env() -> dict[str, str]:
    import os

    # Deny-list the parent's own provider keys and Ollama host (e.g. a
    # developer's local .env) so a run can never silently fall back to them
    # instead of the user's own stored key.
    return {k: v for k, v in os.environ.items() if k not in PARENT_ENV_DENYLIST}


def _update_status(task_id: int, status: str, *, started_at: str | None = None) -> None:
    with get_conn() as conn:
        if started_at:
            conn.execute("UPDATE tasks SET status = ?, started_at = ? WHERE id = ?", (status, started_at, task_id))
        else:
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _mark_failed(task_id: int, reason: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = 'failed', finished_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), task_id),
        )
    # `reason` is deliberately not persisted verbatim to the DB in this MVP to
    # avoid accidentally storing subprocess stderr (which could echo secrets
    # from a misbehaving dependency); surface it via server logs instead.
    print(f"[run_launcher] task {task_id} failed: {reason}", file=sys.stderr)


def _record_completion(
    task_id: int,
    *,
    artifact_root: Path,
    case_name: str,
    provider: str,
    model_id: str,
    finished_at: str,
) -> None:
    run_dir = _run_dir_for_case(artifact_root, case_name)
    token_spend: dict[str, int] = {}
    total_input = total_output = 0
    if run_dir is not None and (run_dir / "state.json").is_file():
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        token_spend = state.get("token_spend", {})
        total_input = state.get("total_input_tokens", 0)
        total_output = state.get("total_output_tokens", 0)

    with get_conn() as conn:
        conn.execute(
            # Store the concrete run dir, not the `current` pointer, which the
            # user's next run of the same case would move out from under it.
            "UPDATE tasks SET status = 'completed', finished_at = ?, run_artifact_path = ? WHERE id = ?",
            (finished_at, str(run_dir) if run_dir else None, task_id),
        )
        for agent, agent_total in token_spend.items():
            # Per-agent input/output split isn't tracked yet (see state.py's
            # add_tokens) — attribute the run-level split proportionally so
            # per-agent $ estimates are still meaningful, not just $0/all-input.
            run_total = sum(token_spend.values()) or 1
            share = agent_total / run_total
            conn.execute(
                """
                INSERT INTO token_spend_events
                    (task_id, agent, provider, model_id, input_tokens, output_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    agent,
                    provider,
                    model_id,
                    round(total_input * share),
                    round(total_output * share),
                    estimate_cost_usd(model_id, round(total_input * share), round(total_output * share)),
                ),
            )
