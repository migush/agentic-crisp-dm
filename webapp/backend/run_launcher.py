"""Launches an existing maads CRISP-DM pipeline run on behalf of a logged-in user.

Security-critical: the user's decrypted provider API key must reach the
subprocess as an env var and NEVER touch disk, logs, or the tasks/DB tables.
Decryption happens just before spawning the subprocess; the plaintext lives
only in this process's memory and the child's env block, both torn down when
the run finishes.
"""

from __future__ import annotations

import json
import os
import signal
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
DEFAULT_RUN_TIMEOUT_SEC = 60 * 60
SOFT_DEADLINE_BUFFER_SEC = 5 * 60
HOSTED_WALL_TIMEOUT_REASON = "hosted_wall_timeout"


def _run_dir_for_case(artifact_root: Path, case_name: str) -> Path | None:
    """The run directory the just-finished subprocess wrote.

    ``<case>/current`` is a plain text file holding the run id, not a symlink to
    a directory — treating it as one (the previous behaviour) yielded a path
    that never exists, so no spend was ever recorded. resolve_active_run_dir
    reads it correctly and falls back to the newest run dir.
    """
    return resolve_active_run_dir(artifact_root / case_name)


def run_timeout_sec() -> int:
    raw = os.getenv("WEBAPP_RUN_TIMEOUT_SEC", str(DEFAULT_RUN_TIMEOUT_SEC)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_RUN_TIMEOUT_SEC
    return value if value > 0 else DEFAULT_RUN_TIMEOUT_SEC


def child_deadline_sec(timeout_sec: int) -> int:
    """In-flow deadline: stop cleanly before the parent wall-clock kill."""
    if timeout_sec <= SOFT_DEADLINE_BUFFER_SEC:
        return max(timeout_sec - 30, 1)
    return timeout_sec - SOFT_DEADLINE_BUFFER_SEC


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
        _mark_failed(task_id, f"Unknown provider: {provider}", artifact_root=None, case_name=case_name)
        return

    artifact_root = paths.user_artifact_root(user_id)
    started_at = datetime.now(timezone.utc).isoformat()
    _update_status(task_id, "running", started_at=started_at)

    timeout_sec = run_timeout_sec()
    child_env = {**_subprocess_base_env(), env_var: decrypted_api_key, "MODEL": model_id}
    child_env["MAADS_RUN_DEADLINE_SEC"] = str(child_deadline_sec(timeout_sec))
    if provider == "ollama_cloud":
        child_env["OLLAMA_BASE_URL"] = OLLAMA_CLOUD_BASE_URL
    argv = [sys.executable, "-m", "maads", "run"]
    if config_path:
        argv.extend(["--config", config_path])
    else:
        argv.extend(["--case", case_name])
    argv.extend(["--model", model_id, "--artifact-dir", str(artifact_root)])
    timed_out = False
    try:
        result = _run_subprocess(argv, cwd=REPO_ROOT, env=child_env, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        result = None
    finally:
        # Best-effort scrub; child_env goes out of scope regardless, but make
        # the "never persisted" guarantee explicit rather than incidental.
        child_env.clear()

    finished_at = datetime.now(timezone.utc).isoformat()
    if timed_out:
        _mark_failed(
            task_id,
            f"Run timed out after {timeout_sec} seconds.",
            artifact_root=artifact_root,
            case_name=case_name,
            halt_reason=HOSTED_WALL_TIMEOUT_REASON,
            provider=provider,
            model_id=model_id,
            finished_at=finished_at,
        )
        return
    if result is None or result.returncode != 0:
        code = None if result is None else result.returncode
        _mark_failed(
            task_id,
            f"maads run exited {code}",
            artifact_root=artifact_root,
            case_name=case_name,
            provider=provider,
            model_id=model_id,
            finished_at=finished_at,
        )
        return

    _record_completion(
        task_id,
        artifact_root=artifact_root,
        case_name=case_name,
        provider=provider,
        model_id=model_id,
        finished_at=finished_at,
    )


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run the pipeline child; SIGTERM the process group on timeout, then kill."""
    popen_kwargs: dict = {
        "args": argv,
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(**popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            stdout, stderr = proc.communicate(timeout=5)
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name != "nt" and proc.pid:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    proc.terminate()


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name != "nt" and proc.pid:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    proc.kill()


def _subprocess_base_env() -> dict[str, str]:
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


def _annotate_run_timeout(run_dir: Path, reason: str) -> None:
    """Best-effort halt flags on the run the child already created."""
    payload: dict = {}
    status_path = run_dir / "status.json"
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["halted"] = True
    payload["halt_reason"] = payload.get("halt_reason") or reason
    payload["activity"] = reason
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, default=str)
        status_path.write_text(text, encoding="utf-8")
        derived = run_dir / "derived"
        if derived.is_dir():
            (derived / "status.json").write_text(text, encoding="utf-8")
    except OSError:
        return


def _mark_failed(
    task_id: int,
    reason: str,
    *,
    artifact_root: Path | None,
    case_name: str,
    halt_reason: str | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    finished_at: str | None = None,
) -> None:
    finished_at = finished_at or datetime.now(timezone.utc).isoformat()
    run_dir = _run_dir_for_case(artifact_root, case_name) if artifact_root is not None else None
    if run_dir is not None and halt_reason:
        _annotate_run_timeout(run_dir, halt_reason)
    with get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = 'failed', finished_at = ?, run_artifact_path = ? WHERE id = ?",
            (finished_at, str(run_dir) if run_dir else None, task_id),
        )
        if run_dir is not None and provider and model_id:
            _insert_token_spend(
                conn,
                task_id,
                run_dir=run_dir,
                provider=provider,
                model_id=model_id,
            )
    # `reason` is deliberately not persisted verbatim to the DB in this MVP to
    # avoid accidentally storing subprocess stderr (which could echo secrets
    # from a misbehaving dependency); surface it via server logs instead.
    print(f"[run_launcher] task {task_id} failed: {reason}", file=sys.stderr)


def _insert_token_spend(
    conn,
    task_id: int,
    *,
    run_dir: Path,
    provider: str,
    model_id: str,
) -> None:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    token_spend = state.get("token_spend", {}) or {}
    total_input = state.get("total_input_tokens", 0) or 0
    total_output = state.get("total_output_tokens", 0) or 0
    run_total = sum(token_spend.values()) or 1
    for agent, agent_total in token_spend.items():
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
    with get_conn() as conn:
        conn.execute(
            # Store the concrete run dir, not the `current` pointer, which the
            # user's next run of the same case would move out from under it.
            "UPDATE tasks SET status = 'completed', finished_at = ?, run_artifact_path = ? WHERE id = ?",
            (finished_at, str(run_dir) if run_dir else None, task_id),
        )
        if run_dir is not None:
            _insert_token_spend(
                conn,
                task_id,
                run_dir=run_dir,
                provider=provider,
                model_id=model_id,
            )
