"""Launches and reports on pipeline runs ("tasks") for the logged-in user.

The decrypted API key only ever appears in the request body of POST
/api/tasks (over HTTPS) and in run_launcher's subprocess env — never stored,
logged, or echoed back in any response.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from . import run_launcher
from .db import get_conn
from .openai_models import list_openai_chat_models
from .paths import known_case_ids
from .routes_auth import require_user

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

_HOSTED_PROVIDER = "openai"


class LaunchTaskRequest(BaseModel):
    case_name: str
    provider: str
    model_id: str
    decrypted_api_key: str


class TaskSummary(BaseModel):
    id: int
    case_name: str
    provider: str
    model_id: str
    status: str
    started_at: str | None
    finished_at: str | None


class TokenSpendEvent(BaseModel):
    agent: str
    provider: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@router.post("", response_model=TaskSummary, status_code=202)
def launch_task(
    body: LaunchTaskRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(require_user),
) -> TaskSummary:
    if body.provider != _HOSTED_PROVIDER:
        raise HTTPException(status_code=400, detail="Hosted tasks require provider=openai.")
    model_id = body.model_id.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")
    # case_name reaches both a filesystem path and the `maads run --case`
    # argument, so it must be one of the configs that actually exist rather
    # than whatever the client sent.
    if body.case_name not in known_case_ids():
        raise HTTPException(status_code=400, detail=f"Unknown case: {body.case_name}")

    live_ids = {entry["id"] for entry in list_openai_chat_models(body.decrypted_api_key)}
    if model_id not in live_ids:
        raise HTTPException(status_code=400, detail="model_id is not available for this API key.")

    task_id = run_launcher.create_task(user_id, body.case_name, _HOSTED_PROVIDER, model_id)
    background_tasks.add_task(
        run_launcher.run_task,
        task_id,
        user_id=user_id,
        case_name=body.case_name,
        provider=_HOSTED_PROVIDER,
        model_id=model_id,
        decrypted_api_key=body.decrypted_api_key,
    )
    return TaskSummary(
        id=task_id,
        case_name=body.case_name,
        provider=_HOSTED_PROVIDER,
        model_id=model_id,
        status="queued",
        started_at=None,
        finished_at=None,
    )


@router.get("", response_model=list[TaskSummary])
def list_tasks(user_id: int = Depends(require_user)) -> list[TaskSummary]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, case_name, provider, model_id, status, started_at, finished_at "
            "FROM tasks WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
    return [TaskSummary(**dict(row)) for row in rows]


@router.get("/{task_id}/spend", response_model=list[TokenSpendEvent])
def task_spend(task_id: int, user_id: int = Depends(require_user)) -> list[TokenSpendEvent]:
    with get_conn() as conn:
        owner = conn.execute("SELECT user_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if owner is None or owner["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="No such task.")
        rows = conn.execute(
            "SELECT agent, provider, model_id, input_tokens, output_tokens, cost_usd "
            "FROM token_spend_events WHERE task_id = ?",
            (task_id,),
        ).fetchall()
    return [TokenSpendEvent(**dict(row)) for row in rows]
