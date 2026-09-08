"""Create, inspect, and manage per-user CRISP-DM cases (raw uploads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import paths, user_cases
from .inspect_csv import problem_type_runnable
from .routes_auth import require_user

router = APIRouter(prefix="/api/cases", tags=["cases"])


class ConfirmCaseRequest(BaseModel):
    display_name: str | None = None
    problem_statement: str | None = None
    problem_type: str | None = None
    target_column: str | None = None
    id_column: str | None = None
    evaluation_metric: str | None = None
    file_roles: dict[str, str] = Field(default_factory=dict)
    mark_ready: bool = False


class CaseListItem(BaseModel):
    kind: str
    case_id: str
    display_name: str
    status: str
    updated_at: str | None = None
    problem_type: str | None = None


def _http_value_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _yaml_fields(user_id: int, case_id: str) -> dict[str, Any]:
    yaml_path = user_cases.case_yaml_path(user_id, case_id)
    if not yaml_path.is_file():
        return {}
    import yaml

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return {
        "problem_statement": raw.get("problem_statement") or "",
        "problem_type": raw.get("problem_type") or "",
        "target_column": raw.get("target_column") or "",
        "id_column": raw.get("id_column") or "",
        "evaluation_metric": raw.get("evaluation_metric") or "",
    }


def _save_uploads(
    user_id: int,
    case_id: str,
    files: list[UploadFile],
    *,
    replace: bool = False,
) -> dict[str, str]:
    if len(files) > user_cases.MAX_FILES:
        raise HTTPException(status_code=400, detail=f"At most {user_cases.MAX_FILES} files per case.")
    dest = user_cases.raw_dir(user_id, case_id)
    dest.mkdir(parents=True, exist_ok=True)
    if replace:
        for existing in dest.iterdir():
            if existing.is_file():
                existing.unlink()
    original_names: dict[str, str] = {} if replace else user_cases.load_original_names(user_id, case_id)
    used: set[str] = {p.name for p in dest.iterdir() if p.is_file()}
    for upload in files:
        if not upload.filename:
            continue
        data = upload.file.read()
        if len(data) > user_cases.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename} exceeds the {user_cases.MAX_UPLOAD_BYTES // (1024 * 1024)} MB per-file limit.",
            )
        stored = user_cases.safe_filename(upload.filename)
        if stored in used:
            stem = Path(stored).stem
            suffix = Path(stored).suffix
            n = 2
            while f"{stem}_{n}{suffix}" in used:
                n += 1
            stored = f"{stem}_{n}{suffix}"
        (dest / stored).write_bytes(data)
        used.add(stored)
        original_names[stored] = Path(upload.filename).name
    return original_names


@router.get("", response_model=list[CaseListItem])
def list_cases(user_id: int = Depends(require_user)) -> list[CaseListItem]:
    items: list[CaseListItem] = []
    for cid in sorted(paths.public_demo_case_ids()):
        items.append(
            CaseListItem(
                kind="demo",
                case_id=cid,
                display_name=cid,
                status="ready",
                updated_at=None,
                problem_type=None,
            )
        )
    for row in user_cases.list_rows(user_id):
        fields = _yaml_fields(user_id, row["case_id"])
        items.append(
            CaseListItem(
                kind="user",
                case_id=row["case_id"],
                display_name=row["display_name"],
                status=row["status"],
                updated_at=row["updated_at"],
                problem_type=fields.get("problem_type") or None,
            )
        )
    return items


@router.post("", status_code=201)
def create_case(
    title: str = Form(...),
    problem_statement: str = Form(""),
    case_id: str | None = Form(default=None),
    problem_type: str = Form("classification"),
    target_column: str = Form(""),
    id_column: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    document: UploadFile | None = File(default=None),
    user_id: int = Depends(require_user),
) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required.")
    reserved = paths.known_case_ids()
    if case_id and case_id.strip():
        try:
            cid = user_cases.validate_case_id(case_id)
        except ValueError as exc:
            raise _http_value_error(exc) from exc
        if cid in reserved:
            raise HTTPException(status_code=400, detail="case_id is reserved for a bundled demo case.")
        if user_cases.get_row(user_id, cid) is not None:
            raise HTTPException(status_code=409, detail="You already have a case with this id.")
    else:
        try:
            cid = user_cases.unique_case_id(user_id, user_cases.slug_from_title(title), reserved)
        except ValueError as exc:
            raise _http_value_error(exc) from exc

    uploads = list(files or [])
    if document is not None and document.filename:
        uploads.append(document)
    if not uploads:
        raise HTTPException(status_code=400, detail="Upload at least one CSV.")

    original_names = _save_uploads(user_id, cid, uploads, replace=True)
    inspect = user_cases.refresh_inspect(user_id, cid)
    user_cases.write_case_yaml(
        user_id,
        cid,
        display_name=title,
        problem_statement=problem_statement.strip(),
        problem_type=problem_type.strip() or "classification",
        target_column=target_column.strip(),
        id_column=id_column.strip(),
        evaluation_metric="",
        file_roles={},
        original_names=original_names,
    )
    status = user_cases.decide_status(
        problem_type=problem_type,
        inspect=inspect,
        mark_ready=False,
    )
    user_cases.upsert_row(user_id, cid, title, status)
    row = user_cases.get_row(user_id, cid)
    return {
        "kind": "user",
        "case_id": cid,
        "display_name": title,
        "status": status,
        "updated_at": row["updated_at"] if row else None,
        "inspect": inspect,
        **_yaml_fields(user_id, cid),
    }


@router.get("/{case_id}")
def get_case(case_id: str, user_id: int = Depends(require_user)) -> dict[str, Any]:
    row = user_cases.get_row(user_id, case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such case.")
    inspect = user_cases.load_inspect_report(user_id, case_id)
    return {
        "kind": "user",
        "case_id": row["case_id"],
        "display_name": row["display_name"],
        "status": row["status"],
        "updated_at": row["updated_at"],
        "inspect": inspect,
        "file_roles": user_cases.load_file_roles(user_id, case_id),
        **_yaml_fields(user_id, case_id),
    }


@router.put("/{case_id}")
def update_case(
    case_id: str,
    body: ConfirmCaseRequest,
    user_id: int = Depends(require_user),
) -> dict[str, Any]:
    row = user_cases.get_row(user_id, case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such case.")
    fields = _yaml_fields(user_id, case_id)
    display_name = (body.display_name or row["display_name"]).strip()
    problem_statement = (
        body.problem_statement if body.problem_statement is not None else fields.get("problem_statement", "")
    )
    problem_type = (
        body.problem_type if body.problem_type is not None else fields.get("problem_type", "classification")
    ).strip()
    target_column = body.target_column if body.target_column is not None else fields.get("target_column", "")
    id_column = body.id_column if body.id_column is not None else fields.get("id_column", "")
    evaluation_metric = (
        body.evaluation_metric if body.evaluation_metric is not None else fields.get("evaluation_metric", "")
    )
    roles = body.file_roles or user_cases.load_file_roles(user_id, case_id)
    original_names = user_cases.load_original_names(user_id, case_id)
    user_cases.write_case_yaml(
        user_id,
        case_id,
        display_name=display_name,
        problem_statement=problem_statement,
        problem_type=problem_type or "classification",
        target_column=(target_column or "").strip(),
        id_column=(id_column or "").strip(),
        evaluation_metric=(evaluation_metric or "").strip(),
        file_roles=roles,
        original_names=original_names,
    )
    inspect = user_cases.load_inspect_report(user_id, case_id)
    status = user_cases.decide_status(
        problem_type=problem_type,
        inspect=inspect,
        mark_ready=body.mark_ready,
    )
    user_cases.upsert_row(user_id, case_id, display_name, status)
    payload = get_case(case_id, user_id)
    if status == "unsupported":
        _ok, reason = problem_type_runnable(problem_type)
        payload["unsupported_reason"] = (
            reason or inspect.get("unsupported_reason") or "This case is not runnable in V1."
        )
    return payload


@router.post("/{case_id}/inspect")
def reinspect_case(
    case_id: str,
    files: list[UploadFile] = File(default=[]),
    user_id: int = Depends(require_user),
) -> dict[str, Any]:
    row = user_cases.get_row(user_id, case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such case.")
    original_names = user_cases.load_original_names(user_id, case_id)
    incoming = [f for f in files if f.filename]
    if incoming:
        original_names.update(_save_uploads(user_id, case_id, incoming, replace=True))
        fields = _yaml_fields(user_id, case_id)
        user_cases.write_case_yaml(
            user_id,
            case_id,
            display_name=row["display_name"],
            problem_statement=fields.get("problem_statement", ""),
            problem_type=fields.get("problem_type", "classification"),
            target_column=fields.get("target_column", ""),
            id_column=fields.get("id_column", ""),
            evaluation_metric=fields.get("evaluation_metric", ""),
            file_roles={},
            original_names=original_names,
        )
    inspect = user_cases.refresh_inspect(user_id, case_id)
    user_cases.upsert_row(user_id, case_id, row["display_name"], row["status"])
    return {"case_id": case_id, "inspect": inspect}


@router.delete("/{case_id}", status_code=204)
def delete_case(case_id: str, user_id: int = Depends(require_user)) -> None:
    if not user_cases.delete_case(user_id, case_id):
        raise HTTPException(status_code=404, detail="No such case.")
