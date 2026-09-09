"""Per-user case directories, YAML drafts, and SQLite rows."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .db import get_conn
from .inspect_csv import inspect_directory, problem_type_runnable

CASE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}$")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_FILES = 12

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_case_id(raw: str) -> str:
    return (raw or "").strip().lower()


def validate_case_id(case_id: str) -> str:
    cid = sanitize_case_id(case_id)
    if not CASE_ID_RE.match(cid):
        raise ValueError(
            "case_id must match ^[a-z][a-z0-9_]{1,62}$ (start with a letter, then letters, digits, underscore)."
        )
    if ".." in cid or "/" in cid or "\\" in cid:
        raise ValueError("case_id must not contain path segments.")
    return cid


def slug_from_title(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (title or "").lower()).strip("_")
    if not s:
        s = "case"
    if not s[0].isalpha():
        s = "case_" + s
    return s[:63]


def safe_filename(name: str) -> str:
    base = Path(name or "upload.csv").name
    base = base.replace("\x00", "")
    cleaned = _UNSAFE_NAME.sub("_", base).strip("._") or "upload.csv"
    if cleaned in {".", ".."}:
        cleaned = "upload.csv"
    return cleaned[:180]


def user_case_dir(user_id: int, case_id: str) -> Path:
    return paths.user_cases_root(user_id) / case_id


def raw_dir(user_id: int, case_id: str) -> Path:
    return user_case_dir(user_id, case_id) / "raw"


def case_yaml_path(user_id: int, case_id: str) -> Path:
    return user_case_dir(user_id, case_id) / "case.yaml"


def inspect_json_path(user_id: int, case_id: str) -> Path:
    return user_case_dir(user_id, case_id) / "inspect.json"


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def default_success(problem_type: str, metric: str | None) -> dict[str, Any]:
    pt = (problem_type or "classification").lower()
    if metric:
        m = metric
    elif "regress" in pt:
        m = "rmse"
    else:
        m = "accuracy"
    direction = "minimize" if m in {"rmse", "rmse_log", "mae", "mse"} else "maximize"
    return {"metric": m, "threshold": 0.0, "direction": direction}


def write_case_yaml(
    user_id: int,
    case_id: str,
    *,
    display_name: str,
    problem_statement: str,
    problem_type: str,
    target_column: str,
    id_column: str,
    evaluation_metric: str,
    file_roles: dict[str, str],
    original_names: dict[str, str],
) -> Path:
    raw = raw_dir(user_id, case_id)
    sources: list[dict[str, Any]] = []
    if raw.is_dir():
        for path in sorted(raw.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            rel = repo_relative(path)
            role = file_roles.get(path.name) or file_roles.get(original_names.get(path.name, ""))
            entry: dict[str, Any] = {
                "path": rel,
                "original_filename": original_names.get(path.name, path.name),
            }
            if role:
                entry["role"] = role
            sources.append(entry)

    from maads.schema_inference import infer_source_paths

    inferred = infer_source_paths(sources)
    train = inferred["train"]
    test = inferred["test"]
    sample = inferred["sample_submission"]

    pt = problem_type or "classification"
    metric = evaluation_metric or ("rmse" if "regress" in pt.lower() else "accuracy")
    payload: dict[str, Any] = {
        "case_id": case_id,
        "kaggle_competition": "",
        "problem_statement": problem_statement,
        "problem_type": pt,
        "target_column": target_column,
        "id_column": id_column,
        "evaluation_metric": metric,
        "data": {
            "sources": sources,
        },
        "feature_hints": {},
        "class_labels": {},
        "success_criterion": default_success(pt, metric),
    }
    data = payload["data"]
    if train:
        data["train_csv"] = train
    if test:
        data["test_csv"] = test
    if sample:
        data["sample_submission_csv"] = sample

    dest = case_yaml_path(user_id, case_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return dest


def load_original_names(user_id: int, case_id: str) -> dict[str, str]:
    yaml_path = case_yaml_path(user_id, case_id)
    if not yaml_path.is_file():
        return {}
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names: dict[str, str] = {}
    for src in (raw.get("data") or {}).get("sources") or []:
        path = src.get("path") or ""
        orig = src.get("original_filename")
        if path and orig:
            names[Path(path).name] = orig
    return names


def load_file_roles(user_id: int, case_id: str) -> dict[str, str]:
    yaml_path = case_yaml_path(user_id, case_id)
    if not yaml_path.is_file():
        return {}
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    roles: dict[str, str] = {}
    for src in (raw.get("data") or {}).get("sources") or []:
        path = src.get("path") or ""
        role = src.get("role")
        if path and role:
            roles[Path(path).name] = role
    return roles


def save_inspect_report(user_id: int, case_id: str, report: dict[str, Any]) -> None:
    import json

    dest = inspect_json_path(user_id, case_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_inspect_report(user_id: int, case_id: str) -> dict[str, Any]:
    import json

    dest = inspect_json_path(user_id, case_id)
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    report = inspect_directory(raw_dir(user_id, case_id))
    save_inspect_report(user_id, case_id, report)
    return report


def refresh_inspect(user_id: int, case_id: str) -> dict[str, Any]:
    report = inspect_directory(raw_dir(user_id, case_id))
    save_inspect_report(user_id, case_id, report)
    return report


def decide_status(
    *,
    problem_type: str,
    inspect: dict[str, Any],
    mark_ready: bool,
) -> str:
    if not mark_ready:
        return "draft"
    ok_type, reason = problem_type_runnable(problem_type)
    if not ok_type:
        return "unsupported"
    if not inspect.get("runnable"):
        return "unsupported"
    return "ready"


def upsert_row(
    user_id: int,
    case_id: str,
    display_name: str,
    status: str,
) -> None:
    now = utc_now()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM user_cases WHERE user_id = ? AND case_id = ?",
            (user_id, case_id),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE user_cases SET display_name = ?, status = ?, updated_at = ? WHERE id = ?",
                (display_name, status, now, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO user_cases (user_id, case_id, display_name, status, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, case_id, display_name, status, now),
            )


def get_row(user_id: int, case_id: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, user_id, case_id, display_name, status, updated_at "
            "FROM user_cases WHERE user_id = ? AND case_id = ?",
            (user_id, case_id),
        ).fetchone()


def list_rows(user_id: int) -> list:
    with get_conn() as conn:
        return conn.execute(
            "SELECT case_id, display_name, status, updated_at "
            "FROM user_cases WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()


def delete_case(user_id: int, case_id: str) -> bool:
    import shutil

    row = get_row(user_id, case_id)
    if row is None:
        return False
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM user_cases WHERE user_id = ? AND case_id = ?",
            (user_id, case_id),
        )
    dest = user_case_dir(user_id, case_id)
    if dest.exists():
        shutil.rmtree(dest)
    return True


def unique_case_id(user_id: int, desired: str, reserved: frozenset[str]) -> str:
    base = validate_case_id(desired)
    candidate = base
    n = 2
    while candidate in reserved or get_row(user_id, candidate) is not None:
        suffix = f"_{n}"
        candidate = (base[: 63 - len(suffix)] + suffix)
        n += 1
        if n > 999:
            raise ValueError("Could not allocate a unique case_id.")
    return candidate
