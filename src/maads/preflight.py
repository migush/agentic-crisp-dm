"""Deterministic case readiness checks — zero LLM tokens."""
from __future__ import annotations

from pathlib import Path

from maads.config import CaseConfig, primary_train_csv, source_locations


def _data_dir_segment(path: Path) -> str | None:
    """Return the path segment immediately under ``data/``, if any.

    Hosted user paths (``data/users/<id>/…``) are skipped — case-id alignment
    applies to demo/operator layouts ``data/<case_id>/…`` only.
    """
    parts = path.resolve().parts
    try:
        idx = parts.index("data")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    segment = parts[idx + 1]
    if segment == "users":
        return None
    return segment


def _check_readable_csv(path: Path, label: str) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        errors.append(f"{label} not found or not a file: {path}")
        return errors
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.read(1)
    except OSError as exc:
        errors.append(f"{label} not readable: {path} ({exc})")
    return errors


def _check_case_id_alignment(path: Path, case_id: str, label: str) -> list[str]:
    segment = _data_dir_segment(path)
    if segment is None:
        return []
    if segment != case_id:
        return [
            f"{label} path uses data/{segment}/ but case_id is {case_id!r}; "
            f"demo inputs must live at data/{case_id}/ (no hyphen aliases)"
        ]
    return []


def _peek_target_column(train: Path, target: str) -> list[str]:
    if not target:
        return []
    try:
        import pandas as pd

        header = pd.read_csv(train, nrows=0)
    except Exception as exc:
        return [f"train CSV could not be opened for target peek: {train} ({exc})"]
    if target not in header.columns:
        return [f"target_column {target!r} not present in train header: {train}"]
    return []


def preflight_case(config: CaseConfig) -> list[str]:
    """Return human-readable errors; empty list means the case is launchable.

    Checks file existence/readability, ``data/<case_id>/`` name alignment for
    demo layouts, and (when ``target_column`` is set) that the column appears
    in the train header. Does not invent path aliases or download data.
    """
    errors: list[str] = []
    case_id = config.case_id
    data = config.data

    try:
        train_s = primary_train_csv(data)
    except ValueError as exc:
        return [str(exc)]

    train = Path(train_s)
    errors.extend(_check_readable_csv(train, "train"))
    errors.extend(_check_case_id_alignment(train, case_id, "train"))

    if data.test_csv:
        test = Path(data.test_csv)
        errors.extend(_check_readable_csv(test, "test"))
        errors.extend(_check_case_id_alignment(test, case_id, "test"))

    if data.sample_submission_csv:
        sample = Path(data.sample_submission_csv)
        errors.extend(_check_readable_csv(sample, "sample_submission"))
        errors.extend(_check_case_id_alignment(sample, case_id, "sample_submission"))

    for src in source_locations(data):
        src_path = Path(src)
        if src_path.resolve() == train.resolve():
            continue
        if data.test_csv and src_path.resolve() == Path(data.test_csv).resolve():
            continue
        if data.sample_submission_csv and src_path.resolve() == Path(
            data.sample_submission_csv
        ).resolve():
            continue
        if not src_path.exists():
            errors.append(f"configured source path missing: {src_path}")

    if not errors and train.is_file():
        errors.extend(_peek_target_column(train, config.target_column))

    return errors
