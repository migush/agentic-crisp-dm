"""Deterministic source-role and column-role inference for hosted user cases.

Users may omit ``target_column``, ``id_column``, file roles, and ``feature_hints``.
Demos already set those in YAML; this module must not override a non-blank value.
Never branch on ``case_id``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_CSV_SUFFIXES = {".csv", ".tsv", ".txt"}

# Token boundaries so "contest.csv" is not treated as a test split.
_TRAIN_RE = re.compile(r"(?:^|[_\-.])train(?:$|[_\-.])", re.I)
_TEST_RE = re.compile(r"(?:^|[_\-.])(?:test|holdout|unlabelled|unlabeled)(?:$|[_\-.])", re.I)
_SAMPLE_RE = re.compile(r"(?:sample.?submission)|(?:^|[_\-.])(?:sample|submission)(?:$|[_\-.])", re.I)

_TARGET_NAME_SCORES: tuple[tuple[str, int], ...] = (
    ("target", 100),
    ("label", 90),
    ("sentiment", 90),
    ("survived", 85),
    ("saleprice", 85),
    ("polarity", 80),
    ("outcome", 70),
    ("class", 55),
    ("category", 50),
    ("rating", 50),
    ("price", 45),
    ("y", 40),
)
_DATEISH_RE = re.compile(r"date|time|timestamp|year|month|day|(?:^|_)at$", re.I)
_ID_NAME_RE = re.compile(r"(?:^|_)id$|userid|user_id|^id$", re.I)


def read_csv(path: str | Path):
    """Read a CSV trying common 8-bit encodings after UTF-8 (hosted uploads)."""
    import pandas as pd

    p = Path(path)
    last_err: Exception | None = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
    if last_err is not None:
        return pd.read_csv(p, encoding="latin-1")
    return pd.read_csv(p)


def filename_role(name: str) -> str | None:
    """Return train/test/sample_submission from a filename, or None."""
    stem = Path(name or "").stem
    if not stem:
        return None
    if _SAMPLE_RE.search(stem):
        return "sample_submission"
    if _TRAIN_RE.search(stem):
        return "train"
    if _TEST_RE.search(stem):
        return "test"
    return None


def _is_csv_path(path: str) -> bool:
    return Path(path).suffix.lower() in _CSV_SUFFIXES


def _src_fields(src: Any) -> tuple[str, str, str]:
    if isinstance(src, dict):
        path = str(src.get("path") or "")
        role = str(src.get("role") or "")
        orig = str(src.get("original_filename") or "")
    else:
        path = str(getattr(src, "path", "") or "")
        role = str(getattr(src, "role", "") or "")
        orig = str(getattr(src, "original_filename", "") or "")
    return path, role, orig


def _explicit_role(role: str) -> str | None:
    r = (role or "").strip().lower()
    if r in {"train", "labelled", "labeled", "primary"}:
        return "train"
    if r in {"test", "holdout", "unlabelled", "unlabeled"}:
        return "test"
    if r in {"sample_submission", "submission"}:
        return "sample_submission"
    return None


def infer_source_paths(sources: Sequence[Any]) -> dict[str, str | None]:
    """Pick train/test/sample paths from ``DataSource`` rows or dicts."""
    train = test = sample = None
    ranked: list[tuple[str, str | None]] = []
    for src in sources or []:
        path, role, orig = _src_fields(src)
        if not path:
            continue
        resolved = _explicit_role(role) or filename_role(orig or Path(path).name)
        ranked.append((path, resolved))
        if resolved == "train" and train is None:
            train = path
        elif resolved == "test" and test is None:
            test = path
        elif resolved == "sample_submission" and sample is None:
            sample = path
    if train is None:
        for path, role in ranked:
            if role in {"test", "sample_submission"}:
                continue
            if _is_csv_path(path):
                train = path
                break
    if train is None:
        for path, role in ranked:
            if role == "sample_submission":
                continue
            if _is_csv_path(path):
                train = path
                break
    return {"train": train, "test": test, "sample_submission": sample}


def _same_file(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a).as_posix() == Path(b).as_posix()


def reconcile_data_paths(data: Any) -> Any:
    """Correct train/test assignment when sources exist but roles were omitted.

    Does not invent a test file. Overrides ``train_csv`` when it is test-named
    and another source is train-named (alphabetical ``*_test.csv`` first).
    """
    sources = getattr(data, "sources", None) or []
    if not sources:
        return data
    inferred = infer_source_paths(sources)
    train = getattr(data, "train_csv", None)
    test = getattr(data, "test_csv", None)
    sample = getattr(data, "sample_submission_csv", None)
    inf_train = inferred["train"]
    inf_test = inferred["test"]
    inf_sample = inferred["sample_submission"]

    train_fn_role = filename_role(Path(train).name) if train else None
    inf_train_role = filename_role(Path(inf_train).name) if inf_train else None

    if inf_train:
        if not train:
            train = inf_train
        elif train_fn_role == "test" and inf_train_role == "train":
            train = inf_train
        elif train_fn_role == "test" and not _same_file(train, inf_train):
            train = inf_train
    if inf_test and not test and not _same_file(inf_test, train):
        test = inf_test
    if train and test and _same_file(train, test) and inf_train and not _same_file(inf_train, test):
        train = inf_train
    if inf_sample and not sample:
        sample = inf_sample
    return data.model_copy(
        update={
            "train_csv": train,
            "test_csv": test,
            "sample_submission_csv": sample,
        }
    )


def _norm_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (col or "").lower())


def _target_name_score(col: str) -> int:
    lname = _norm_name(col)
    best = 0
    for token, pts in _TARGET_NAME_SCORES:
        if token == "y":
            if lname == "y":
                best = max(best, pts)
            continue
        if lname == token or lname.endswith(token) or token in lname:
            best = max(best, pts)
    return best


def infer_column_roles(
    *,
    columns: Sequence[str],
    cardinality: dict[str, int],
    dtypes: dict[str, str],
    n_rows: int,
    missing: dict[str, int] | None = None,
    columns_only_in_train: Sequence[str] | None = None,
    problem_type: str = "classification",
) -> dict[str, str]:
    """Score a likely target and identifier from measured column stats."""
    missing = missing or {}
    only_train = set(columns_only_in_train or [])
    n_rows = max(int(n_rows or 0), 0)
    is_reg = "regress" in (problem_type or "").lower()
    ranked: list[tuple[int, str]] = []
    for col in columns:
        if not col:
            continue
        k = int(cardinality.get(col) or 0)
        if k <= 1:
            continue
        if n_rows and k == n_rows:
            continue
        if n_rows >= 20 and k >= int(0.95 * n_rows):
            continue
        score = _target_name_score(col)
        if col in only_train:
            score += 80
        dt = str(dtypes.get(col) or "")
        is_num = dt.lower().startswith(("int", "float", "uint"))
        miss = int(missing.get(col) or 0)
        if miss > 0.4 * max(n_rows, 1):
            score -= 40
        if _DATEISH_RE.search(col):
            score -= 35
        if is_reg:
            if is_num and k > 20:
                score += 40
        else:
            if 2 <= k <= 20:
                score += 40
            elif k <= 50:
                score += 15
        ranked.append((score, col))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    target = ""
    if ranked and ranked[0][0] >= 40:
        target = ranked[0][1]

    id_ranked: list[tuple[int, str]] = []
    for col in columns:
        if not col or col == target:
            continue
        k = int(cardinality.get(col) or 0)
        if not n_rows or k != n_rows:
            continue
        score = 10
        lname = col.lower()
        if _ID_NAME_RE.search(lname.replace("-", "_")):
            score += 50
        if "user" in lname:
            score += 10
        if str(dtypes.get(col) or "").lower().startswith(("int", "uint")):
            score += 20
        id_ranked.append((score, col))
    id_ranked.sort(key=lambda item: (-item[0], item[1]))
    id_column = ""
    if id_ranked and id_ranked[0][0] >= 20:
        id_column = id_ranked[0][1]
    return {"target": target, "id": id_column}


def infer_column_roles_from_profile(
    profile: dict[str, Any],
    problem_type: str = "classification",
) -> dict[str, str]:
    return infer_column_roles(
        columns=list(profile.get("columns") or []),
        cardinality=dict(profile.get("cardinality") or {}),
        dtypes=dict(profile.get("dtypes") or {}),
        n_rows=int(profile.get("n_rows") or 0),
        missing=dict(profile.get("missing") or {}),
        columns_only_in_train=list(profile.get("columns_only_in_train") or []),
        problem_type=problem_type,
    )


def infer_feature_hints_from_profile(
    profile: dict[str, Any],
    *,
    target: str,
    id_column: str,
) -> dict[str, Any]:
    """Fill empty hint keys: free-text first, weak categoricals, missing numerics."""
    columns = list(profile.get("columns") or [])
    card = dict(profile.get("cardinality") or {})
    dtypes = dict(profile.get("dtypes") or {})
    missing = dict(profile.get("missing") or {})
    n_rows = max(int(profile.get("n_rows") or 1), 1)
    reserved = {c for c in (target, id_column) if c}
    text_free: list[str] = []
    cat_weak: list[str] = []
    num_missing: list[str] = []
    for col in columns:
        if col in reserved:
            continue
        k = int(card.get(col) or 0)
        dt = str(dtypes.get(col) or "").lower()
        is_num = dt.startswith(("int", "float", "uint", "bool"))
        if is_num:
            if int(missing.get(col) or 0) > 0:
                num_missing.append(col)
            continue
        if k >= max(50, int(0.3 * n_rows)) or (
            n_rows >= 8 and k >= int(0.8 * n_rows)
        ):
            text_free.append(col)
        elif 2 <= k <= 50:
            cat_weak.append(col)
    text_free.sort(key=lambda c: -int(card.get(c) or 0))
    hints: dict[str, Any] = {}
    if text_free:
        hints["text_free"] = text_free
        hints["representation_options"] = ["tfidf_logreg"]
    if cat_weak:
        hints["categorical_weak"] = cat_weak
    if num_missing:
        hints["numeric_with_missing"] = num_missing
    return hints


def apply_inferred_schema(state: Any, profile: dict[str, Any] | None = None) -> list[str]:
    """Persist inferred paths/columns/hints onto ``state.config`` when blank."""
    fields: list[str] = []
    data = reconcile_data_paths(state.config.data)
    if (
        data.train_csv != state.config.data.train_csv
        or data.test_csv != state.config.data.test_csv
        or data.sample_submission_csv != state.config.data.sample_submission_csv
    ):
        state.config = state.config.model_copy(update={"data": data})
        fields.append("config.data")
    if not profile:
        return fields
    roles = infer_column_roles_from_profile(profile, state.config.problem_type)
    if state.adopt_target_if_blank(roles.get("target")):
        fields.append("config.target_column")
    if state.adopt_id_if_blank(roles.get("id")):
        fields.append("config.id_column")
    if not (state.config.feature_hints or {}):
        hints = infer_feature_hints_from_profile(
            profile,
            target=state.resolved_target(),
            id_column=state.config.id_column or "",
        )
        if hints:
            state.config = state.config.model_copy(update={"feature_hints": hints})
            fields.append("config.feature_hints")
    return fields
