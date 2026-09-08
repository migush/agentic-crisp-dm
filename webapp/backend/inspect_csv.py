"""Observational CSV profiling for the Cases wizard.

This module never mutates files, never encodes labels, never splits tables,
and never writes a sample submission. Output is facts for the user (and a
draft ``case.yaml``), not a prep plan.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

_TRY_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
_DOCUMENT_SUFFIXES = {".md", ".markdown", ".pdf", ".txt"}
_UNSUPPORTED_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp3",
    ".wav",
    ".mp4",
    ".parquet",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
}

SAMPLE_VALUE_CAP = 5


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Return ``(text, encoding_name)``. latin-1 is the last-resort fallback."""
    for enc in _TRY_ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample.splitlines()[0] if sample.splitlines() else "":
            return "\t"
        if ";" in sample.splitlines()[0] if sample.splitlines() else "":
            return ";"
        return ","


def _file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _UNSUPPORTED_SUFFIXES:
        return "unsupported"
    if suffix in _CSV_SUFFIXES:
        return "csv"
    if suffix in _DOCUMENT_SUFFIXES:
        return "document"
    if suffix == "":
        return "csv"
    return "document"


def profile_csv_text(text: str, *, encoding: str, filename: str) -> dict[str, Any]:
    sample = "\n".join(text.splitlines()[:20])
    delimiter = sniff_delimiter(sample) if sample.strip() else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        return {
            "filename": filename,
            "original_filename": filename,
            "encoding": encoding,
            "delimiter": delimiter,
            "n_rows": 0,
            "n_cols": 0,
            "columns": [],
            "parse_warnings": ["empty file"],
            "kind": "csv",
        }

    header = [h.strip() for h in header]
    warnings: list[str] = []
    if not header or any(not name for name in header):
        warnings.append("one or more header cells are empty")
    if len(header) != len(set(header)):
        warnings.append("duplicate column names")

    columns: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "n_missing": 0,
            "n_unique": 0,
            "sample_values": [],
            "_seen": set(),
        }
        for name in header
    }
    n_rows = 0
    for row in reader:
        n_rows += 1
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        for name, raw in zip(header, row):
            cell = "" if raw is None else str(raw).strip()
            col = columns[name]
            if cell == "":
                col["n_missing"] += 1
                continue
            seen: set[str] = col["_seen"]
            if cell not in seen:
                if len(col["sample_values"]) < SAMPLE_VALUE_CAP:
                    col["sample_values"].append(cell)
                seen.add(cell)

    out_cols = []
    for name in header:
        col = columns[name]
        n_unique = len(col["_seen"])
        dtype = _guess_dtype(col["sample_values"], n_rows=n_rows, n_missing=col["n_missing"])
        out_cols.append(
            {
                "name": name,
                "dtype": dtype,
                "n_missing": col["n_missing"],
                "n_unique": n_unique,
                "sample_values": col["sample_values"],
            }
        )

    return {
        "filename": filename,
        "original_filename": filename,
        "encoding": encoding,
        "delimiter": delimiter,
        "n_rows": n_rows,
        "n_cols": len(header),
        "columns": out_cols,
        "parse_warnings": warnings,
        "kind": "csv",
    }


def _guess_dtype(samples: list[str], *, n_rows: int, n_missing: int) -> str:
    if not samples:
        return "empty"
    numeric = 0
    integer = 0
    for val in samples:
        try:
            float(val.replace(",", ""))
            numeric += 1
            if "." not in val and "e" not in val.lower():
                integer += 1
        except ValueError:
            pass
    if numeric == len(samples) and integer == len(samples):
        return "integer"
    if numeric == len(samples):
        return "float"
    if any(len(v) > 80 for v in samples):
        return "text"
    return "string"


def profile_path(path: Path, *, original_filename: str | None = None) -> dict[str, Any]:
    """Profile one stored file. Never executes the contents."""
    display = original_filename or path.name
    kind = _file_kind(path)
    if kind == "unsupported":
        return {
            "filename": path.name,
            "original_filename": display,
            "encoding": "",
            "delimiter": "",
            "n_rows": 0,
            "n_cols": 0,
            "columns": [],
            "parse_warnings": [f"file type {path.suffix or '(none)'} is not a tabular CSV"],
            "kind": "unsupported",
        }
    if kind == "document":
        return {
            "filename": path.name,
            "original_filename": display,
            "encoding": "",
            "delimiter": "",
            "n_rows": 0,
            "n_cols": 0,
            "columns": [],
            "parse_warnings": [],
            "kind": "document",
        }

    data = path.read_bytes()
    text, encoding = decode_bytes(data)
    report = profile_csv_text(text, encoding=encoding, filename=path.name)
    report["original_filename"] = display
    return report


def inspect_directory(raw_dir: Path) -> dict[str, Any]:
    """Profile every file under ``raw_dir`` (non-recursive)."""
    files: list[dict[str, Any]] = []
    if raw_dir.is_dir():
        for path in sorted(raw_dir.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                files.append(profile_path(path))
    csv_files = [f for f in files if f.get("kind") == "csv"]
    unsupported = [f for f in files if f.get("kind") == "unsupported"]
    runnable = bool(csv_files) and not unsupported
    reason = None
    if unsupported and not csv_files:
        reason = "Uploaded files are not tabular CSVs (images, audio, or other binaries are not runnable in V1)."
    elif unsupported:
        reason = "At least one uploaded file is not a tabular CSV."
    elif not files:
        reason = "No files uploaded."
    elif not csv_files:
        reason = "Need at least one CSV. Documents alone are not runnable."
    return {
        "files": files,
        "runnable": runnable,
        "unsupported_reason": reason,
    }


def problem_type_runnable(problem_type: str) -> tuple[bool, str | None]:
    """V1 ships supervised classification and regression only."""
    normalized = (problem_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    allowed = {
        "",
        "classification",
        "binary_classification",
        "multiclass_classification",
        "regression",
        "prediction",
    }
    blocked = {
        "clustering": "Clustering / segmentation is not runnable yet.",
        "segmentation": "Clustering / segmentation is not runnable yet.",
        "association": "Association / dependency analysis is not runnable yet.",
        "dependency": "Association / dependency analysis is not runnable yet.",
        "description": "Description-only studies are not runnable yet.",
        "summarization": "Description-only studies are not runnable yet.",
        "image": "Image inputs are not runnable yet.",
        "audio": "Audio inputs are not runnable yet.",
        "forecasting": "True forecasting is not runnable yet.",
        "time_series": "True forecasting is not runnable yet.",
    }
    if normalized in blocked:
        return False, blocked[normalized]
    if normalized in allowed:
        return True, None
    return False, f"Problem type {problem_type!r} is not runnable in V1 (supervised CSV classification or regression)."
