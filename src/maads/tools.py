"""Tools agents may call.

`PythonExec` and `FileIO` are working. `RAGRetriever` lives in `maads.rag`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maads.rag import RAGRetriever  # re-export for tests and callers


# ────────────────────────────────────────────────────────────────────────────
# PythonExec — the sandbox every agent uses to actually run pandas / sklearn.
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False
    script_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "timed_out": self.timed_out,
            "script_path": str(self.script_path) if self.script_path else None,
        }


def _safe_exec_label(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip()).strip("_")
    return slug[:80] if slug else "run"


class PythonExec:
    """Run Python code in a subprocess, capture stdout/stderr/return code.

    This is intentionally simple. It is NOT a security sandbox — the
    execution environment is trusted. It enforces a wall-clock
    timeout to stop runaway agent code.

    When ``keep_scripts`` is True (default), every executed script and its
    captured stdout/stderr are written under ``workdir/exec/`` for later review.
    """

    def __init__(
        self,
        workdir: Path,
        timeout_seconds: int = 90,
        *,
        keep_scripts: bool = True,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.keep_scripts = keep_scripts
        self._exec_seq = 0
        if keep_scripts:
            (self.workdir / "exec").mkdir(parents=True, exist_ok=True)

    def run(
        self,
        code: str,
        extra_env: dict[str, str] | None = None,
        *,
        label: str = "",
    ) -> ExecResult:
        script_path: Path
        delete_after = False

        if self.keep_scripts:
            self._exec_seq += 1
            stem = f"{self._exec_seq:05d}"
            if label:
                stem = f"{stem}_{_safe_exec_label(label)}"
            script_path = self.workdir / "exec" / f"{stem}.py"
            script_path.write_text(code, encoding="utf-8")
        else:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", dir=self.workdir, delete=False
            ) as fh:
                fh.write(code)
                script_path = Path(fh.name)
            delete_after = True

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=self._make_env(extra_env),
            )
            result = ExecResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                script_path=script_path if self.keep_scripts else None,
            )
        except subprocess.TimeoutExpired as e:
            result = ExecResult(
                ok=False,
                stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                stderr=f"Timed out after {self.timeout_seconds} seconds.",
                return_code=-1,
                timed_out=True,
                script_path=script_path if self.keep_scripts else None,
            )

        if self.keep_scripts:
            self._persist_exec_artifacts(script_path, result, label=label)
        elif delete_after:
            try:
                script_path.unlink()
            except OSError:
                pass

        return result

    def _persist_exec_artifacts(
        self,
        script_path: Path,
        result: ExecResult,
        *,
        label: str,
    ) -> None:
        stdout_path = script_path.with_suffix(".stdout.txt")
        stderr_path = script_path.with_suffix(".stderr.txt")
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")

        manifest = self.workdir / "exec" / "manifest.jsonl"
        substep = ""
        try:
            from maads.observability import context as obs_ctx

            substep = obs_ctx.current_substep.get() or ""
        except Exception:
            pass
        record = {
            "seq": self._exec_seq,
            "label": label or None,
            "substep": substep or None,
            "script": script_path.name,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "ok": result.ok,
            "return_code": result.return_code,
            "timed_out": result.timed_out,
        }
        line = json.dumps(record, default=str) + "\n"
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(line)
        collected_manifest = (
            self.workdir.parent / "collected" / "sandbox" / "manifest.jsonl"
        )
        if collected_manifest.parent.name == "sandbox":
            collected_manifest.parent.mkdir(parents=True, exist_ok=True)
            with collected_manifest.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def _make_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        import os
        env = os.environ.copy()
        if extra:
            env.update(extra)
        return env


# ────────────────────────────────────────────────────────────────────────────
# FileIO — read/write helpers under a per-run artifact directory.
# ────────────────────────────────────────────────────────────────────────────

class FileIO:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, data: Any) -> Path:
        path = self.artifact_dir / name
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def write_text(self, name: str, content: str) -> Path:
        path = self.artifact_dir / name
        path.write_text(content)
        return path

    def copy_in(self, src: Path) -> Path:
        dst = self.artifact_dir / Path(src).name
        shutil.copy2(src, dst)
        return dst

    def path_for(self, name: str) -> Path:
        return self.artifact_dir / name


# ────────────────────────────────────────────────────────────────────────────
# inspect_dataset — pre-flight column/schema summary for DE substeps
# ────────────────────────────────────────────────────────────────────────────

def inspect_dataset(
    train_csv: str | Path,
    test_csv: str | Path | None = None,
    *,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Return row counts, dtypes, and train/test column diff for codegen context."""
    import pandas as pd

    train_path = Path(train_csv)
    if not train_path.exists():
        return {"error": f"train file not found: {train_path}"}

    tr = pd.read_csv(train_path, nrows=5000)
    out: dict[str, Any] = {
        "train_rows": int(len(tr)),
        "train_columns": list(tr.columns),
        "train_dtypes": {c: str(tr[c].dtype) for c in tr.columns},
        "train_missing": {c: int(tr[c].isna().sum()) for c in tr.columns},
    }
    if target_column and target_column in tr.columns:
        out["target_present_in_train"] = True
        out["target_missing_train"] = int(tr[target_column].isna().sum())
    elif target_column:
        out["target_present_in_train"] = False

    if test_csv:
        te_path = Path(test_csv)
        if te_path.exists():
            te = pd.read_csv(te_path, nrows=5000)
            out["test_rows"] = int(len(te))
            out["test_columns"] = list(te.columns)
            tr_set, te_set = set(tr.columns), set(te.columns)
            out["columns_only_in_train"] = sorted(tr_set - te_set)
            out["columns_only_in_test"] = sorted(te_set - tr_set)
            out["columns_shared"] = sorted(tr_set & te_set)
        else:
            out["test_error"] = f"test file not found: {te_path}"
        return out
    return out


__all__ = ["ExecResult", "FileIO", "PythonExec", "RAGRetriever"]
