"""Artifact-scope dependencies shared by local and hosted dashboards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from maads.dashboard import store


@dataclass(frozen=True)
class CaseScope:
    """Filesystem roots one dashboard request may read or write."""

    write_root: Path
    demo_roots: tuple[Path, ...] = ()

    def list_cases(self) -> list[dict]:
        cases: list[dict] = []
        seen: set[str] = set()
        for root, read_only in ((self.write_root, False), *((root, True) for root in self.demo_roots)):
            resolved_root = root.resolve()
            for case in store.list_cases(resolved_root):
                case_id = str(case.get("case_id", ""))
                candidate = (resolved_root / case_id).resolve()
                if not _safe_segment(case_id) or candidate.parent != resolved_root or case_id in seen:
                    continue
                seen.add(case_id)
                cases.append({**case, "read_only": read_only})
        return cases

    def case_path(self, case_id: str) -> Path:
        if not _safe_segment(case_id):
            raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
        for root in (self.write_root, *self.demo_roots):
            resolved_root = root.resolve()
            candidate = (resolved_root / case_id).resolve()
            if candidate.parent == resolved_root and candidate.is_dir():
                return candidate
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    def case_dir(self, case_id: str, run_id: str | None = None) -> Path:
        if run_id is not None and not _safe_segment(run_id):
            raise HTTPException(status_code=404, detail=f"Run not found: {case_id}/{run_id}")
        case_path = self.case_path(case_id)
        try:
            run_dir = store.case_dir(case_path.parent, case_id, run_id).resolve()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run_dir != case_path and case_path not in run_dir.parents:
            raise HTTPException(status_code=404, detail=f"Run not found: {case_id}/{run_id or 'current'}")
        return run_dir

    def is_read_only(self, case_id: str) -> bool:
        """Return whether the selected case comes from a shared demo root."""
        return self.case_path(case_id).parent != self.write_root.resolve()


def _safe_segment(value: str) -> bool:
    return bool(value) and Path(value).name == value and value not in {".", ".."}


def case_scope_dep() -> CaseScope:
    """Use the process-wide artifact root for the standalone dashboard."""
    from maads.dashboard.server import get_artifact_root

    return CaseScope(write_root=get_artifact_root())


def launch_guard_dep() -> None:
    """Allow launches in standalone mode; the hosted app overrides this."""
