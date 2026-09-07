from pathlib import Path

import pytest
from fastapi import HTTPException

from maads.dashboard.deps import CaseScope


def test_case_scope_rejects_case_symlink_outside_root(tmp_path: Path) -> None:
    own = tmp_path / "own"
    outside = tmp_path / "outside"
    own.mkdir()
    outside.mkdir()
    (own / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HTTPException, match="Case not found"):
        CaseScope(own).case_path("escape")


def test_case_scope_rejects_run_symlink_outside_case(tmp_path: Path) -> None:
    own = tmp_path / "own"
    case = own / "sample"
    outside = tmp_path / "outside"
    (case / "runs").mkdir(parents=True)
    outside.mkdir()
    (case / "runs" / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HTTPException, match="Run not found"):
        CaseScope(own).case_dir("sample", "escape")


def test_case_scope_marks_demo_cases_read_only(tmp_path: Path) -> None:
    own = tmp_path / "own"
    demo = tmp_path / "demo"
    (own / "private").mkdir(parents=True)
    (demo / "sample").mkdir(parents=True)
    scope = CaseScope(own, (demo,))

    assert scope.is_read_only("private") is False
    assert scope.is_read_only("sample") is True
