"""Filesystem boundaries for hosted account data and shared demo runs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


def known_case_ids() -> frozenset[str]:
    """Return case IDs backed by a top-level YAML configuration file."""
    return frozenset(path.stem for path in CONFIG_ROOT.glob("*.yaml") if path.is_file())


def user_artifact_root(user_id: int) -> Path:
    """Return the isolated writable artifact root for one account."""
    return REPO_ROOT / "data" / "users" / str(user_id) / "artifacts"


def demo_artifact_root() -> Path:
    """Return the repository's shared, read-only demo artifact root."""
    return REPO_ROOT / "artifacts"
