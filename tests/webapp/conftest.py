"""Shared setup for the account-backend tests."""

from __future__ import annotations

import pytest

import webapp.backend.paths as paths_module


@pytest.fixture(autouse=True)
def webapp_test_env(monkeypatch, tmp_path):
    # create_app() refuses to start on the published fallback JWT secret; tests
    # opt in explicitly rather than each setting a real one.
    monkeypatch.setenv("WEBAPP_ALLOW_DEV_SECRET", "1")
    # TestClient speaks plain HTTP, so a Secure cookie would never be stored.
    monkeypatch.setenv("WEBAPP_INSECURE_COOKIES", "1")
    # Keep per-user artifact roots out of the real repo's data/ directory.
    monkeypatch.setattr(
        paths_module,
        "user_artifact_root",
        lambda user_id: _mkdir(tmp_path / "users" / str(user_id) / "artifacts"),
    )
    monkeypatch.setattr(
        paths_module, "demo_artifact_root", lambda: _mkdir(tmp_path / "demo"),
    )


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
