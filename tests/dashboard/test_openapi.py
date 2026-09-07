"""OpenAPI documentation for the dashboard API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from maads.dashboard.server import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        "maads.dashboard.server._artifact_root",
        tmp_path,
        raising=False,
    )
    return TestClient(create_app())


def test_openapi_schema_under_api_prefix(client: TestClient) -> None:
    res = client.get("/api/openapi.json")
    assert res.status_code == 200
    schema = res.json()
    assert schema["info"]["title"] == "MAADS Trace Dashboard API"
    assert "/api/health" in schema["paths"]
    assert "/api/cases" in schema["paths"]


def test_swagger_ui_under_api_prefix(client: TestClient) -> None:
    res = client.get("/api/docs")
    assert res.status_code == 200
    assert "swagger" in res.text.lower()


def test_redoc_under_api_prefix(client: TestClient) -> None:
    res = client.get("/api/redoc")
    assert res.status_code == 200
    assert "redoc" in res.text.lower()


def test_root_openapi_docs_disabled(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
