"""Dashboard HTTP server (FastAPI + optional static frontend)."""
from __future__ import annotations

import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from maads.dashboard.api import router

_artifact_root: Path = Path("artifacts")


def get_artifact_root() -> Path:
    return _artifact_root


def create_app(static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="MAADS Trace Dashboard API",
        version="0.1.0",
        description=(
            "HTTP API for the MAADS trace monitoring dashboard: case discovery, "
            "run artifacts, live polling, reports, and pipeline launch."
        ),
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    if static_dir and static_dir.is_dir():
        assets = static_dir / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                from fastapi import HTTPException
                raise HTTPException(status_code=404)
            index = static_dir / "index.html"
            if index.is_file():
                return FileResponse(index)
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Frontend not built")

    return app


def run_dashboard(
    *,
    artifact_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    static_dir: Path | None = None,
    open_browser: bool = True,
    case_id: str | None = None,
) -> None:
    """Start uvicorn and optionally open the browser."""
    global _artifact_root
    _artifact_root = artifact_dir.resolve()

    from maads.dashboard import store

    cases = store.list_cases(_artifact_root)
    print(f"Cases found: {len(cases)} ({', '.join(c['case_id'] for c in cases) or 'none'})")
    if not cases:
        print(
            "WARNING: no runs with status.json under artifact root. "
            "Start a pipeline run first, or pass --artifact-dir to the directory "
            "that contains per-case folders (e.g. artifacts/).",
        )

    app = create_app(static_dir=static_dir)

    url = f"http://{host}:{port}/"
    if case_id:
        url = f"{url}?case={case_id}"

    if open_browser:
        webbrowser.open(url)

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
