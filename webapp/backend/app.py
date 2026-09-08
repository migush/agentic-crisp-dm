"""FastAPI app for the maads.mirogeorgiev.eu account product.

This app owns the whole origin. It serves its own SPA and API at ``/`` and
``/api``, and mounts the trace dashboard (``src/maads/dashboard``) at
``/dashboard`` — authenticated and scoped to the calling account.

They stay separate apps: this one is internet-facing with its own CORS
allowlist, while the dashboard keeps working standalone via
``python -m maads dashboard`` with no accounts and its wide-open
``allow_origins=["*"]``. But they cannot be separate *origins* — both serve a
SPA at ``/`` and an API at ``/api``, so a reverse proxy splitting them by path
silently 404s one app's entire UI. Hence the mount, in ``build_dashboard_app``.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from maads.paths import repo_root

from . import paths
from .auth import assert_secret_configured
from .db import init_db
from .routes_auth import require_user_flexible
from .routes_auth import router as auth_router
from .routes_cases import router as cases_router
from .routes_keys import router as keys_router
from .routes_models import router as models_router
from .routes_tasks import router as tasks_router

DASHBOARD_MOUNT = "/dashboard"


def create_app() -> FastAPI:
    assert_secret_configured()
    init_db()
    app = FastAPI(title="maads account service")

    allowed_origins = [
        origin.strip()
        for origin in os.environ.get("WEBAPP_ALLOWED_ORIGINS", "https://maads.mirogeorgiev.eu").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(auth_router)
    app.include_router(keys_router)
    app.include_router(models_router)
    app.include_router(tasks_router)
    app.include_router(cases_router)

    # Mounted before the SPA fallback below, so /dashboard/* reaches the trace
    # dashboard instead of being swallowed by this app's catch-all.
    app.mount(DASHBOARD_MOUNT, build_dashboard_app())
    _serve_account_spa(app)
    return app


def _serve_account_spa(app: FastAPI) -> None:
    """Serve webapp/frontend/dist from this app.

    The reverse proxy now forwards the whole origin here rather than serving
    the built SPA itself, so that both apps agree on one /api namespace.
    Skipped when the frontend hasn't been built (backend-only dev, tests).
    """
    static_dir = repo_root() / "webapp" / "frontend" / "dist"
    if not static_dir.is_dir():
        return

    assets = static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="account-assets")

    index = static_dir / "index.html"

    @app.get(DASHBOARD_MOUNT, include_in_schema=False)
    def dashboard_slash() -> RedirectResponse:
        # Starlette's own mount redirect never fires here: the catch-all below
        # matches "/dashboard" first and would hand back this app's SPA. The
        # trailing slash matters — the dashboard bundle uses relative asset URLs.
        return RedirectResponse(f"{DASHBOARD_MOUNT}/", status_code=307)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        # Client-side routes (/login, /profile, …) all resolve to index.html;
        # unmatched API paths must still 404 rather than return HTML.
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Frontend not built")
        return FileResponse(index)


def build_dashboard_app() -> FastAPI:
    """The existing trace dashboard, authenticated and scoped to one account.

    The dashboard app is reused verbatim; only its two seams are swapped, so
    ``python -m maads dashboard`` keeps behaving exactly as before:

    * ``case_scope_dep`` — instead of the single process-wide artifact root,
      each request gets that user's own root plus the repo's read-only demo
      runs. Because the replacement depends on ``require_user_flexible``, an
      unauthenticated request is rejected before any handler runs, which is
      what authenticates all ~30 dashboard routes at once.
    * ``launch_guard_dep`` — POST /run is refused here; a hosted run needs the
      user's provider key, which only exists decrypted in their browser, so
      launches go through POST /api/tasks instead.
    """
    from maads.dashboard.deps import CaseScope, case_scope_dep, launch_guard_dep
    from maads.dashboard.server import create_app as create_dashboard_app

    static_dir = repo_root() / "dashboard" / "dist"
    dash_app = create_dashboard_app(static_dir=static_dir if static_dir.is_dir() else None)

    def scoped(user_id: int = Depends(require_user_flexible)) -> CaseScope:
        return CaseScope(
            write_root=paths.user_artifact_root(user_id),
            demo_roots=(paths.demo_artifact_root(),),
        )

    def no_launch(_: int = Depends(require_user_flexible)) -> None:
        raise HTTPException(
            status_code=403,
            detail="Start runs from the Tasks page — a run needs your provider API key.",
        )

    dash_app.dependency_overrides[case_scope_dep] = scoped
    dash_app.dependency_overrides[launch_guard_dep] = no_launch
    return dash_app


app = create_app()
