"""FastAPI app for the maads.mirogeorgiev.eu account product.

Separate from ``src/maads/dashboard`` (the local-only trace viewer) on
purpose: this app is internet-facing, so it gets its own CORS allowlist
instead of the dashboard's wide-open ``allow_origins=["*"]``. See
`webapp/backend/README.md`-equivalent context in the plan doc for why.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maads.model_catalog import model_catalog

from .db import init_db
from .routes_auth import router as auth_router
from .routes_keys import router as keys_router
from .routes_tasks import router as tasks_router


def create_app() -> FastAPI:
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
    app.include_router(tasks_router)

    @app.get("/api/models")
    def list_models() -> dict:
        return model_catalog()

    return app


app = create_app()
