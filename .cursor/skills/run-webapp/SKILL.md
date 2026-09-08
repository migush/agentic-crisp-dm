---
name: run-webapp
description: Run the hosted-style maads account webapp (FastAPI :8766 + React Router :5174) and the mounted dashboard. Use when working on auth, tasks, API keys, JWT, SQLite, or /dashboard mount.
---

# Run the account webapp

## Backend

```bash
source .venv/bin/activate
pip install -e ".[webapp,dashboard]"
WEBAPP_ALLOW_DEV_SECRET=1 WEBAPP_INSECURE_COOKIES=1 \
  uvicorn webapp.backend.app:app --reload --port 8766
```

`WEBAPP_JWT_SECRET` is required in production. The published fallback is blocked unless `WEBAPP_ALLOW_DEV_SECRET=1`.

SQLite file: `data/webapp.db` (gitignored via `data/`). On startup `init_db` migrates a live `users` table in place (`email` → `username`, keep `id`); do not delete the file.

## Frontend

```bash
cd webapp/frontend && npm install && npm run dev
```

Port **5174**, proxy `/api` → `127.0.0.1:8766`. Routes: `/login`, `/register`, `/profile`, `/cases`, `/tasks`. Dashboard is a separate SPA at `/dashboard/` (needs `dashboard/dist` on the backend process).

## Full origin (how deploy works)

```bash
cd dashboard && npm run build
cd webapp/frontend && npm run build
uvicorn webapp.backend.app:app --port 8766
```

Caddy must reverse-proxy the **whole host** to `:8766`. Do not path-split to `:8765`.

## Tests

```bash
pytest tests/webapp/ -q
```

## Invariants

- Auth is username-only (no login password). `/register` and `/login` take `{ username }`.
- Launch runs via `POST /api/tasks`, not dashboard `POST /api/run` (403 + `LaunchUnavailable`)
- Hosted keys and launches are OpenAI or Ollama Cloud (`provider=openai` / `ollama_cloud`). Tasks model dropdown comes from `POST /api/models` with `{ provider, decrypted_api_key }` (live list after passphrase unlock), not the static catalog.
- User artifacts: `data/users/<id>/artifacts/`; demo reads repo `artifacts/`
- Store keys as ciphertext only; plaintext key only in launcher child env (`OPENAI_API_KEY` or `OLLAMA_API_KEY` + `MODEL`; Ollama Cloud also sets `OLLAMA_BASE_URL=https://ollama.com`)
