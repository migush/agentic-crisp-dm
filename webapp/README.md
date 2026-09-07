# maads account product (webapp/)

Adds accounts, BYO OpenAI and Ollama Cloud keys, and per-task cost tracking on top of
the existing `maads` pipeline, served at `https://maads.mirogeorgiev.eu`.

This app owns the origin and **mounts the existing trace dashboard**
(`src/maads/dashboard/`) at `/dashboard`, authenticated and scoped to the
logged-in account:

| Path | Serves |
|---|---|
| `/` | account SPA (login, register, profile, tasks) |
| `/api/auth`, `/api/keys`, `/api/tasks`, `/api/models` | account API |
| `/dashboard/` | trace dashboard SPA (login required) |
| `/dashboard/api/…` | trace dashboard API (login required, per-user) |

The two apps stay separate Python packages — this one is internet-facing and
gets its own CORS allowlist, while the dashboard keeps working standalone via
`python -m maads dashboard` with no accounts at all. They are wired together
only at the mount in `backend/app.py`, which swaps two dependencies
(`case_scope_dep`, `launch_guard_dep`) on the dashboard app.

## Per-user isolation

Each account gets its own artifact root at `data/users/<user_id>/artifacts/`,
passed to the pipeline as `maads run --artifact-dir …`. Runs are therefore
never shared, and the dashboard only ever reads the caller's own root. The
repo's own `artifacts/` tree is additionally exposed to every account as
**read-only demo cases** (tagged `read_only` in `GET /dashboard/api/cases`).

## Layout

- `backend/` — FastAPI app: username-only JWT auth, ciphertext-only OpenAI
  and Ollama Cloud key storage, live model listing (`POST /api/models` with
  `provider`), task launch/spend reporting, per-user artifact roots (`paths.py`). SQLite file at
  `data/webapp.db` (gitignored). `init_db` migrates a live `users` table in
  place (`email` → `username`, preserve `id`); do not delete the database.
- `frontend/` — React + TypeScript + Vite + Tailwind SPA. All API-key
  encryption/decryption happens here via the Web Crypto API
  (`src/lib/crypto.ts`) — the backend never sees a plaintext provider key
  except transiently at launch / model list.

## Running locally

```bash
# Backend (from repo root, with the maads package installed):
pip install -e ".[webapp,dashboard]"
WEBAPP_ALLOW_DEV_SECRET=1 WEBAPP_INSECURE_COOKIES=1 \
  uvicorn webapp.backend.app:app --reload --port 8766

# Frontend (separate terminal):
cd webapp/frontend
npm install
npm run dev   # proxies /api to 127.0.0.1:8766, see vite.config.ts
```

`npm run dev` only serves the account SPA. To exercise `/dashboard` too, build
both frontends and hit port 8766 directly:

```bash
(cd dashboard && npm install && npm run build)
(cd webapp/frontend && npm install && npm run build)
```

The dashboard build uses relative asset URLs, so the same `dashboard/dist`
works at `/` (standalone) and at `/dashboard/` (mounted) with no rebuild.

Environment variables:
- `WEBAPP_JWT_SECRET` — **required**; the app refuses to start without it
  (`openssl rand -hex 32`). Set `WEBAPP_ALLOW_DEV_SECRET=1` to use the
  published dev fallback for local work only.
- `WEBAPP_ALLOWED_ORIGINS` — comma-separated CORS allowlist (defaults to
  `https://maads.mirogeorgiev.eu`).
- `WEBAPP_INSECURE_COOKIES=1` — drop `Secure` on the session cookie so it
  works over plain-HTTP localhost. Never set this in the deployment.

## Security model (read before deploying)

Sessions use a 12h HS256 JWT, returned to the account SPA as a bearer token
*and* set as an `HttpOnly` cookie. Register and login take only a username
(3–32 chars for new accounts: `[A-Za-z0-9._-]`). There is no login password —
anyone who knows a username can open that account. The cookie exists because
the trace dashboard downloads handoff zips through plain `<a download>` links
and embeds report figures as `<img src>`, neither of which can carry an
`Authorization` header. There is no revocation list — logout clears the cookie
and the local token, but an already-issued JWT stays valid until it expires.

The provider API key is encrypted client-side (AES-GCM, key derived via PBKDF2
from a user-chosen passphrase) before it ever reaches the backend — the
backend and its SQLite DB only ever hold ciphertext, unique per
`(user_id, provider)` for `openai` and `ollama_cloud`. The plaintext key is
sent only over HTTPS to `POST /api/models` (to list chat models for that key)
and `POST /api/tasks` (to launch). `run_launcher.py` injects `OPENAI_API_KEY`
or `OLLAMA_API_KEY` (+ `OLLAMA_BASE_URL=https://ollama.com` for Ollama Cloud)
and `MODEL` into the child env for the lifetime of `maads run` and never logs
or persists them. Losing the passphrase means the stored key is unrecoverable
by design — there is no reset path, only delete-and-re-add.

Hosted product accepts OpenAI and Ollama Cloud keys. Local Ollama (localhost)
stays in the standalone pipeline / local dashboard catalog.

## Deployment

`maads.mirogeorgiev.eu` is served via a manually-configured reverse proxy
outside this repo. `Caddyfile.example` is the whole config: a single
`reverse_proxy` to this backend on `:8766`, which serves both SPAs and both
APIs. Apply the equivalent change on the actual box.

Do **not** split the two apps across ports with per-path `handle` blocks. Both
apps serve a SPA at `/` and an API at `/api`, so whichever block claims `/api/*`
silently 404s the other app's entire UI — that is what took the dashboard down
after the account product was first deployed.

Deploy steps: build both frontends (see above), then run the backend with
`WEBAPP_JWT_SECRET` set. Both `dashboard/dist` and `webapp/frontend/dist` are
read from their in-repo locations, so no copying to `/srv` is needed.
Existing `data/webapp.db` is migrated on startup; do not recreate it from
scratch.
