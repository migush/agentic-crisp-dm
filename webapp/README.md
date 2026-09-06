# maads account product (webapp/)

Adds accounts, BYO LLM provider keys, and per-task cost tracking on top of
the existing `maads` pipeline — reachable (once deployed) alongside the
existing trace dashboard at `https://maads.mirogeorgiev.eu`. See the design
rationale in the approved plan (`/Users/miroslavgeorgiev/.claude/plans/plan-for-the-following-fluffy-sutton.md`)
for why this is a separate app from `src/maads/dashboard/`.

## Layout

- `backend/` — FastAPI app: auth (argon2 + JWT), ciphertext-only key storage,
  task launch/spend reporting. SQLite file at `data/webapp.db` (gitignored).
- `frontend/` — React + TypeScript + Vite + Tailwind SPA. All API-key
  encryption/decryption happens here via the Web Crypto API
  (`src/lib/crypto.ts`) — the backend never sees a plaintext provider key.

## Running locally

```bash
# Backend (from repo root, with the maads package installed):
pip install -e ".[webapp]"
uvicorn webapp.backend.app:app --reload --port 8766

# Frontend (separate terminal):
cd webapp/frontend
npm install
npm run dev   # proxies /api to 127.0.0.1:8766, see vite.config.ts
```

Environment variables:
- `WEBAPP_JWT_SECRET` — required in any non-local deployment (`openssl rand -hex 32`).
- `WEBAPP_ALLOWED_ORIGINS` — comma-separated CORS allowlist (defaults to
  `https://maads.mirogeorgiev.eu`).

## Security model (read before deploying)

The provider API key is encrypted client-side (AES-GCM, key derived via
PBKDF2 from a user-chosen passphrase) before it ever reaches the backend —
the backend and its SQLite DB only ever hold ciphertext. The one point where
plaintext exists server-side is transient: launching a task
(`POST /api/tasks`) sends the just-decrypted key once over HTTPS, and
`run_launcher.py` holds it only in memory for the lifetime of the `maads run`
subprocess, injected as an env var and never logged or persisted. Losing the
passphrase means the stored key is unrecoverable by design — there is no
reset path, only delete-and-re-add.

## Deployment

`maads.mirogeorgiev.eu` is already served via a manually-configured reverse
proxy outside this repo. `Caddyfile.example` shows how to add routes for this
app's backend (`/api/*`) and built frontend alongside whatever already proxies
to the dashboard — apply the equivalent change on the actual box.
