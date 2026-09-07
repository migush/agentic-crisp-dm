---
name: run-trace-dashboard
description: Start the local MAADS trace dashboard FastAPI API and Vite dev UI. Use when the user wants the dashboard, trace UI, port 8765, Launch tab, or live run monitoring.
---

# Run the trace dashboard

Two processes for local hot reload. Communications contain full LLM prompts — bind **127.0.0.1** only.

## API (required)

```bash
source .venv/bin/activate
pip install -e ".[dashboard]"
python -m maads dashboard --case titanic --no-open
# defaults: --host 127.0.0.1 --port 8765 --artifact-dir artifacts
```

Health: `http://127.0.0.1:8765/api/health`  
Docs: `http://127.0.0.1:8765/api/docs`

If UI shows **No cases**, a run with `status.json` must exist under `artifacts/<case>/runs/`, or restart after `maads run` starts.

## Vite UI (dev)

```bash
cd dashboard && npm install && npm run dev
```

Port **5173**, proxies `/api` → `:8765`. Do not add react-router.

## Production-ish (one process)

```bash
cd dashboard && npm install && npm run build
python -m maads dashboard --case titanic
```

Serves `dashboard/dist` from the API process.

## Hosted vs local

| | Local `:8765` | Webapp mount `/dashboard` |
|---|---|---|
| Auth | none | JWT cookie / Bearer |
| Launch | `Launch.tsx` + `POST /api/run` | `LaunchUnavailable` + 403 |

Backend code is `src/maads/dashboard/`, not `dashboard/*.py`.
