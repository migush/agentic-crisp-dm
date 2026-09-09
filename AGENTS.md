# AGENTS.md — maads

`maads` is a **six-agent** CRISP-DM 1.0 pipeline for Kaggle-style problems. CrewAI Flow (`CrispDMFlow`) orchestrates; specialists mutate a Pydantic `CrispDMState`. The same agent code and prompts must run all demo cases unmodified — only YAML in `configs/` and optional `knowledge/<case>_experience.md` differ.

**Agents:** Project Manager, Domain, Data Engineer, Data Scientist, Developer, **Storyteller**.

Canonical personas live in `src/maads/config/agents.yaml`. Phase crews are thin JSON kickoff routers.

## Layout

| Path | Role |
|---|---|
| `src/maads/` | Installable package (src layout, Python 3.10–3.13) |
| `tests/` | Pytest; outside the package |
| `configs/` | Case YAML (`titanic`, `house_prices`, `disaster_tweets`, `titanic_loopdemo`) |
| `skills/` | **MAADS runtime** agent skills (loops, JSON contract). Not Cursor skills. |
| `src/maads/dashboard/` | FastAPI trace API (`create_app`, bind `127.0.0.1:8765`) |
| `dashboard/` | Vite + React 18 trace UI (tab state, **no react-router**) |
| `webapp/` | Hosted account product (`:8766`, SQLite, username JWT); mounts dashboard at `/dashboard` |
| `artifacts/<case>/runs/<run_id>/` | Per-run outputs; `current` is a **text file** naming the run id (not a symlink) |

Graphify hubs: `CrispDMState`, `RunPaths`, `TraceRun`, `load_case_config`, `StateDelta`, `PythonExec`, `CrispDMFlow`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # pytest, coverage
# extras as needed:
pip install -e ".[dashboard]"    # FastAPI + uvicorn
pip install -e ".[webapp]"       # FastAPI + uvicorn + PyJWT
cp .env.example .env             # MODEL and/or OPENAI_API_KEY
```

No ruff, black, mypy, eslint, or prettier in this repo. Python quality bar is pytest. Frontends typecheck only via `npm run build` (`tsc && vite build`).

## Commands (verified in `__main__.py` / package.json)

```bash
python -m maads run --case titanic
python -m maads run --config configs/titanic.yaml
python -m maads data download --case titanic
python -m maads flow plot
python -m maads dashboard --case titanic --no-open   # :8765
python -m maads artifacts render --run <run_dir>
python -m maads artifacts backfill-timing

MAADS_TRACE=0 coverage run -m pytest tests/test_path_coverage.py -q   # ~1 min, do this first
pytest
pytest tests/dashboard/ -q     # needs .[dashboard]
pytest tests/webapp/ -q        # needs .[webapp]

cd dashboard && npm install && npm run dev           # :5173, proxies /api → :8765
cd webapp/frontend && npm install && npm run dev     # :5174, proxies /api → :8766
```

Webapp backend (dev):

```bash
pip install -e ".[webapp,dashboard]"
WEBAPP_ALLOW_DEV_SECRET=1 WEBAPP_INSECURE_COOKIES=1 \
  uvicorn webapp.backend.app:app --reload --port 8766
```

## Control flow

```
run → P1 BU → P2 DU → checkpoint 3.1 (Loop A → P1 | advance → P3)
 → P4 Modeling → checkpoint 5.1 (Loop B → P3 | advance → P5)
 → checkpoint 5.2 (Loop C → P1 | advance → P5 tail)
 → P6 Deployment → complete
```

Loop caps live in `skills/crisp-dm-loops/SKILL.md` and `flow/phase_runner.py`. Read both before touching routers.

Every specialist substep: `capabilities.execution_evidence` → crew `kickoff_substep` (LLM JSON) → `capabilities.apply_response`. PM `plan()` at checkpoint substeps returns `Plan(action in {advance, loop_back, halt})`.

## Coding rules

- Case variance only in YAML `feature_hints`, `success_criterion`, data paths, and knowledge markdown. Never `if case_id == "titanic"` in agents/crews/capabilities.
- Prompt context: `state.view_for(agent_name)`. Never dump full `CrispDMState`.
- Agent LLM output is JSON only (`src/maads/output_contracts.py` + `skills/json-output-contract/SKILL.md`).
- Tests patch `maads.crew.run_text_task` / `run_json_task` (autouse stub in `tests/conftest.py`). Set `MAADS_SKIP_MODEL_PROBE=1` is already done there.
- `PythonExec` is a subprocess timeout, **not** a security sandbox.
- Standalone dashboard has **no auth** and `CORS *` — local only. Communications contain full prompts.
- Hosted runs start from webapp **Tasks**, not dashboard Launch (`launch_guard_dep` → 403; UI shows `LaunchUnavailable`).
- Do not reverse-proxy webapp and dashboard as separate origins/paths; both claim `/` and `/api`. Mount model in `webapp/backend/app.py` is required.

## Env (see `.env.example`)

`MODEL`, `MODEL_CODE`, `OPENAI_API_KEY`, `MAX_TOKENS_PER_RUN`, `MAADS_RUN_DEADLINE_SEC`, `MAADS_TRACE`, `MAADS_TRACE_LLM_IO`, `MAADS_PROGRESS`, `MAADS_SKIP_MODEL_PROBE`. Webapp: `WEBAPP_JWT_SECRET` (required in prod), `WEBAPP_ALLOW_DEV_SECRET`, `WEBAPP_INSECURE_COOKIES`, `WEBAPP_ALLOWED_ORIGINS`, `WEBAPP_RUN_TIMEOUT_SEC`.

## Success criteria

(1) Same agents/prompts for all cases via config only. (2) Valid Kaggle `submission.csv` beating the trivial baseline. (3) Paper across cases. One finished case delivers (2) plus `final_state.json` + `trace/`.

Read `workflow_state.md` at session start; update it before ending. Do not commit unless asked.
