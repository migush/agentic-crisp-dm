# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`maads` is a six-agent system (Project Manager, Domain, Data Engineer, Data Scientist, Developer, Storyteller)
that walks Kaggle-style problems through the CRISP-DM 1.0 process model. CrewAI powers LLM calls; a typed
shared state (`CrispDMState`) and trace tooling make runs observable. The same agent code and prompts must
work unmodified across all three demo cases (titanic, house_prices, disaster_tweets) — only the per-case
YAML config differs. See `docs/ARCHITECTURE.md` for the full flow graph and module table.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set OPENAI_API_KEY and/or MODEL — see .env.example for Ollama vs cloud
```
Python 3.10–3.13 (`pyproject.toml` `requires-python`). Package uses **src layout** — importable code is
`src/maads`, tests live outside the package at `tests/`.

## Common commands

```bash
# Run the pipeline for a case
python -m maads run --case titanic
python -m maads run --config <path>

# Download case data
python -m maads data download --case titanic
python -m maads data download --competition <slug>

# Plot the CrewAI Flow graph
python -m maads flow plot

# Trace dashboard (API server; serves built dashboard/dist if present)
pip install -e ".[dashboard]"
python -m maads dashboard --case titanic --no-open
# Dev UI with hot reload (proxies /api to :8765):
cd dashboard && npm install && npm run dev

# Tests
pytest                                              # full suite
MAADS_TRACE=0 coverage run -m pytest tests/test_path_coverage.py -q   # fast, mocked-LLM path coverage (~1 min)
pytest tests/test_flow_happy_path.py -q             # single file
pytest tests/test_flow_happy_path.py::test_name -q  # single test
coverage report --show-missing
```

Dashboard binds `127.0.0.1:8765`. It reads `artifacts/<case>/runs/<run_id>/` via the `current` text run-id pointer;
API docs at `/api/docs`, schema at `/api/openapi.json`. Communications transcripts contain full LLM
prompts — local use only, never expose this dashboard externally.

Key env vars (see `.env.example`): `MODEL`, `MAX_TOKENS_PER_RUN`, `MAADS_TRACE`, `MAADS_TRACE_LLM_IO`,
`MAADS_PROGRESS` (`--quiet` disables the live progress bar), `MAADS_SKIP_MODEL_PROBE` (tests set this).

## Architecture

### Control flow (CrewAI Flow, `flow/crisp_dm_flow.py`)

```
run → phase_1 (Business Understanding) → phase_2 (Data Understanding)
    → checkpoint_3_1 (PM router) --loop A--> phase_1
                                 --advance--> phase_3 (Data Preparation)
    → phase_4 (Modeling)
    → checkpoint_5_1 (PM router) --loop B--> phase_3
                                 --advance--> phase_5 (Evaluation)
    → checkpoint_5_2 (PM router) --loop C--> phase_1
                                 --advance--> phase_5_tail
    → phase_6 (Deployment) → complete
```

Loop contours (when each back-edge fires, retry caps) are defined declaratively in
`skills/crisp-dm-loops/SKILL.md` — read it before touching router/loop logic.

### Module map

| Path | Role |
|---|---|
| `flow/crisp_dm_flow.py` | `@start`/`@listen`/`@router` flow graph definition |
| `flow/phase_runner.py` | Shared substep dispatch, advance, loop-back, retry caps |
| `flow/routers.py` | PM checkpoint routing helpers |
| `flow/tracing.py` | Trace + status flush hooks fired on flow transitions |
| `crews/<name>_crew/` | Thin per-role kickoff routers (JSON via `kickoff_json`); canonical YAML in `src/maads/config/` |
| `crews/kickoff.py` | `kickoff_json` — one-agent Crew kickoff used by every LLM substep |
| `capabilities/` | Deterministic Python execution per role (data_engineer, data_scientist, developer, domain) — sandboxed code execution + JSON response application, not LLM calls |
| `state.py` | `CrispDMState` — single source of truth for run state (583 lines; read before changing state shape) |
| `agents.py` | The six agent wrappers (PM, Domain, Data Engineer, Data Scientist, Developer, Storyteller) |
| `crew.py` | The CrewAI LLM call seam — tests monkeypatch `maads.crew.run_text_task` to fake LLM output |
| `observability/` | OpenTelemetry-based trace export: timeline, narrative, diagrams, LLM communications log |
| `reports/` | Post-run artifacts: case report, execution analysis, final report, handoff/postmortem docs |
| `dashboard/` | FastAPI backend (`api.py`, `server.py`, `store.py`, `aggregators.py`) for the trace UI |
| `knowledge_setup.py` + `knowledge/` (repo root) | Per-case "experience" markdown fed back into agent prompts across runs |

### Substep execution pattern

Every substep in every phase follows the same three-step dispatch:
1. `capabilities.execution_evidence` (when the substep needs deterministic evidence gathering)
2. Phase crew `kickoff_substep` → single-agent CrewAI kickoff (LLM call)
3. `capabilities.apply_response` (validates/applies the agent's JSON response to state)

Agents must respond with JSON only, per `skills/json-output-contract/SKILL.md`. Authored code that agents
generate must follow the Kaggle submission contract (`skills/kaggle-submission-contract/SKILL.md`) and
leakage/CV discipline (`skills/leakage-cv-discipline/SKILL.md`).

### Artifacts layout (per case, per run)

`artifacts/<case>/runs/<run_id>/` holds `status.json`, `process.json`, `state.json`, `trace/`
(timelines, diagrams, `communications.md` — full agent↔LLM transcript). `artifacts/<case>/current` is a
text file naming the latest run id; `runs_index.json` lists all runs for a case.

### Testing conventions

- `tests/conftest.py` autouse-monkeypatches `maads.crew.run_text_task` with `tests/fixtures/titanic_exec.py`'s
  `fake_run_text_task`, so most tests never call a real LLM.
- `src/maads/testing/fake_llm.py` and `flow_harness.py` provide fakes/harness for flow-level tests.
- `tests/test_path_coverage.py` is the fast, mocked, full-pipeline path-coverage check — run this first when
  validating a change before the full suite.
- Coverage config (`pyproject.toml`) sources `maads`, omits `*/testing/*`.

### Success criteria for this project (from README)

Three deliverables, not just a working pipeline: (1) complete, general agent code/prompts that run all three
cases unmodified via config only; (2) a valid Kaggle `submission.csv` beating the trivial baseline per case,
with the public leaderboard score recorded; (3) a short paper reporting results across all cases (scores,
token cost, what didn't work). A single completed case delivers (2) for that case plus the audit trail
(`final_state.json`, `trace/`) proving how the system produced it.
