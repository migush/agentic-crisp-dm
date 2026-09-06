# maads — Multi-Agent Automated Data Science

A five-agent system that walks Kaggle-style problems through the **CRISP-DM 1.0**
process model. The **Project Manager** orchestrates each turn; specialist agents
(domain, data engineer, data scientist, developer) own their substeps. CrewAI
powers LLM calls; a typed shared state (`CrispDMState`) and trace tooling make
runs observable.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the Flow graph and module layout.

## Project layout

```
configs/                  Case YAML files (titanic, house_prices, disaster_tweets)
data/                     Downloaded competition CSVs
artifacts/<case>/         Per-run outputs (submission, trace, final_state.json)
src/maads/                Installable Python package (src layout)
  config/                 Canonical agents.yaml + tasks.yaml (CrewAI)
  flow/                   CrewAI Flow orchestration (CrispDMFlow)
  crews/                  Phase-scoped @CrewBase crews (per-phase tasks.yaml)
  agents.py               Five agent wrappers
  crew.py                 CrewAI LLM seam
  observability/          Trace export (timeline, narrative, diagrams)
tests/                    Pytest suite (outside the installable package)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"

cp .env.example .env
# Set OPENAI_API_KEY and/or MODEL (see .env.example for Ollama vs cloud)
```

Python **3.10–3.13** required (`requires-python` in `pyproject.toml`).

## Download data

```bash
python -m maads data download --case titanic
```

## Run the pipeline

```bash
python -m maads run --case titanic
# or, after install:
maads run --case titanic
```

Runs via **CrewAI Flow** (`CrispDMFlow`).

```bash
python -m maads flow plot
```

While running, watch `artifacts/titanic/status.json` or the stderr progress bar.
After completion, inspect `artifacts/titanic/trace/` for timelines, diagrams, and
`communications.md` (full agent–LLM prompt/response transcript).

Use `--quiet` or `MAADS_PROGRESS=0` to disable the live progress bar.

## Trace dashboard

Monitor live and completed runs in a local web UI (progress, token spend,
agent–LLM communications, architecture diagram).

```bash
pip install -e ".[dashboard]"

# Terminal 1 — run the pipeline
python -m maads run --case titanic

# Terminal 2 — API server (serves built frontend if dashboard/dist exists)
python -m maads dashboard --case titanic --no-open

# Dev UI with hot reload (proxies /api to :8765)
cd dashboard && npm install && npm run dev
```

Production-ish (single process serves API + static UI):

```bash
cd dashboard && npm install && npm run build
python -m maads dashboard --case titanic
```

The dashboard binds to `127.0.0.1:8765` by default. It reads
`artifacts/<case>/runs/<run_id>/` via the `current` symlink. If the UI shows
**No cases**, restart the dashboard after `maads run` starts, or check
`http://127.0.0.1:8765/api/health` for the artifact root and detected cases.
Interactive API docs: `http://127.0.0.1:8765/api/docs` (OpenAPI schema at `/api/openapi.json`).
Communications contain full prompts — local use only.

## Tests

```bash
# Fast path-coverage run (mocked LLM, ~1 min)
MAADS_TRACE=0 coverage run -m pytest tests/test_path_coverage.py -q
coverage report --show-missing

# Full test suite
pytest
```

## CLI reference

| Command | Description |
|---------|-------------|
| `maads run --case <name>` | Run from `configs/<name>.yaml` |
| `maads run --config <path>` | Run from an explicit config file |
| `maads data download --case <name>` | Download bundled case data |
| `maads data download --competition <slug>` | Download any Kaggle competition |
| `maads dashboard [--case <name>]` | Launch trace monitoring web UI |

## Environment variables

See [`.env.example`](.env.example) for `MODEL`, `MAX_TOKENS_PER_RUN`, `MAADS_TRACE`,
`MAADS_TRACE_LLM_IO`, `MAADS_PROGRESS`, and Ollama settings.

Success for this project is three deliverables, not just a pipeline that runs:

Complete code — a complete, working multi-agent system that is general: the same agent code and prompts run all three demonstration Kaggle cases (Titanic, House Prices, Disaster Tweets), with only the per-case YAML config differing. Agents that emit code actually execute it; the CRISP-DM loops fire when warranted.
Forecast — the actual predictions for each case: a Kaggle-valid submission.csv that beats the trivial baseline, with the observed public-leaderboard score recorded.
Paper — a short (4–8 page) conference-style paper that situates the work against prior automated-data-science systems and honestly reports results across all cases combined (scores, per-agent/per-provider token cost, what didn't work).
A single completed case delivers pillar 2 for that case — one valid, scored forecast plus the audit trail (final_state.json, trace/) proving how the system produced it. Doing that on three datasets with untouched agent code proves pillar 1; the paper (3) is written once, at the end, over all the evidence.
