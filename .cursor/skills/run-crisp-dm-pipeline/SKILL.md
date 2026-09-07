---
name: run-crisp-dm-pipeline
description: Run the maads CRISP-DM pipeline for a bundled case or YAML config, including data download, env, artifacts, and success checks. Use when the user asks to run maads, kick off titanic/house_prices/disaster_tweets, or execute python -m maads run.
---

# Run the CRISP-DM pipeline

## Prerequisites

- venv + `pip install -e ".[dev]"`
- `.env` from `.env.example` with `MODEL` (e.g. `ollama/gemma2:9b`) and/or `OPENAI_API_KEY`
- Kaggle credentials if data is not already under `data/<case>/`

## Checklist

```
- [ ] Activate venv
- [ ] Data present (or download)
- [ ] .env loaded
- [ ] Run with --case or --config
- [ ] Inspect artifacts/<case>/current and status.json
- [ ] Confirm submission / final_state if the run completed
```

## Commands

```bash
source .venv/bin/activate
python -m maads data download --case titanic
python -m maads run --case titanic
# or
python -m maads run --config configs/titanic.yaml --quiet
# override model for one run
python -m maads run --case titanic --model gpt-4o-mini --max-tokens 500000
```

Flags that exist: `--case` XOR `--config`, `--config-dir`, `--artifact-dir`, `--quiet`/`-q`, `--model`, `--max-tokens`.

Exit `0` only if `ml_run_succeeded(state)`. Interrupt is `130`.

## After the run

- Pointer: `artifacts/<case>/current` is a **text file** with the run id.
- Run dir: `artifacts/<case>/runs/<run_id>/` (`status.json`, `final_state.json`, `trace/`, `reports/`).
- Watch live: `status.json` or `python -m maads dashboard --case titanic --no-open`.

Do not call a real LLM from pytest. For a cheap code-path check use the **run-pytest-suite** skill instead.

## Additional resources

- Case YAML fields: [case-yaml.md](case-yaml.md)
