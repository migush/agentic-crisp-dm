---
name: run-pytest-suite
description: Run maads tests — fast mocked path-coverage first, then full pytest, dashboard, or webapp slices. Use when validating a change, before a PR, or when the user asks to run pytest/coverage.
---

# Run pytest

No live LLM. Root `conftest.py` stubs `maads.crew.run_text_task` and sets `MAADS_SKIP_MODEL_PROBE=1`.

## Fast gate (~1 min) — always first for flow/agent/capability changes

```bash
source .venv/bin/activate
pip install -e ".[dev]"
MAADS_TRACE=0 coverage run -m pytest tests/test_path_coverage.py -q
coverage report --show-missing
```

## Full suite

```bash
pytest
```

## Slices

```bash
pip install -e ".[dev,dashboard,webapp]"
pytest tests/dashboard/ -q
pytest tests/webapp/ -q
pytest tests/test_flow_happy_path.py -q
pytest tests/test_flow_happy_path.py::test_name -q
```

`tests/test_dashboard_store.py` is at suite root, not under `tests/dashboard/`.

## Frontend

No unit-test runner. Typecheck:

```bash
cd dashboard && npm run build
cd webapp/frontend && npm run build
```

## Do not

- Unset the autouse LLM stub for “more realistic” unit tests
- Expect GitHub Actions to run pytest (deploy workflow only)
- Invoke ruff/eslint/prettier — they are not project tools
