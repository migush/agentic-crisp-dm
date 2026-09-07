---
name: add-demo-case
description: Add or update a Kaggle demo case using YAML config only, keeping agent/crew/flow code case-agnostic. Use when adding a competition, cloning titanic.yaml, or editing feature_hints / success_criterion.
---

# Add or update a demo case

Agent code, crews, prompts, and `CrispDMFlow` must stay **case-agnostic**. New cases are config (+ knowledge markdown, + optional download shorthand).

## Checklist

```
- [ ] Copy configs/titanic.yaml → configs/<case_id>.yaml
- [ ] Fill CaseConfig fields (see below)
- [ ] Optional: knowledge/<case_id>_experience.md
- [ ] Optional: CASE_SHORTHANDS in src/maads/data_utils.py for `maads data download --case`
- [ ] Download data; do not commit data/ or artifacts/
- [ ] Smoke: python -m maads run --case <case_id> (or mocked path-coverage if only wiring)
- [ ] No `if case_id == ...` in src/maads
```

## YAML shape

Mirror `configs/titanic.yaml`:

- Identity: `case_id`, `kaggle_competition`, `problem_statement`
- Modeling: `problem_type`, `target_column`, `id_column`, `evaluation_metric`
- `data.train_csv` / `test_csv` / `sample_submission_csv` (paths under `data/<case_id>/`)
- `feature_hints` (numeric_with_missing, categorical, text_free, …)
- `success_criterion.metric` + `threshold` (Loop C)

NLP vs tabular: put signals in `feature_hints` (e.g. `text_free`, `representation_options`). Capabilities may branch on those keys, never on `case_id`.

## Download wiring

```python
# src/maads/data_utils.py — only if you want --case shorthand
CASE_SHORTHANDS = {
    "titanic": "titanic",
    "house_prices": "house-prices-advanced-regression-techniques",
    "disaster_tweets": "nlp-getting-started",
    # "<case_id>": "<kaggle-slug>",
}
```

Arbitrary competitions: `python -m maads data download --competition <slug>` plus an explicit `--config`.

## Forbidden

Editing `agents.py`, crew `tasks.yaml` formatters, or `CrispDMFlow` to special-case the new dataset.
