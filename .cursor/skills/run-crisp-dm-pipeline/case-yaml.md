# Case YAML

`CaseConfig` (`src/maads/config.py`) fields used by `load_case_config`:

| Field | Purpose |
|---|---|
| `case_id` | Artifact folder name |
| `kaggle_competition` | Kaggle slug |
| `problem_statement` | Domain prompt |
| `problem_type` | e.g. `binary_classification` |
| `target_column`, `id_column`, `evaluation_metric` | Modeling contract |
| `data.train_csv` / `test_csv` / `sample_submission_csv` | Paths relative to repo root |
| `feature_hints` | The **only** place for case-specific column knowledge |
| `class_labels` | Optional |
| `success_criterion.metric` / `threshold` | Evaluation / Loop C |

Bundled files: `configs/titanic.yaml`, `house_prices.yaml`, `disaster_tweets.yaml`, `titanic_loopdemo.yaml`.

`maads data download --case X` requires `CASE_SHORTHANDS` in `src/maads/data_utils.py`. Otherwise use `--competition <slug>`.
