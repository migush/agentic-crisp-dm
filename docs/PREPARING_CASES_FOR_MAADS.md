# Preparing a Case for MAADS

MAADS runs supervised tabular or text **classification** (any number of discrete labels) and **regression**. Hosted users upload **raw CSVs plus a problem description**. Data Understanding and Data Preparation are agent work during the CRISP-DM run — not an upload checklist.

Bundled demos (Titanic, House Prices, Disaster Tweets) still use a Kaggle-shaped package. That shape is the **demo/operator** contract, not a requirement for the Cases page.

## Hosted users (Cases page)

1. Open **Cases** and describe the problem in your own words (optional PDF or markdown).
2. Upload the CSVs you have. UTF-8 is preferred; latin-1 and other common 8-bit encodings are accepted. A separate test file and sample submission are optional.
3. Review observational facts: column names, row counts, encoding, missingness, unique-value counts. You may name a suspected target or ID, or leave that to the agents. Messy files can still be marked ready.
4. Mark the case ready. Launch it from **Tasks** with any stored OpenAI or Ollama Cloud key and any live model. The same case can be launched many times.

The product does **not** encode labels, split files, drop test targets, or write a sample submission at upload time. Project Manager, Domain, Data Engineer, Data Scientist, and Developer decide encoding, splits, holdout use, feature treatment, and submission shape during the run.

V1 is not runnable for clustering/segmentation, association rules, description-only studies, images/audio, multi-table relational cases, or true forecasting. The Cases page labels those as not runnable rather than failing silently.

Per-user files live under `data/users/<id>/cases/<case_id>/` (`raw/` plus `case.yaml`). Prepared tables are written only under the run artifact directory.

## Demo / operator contract (Kaggle-shaped YAML)

Operators who install a bundled demo under `configs/` and `data/` should still provide:

1. `train.csv` — labelled examples used to train and validate models.
2. `test.csv` — unlabelled examples for which MAADS must make predictions.
3. `sample_submission.csv` — the exact schema and row order required for predictions.
4. `configs/<case_id>.yaml` — problem description, column roles, metric, and file locations.

This contract is based on the three demonstration cases: Titanic, House Prices, and Disaster Tweets.

## 1. What the demonstration cases have in common

| Case | Problem type | Train shape | Test shape | Target | ID | Submission columns | Metric |
|---|---|---:|---:|---|---|---|---|
| Titanic | Binary classification | 891 × 12 | 418 × 11 | `Survived` | `PassengerId` | `PassengerId`, `Survived` | Accuracy |
| House Prices | Regression | 1,460 × 81 | 1,459 × 80 | `SalePrice` | `Id` | `Id`, `SalePrice` | RMSE on log price |
| Disaster Tweets | Binary text classification | 7,613 × 5 | 3,263 × 4 | `target` | `id` | `id`, `target` | F1 |

In every demo:

- one row represents one observation;
- the training file contains the ID, predictors, and target;
- the test file contains the same ID and predictors, but not the target;
- the target is the only training column absent from the test file;
- the ID is present, non-null, and unique within each file;
- the sample submission has exactly one row per test row;
- submission IDs match the test IDs in the same order;
- the first submission column is the ID and the other column is the prediction;
- predictor data may be numeric, categorical, free text, or a mixture;
- missing predictor values are allowed, but missing target or ID values are not.

The examples also show that MAADS can handle narrow tabular data, wide mixed-type tabular data, and text-led data. Missing values are normal: do not replace them blindly when a missing value has domain meaning, such as “no garage” or “no basement.”

## 2. Supported case scope

The modelling flow supports:

- `classification` / `binary_classification` / `multiclass_classification` — discrete labels of any cardinality (strings or integers);
- `regression` — a numeric target.

For the safest demo results, use one of the demonstrated metrics:

| Problem type | `evaluation_metric` | Direction |
|---|---|---|
| Classification | `accuracy` | `maximize` |
| Classification | `f1` | `maximize` |
| Regression with a positive, right-skewed target | `rmse_log` | `minimize` |

Clustering, association/dependency, description-only studies, ranking, images/audio, multi-table relational cases, and true forecasting are out of V1.

## 3. Demo `train.csv`

Requirements for bundled demos:

- Use a comma-delimited CSV with a single header row and UTF-8 text.
- Give every column a non-empty, unique, case-sensitive name.
- Include exactly one target column named in `case.yaml`.
- Include exactly one ID column named in `case.yaml`.
- Include at least one predictor column besides the ID and target.
- Make every target value non-null.
- For classification, labels may be integers or strings; agents encode them.
- For regression, use finite numeric target values. If using `rmse_log`, targets and predictions must be valid for the competition’s log transform; normally this means non-negative values.
- Make IDs non-null and unique within the training file.
- Keep one observation per row at the granularity described in the problem statement.
- Do not add an exported DataFrame index such as `Unnamed: 0`.

Predictors may contain null values. Preserve raw categories and text when possible; MAADS performs cleaning and feature preparation during the CRISP-DM run. If a sentinel such as `NA`, `None`, `0`, or an empty string means “feature absent” rather than “unknown,” document that in `feature_hints`.

## 4. Demo `test.csv`

- Use the same CSV encoding, delimiter, quoting rules, and header spelling as `train.csv`.
- Include the same predictor columns as the training file.
- Include the same ID column.
- Do **not** include the target column.
- Do not introduce test-only columns.
- Make test IDs non-null and unique.
- Keep test rows in the intended submission order.

The recommended column relationship is:

```text
set(train columns) - {target column} == set(test columns)
```

Hosted Cases uploads are not required to follow this. Agents decide whether a labelled second file is holdout, leakage, or unused, and whether to create a split when only one table exists.

## 5. Demo `sample_submission.csv`

For demos, the sample submission is a schema contract. MAADS uses it to validate the final `submission.csv`.

When a hosted user does not upload one, the Developer invents a schema (typically identifier plus prediction) and owns that schema.

Demo requirements:

- Include exactly the columns required by the destination system, with exact spelling, capitalization, and order.
- Include exactly as many rows as `test.csv`.
- Copy the test IDs exactly and in the same order.
- Use a prediction column whose name matches the required output target.
- Do not write a DataFrame index.

## 6. Write `case.yaml` (demo / operator)

Use a lowercase snake-case identifier, for example `customer_churn`. The configuration filename should be the same identifier: `customer_churn.yaml`.

```yaml
case_id: customer_churn
kaggle_competition: customer-churn
problem_statement: |
  Predict whether an active customer will churn during the next 30 days.
  Each row represents one customer at the prediction cutoff date.

problem_type: binary_classification
target_column: churned
id_column: customer_id
evaluation_metric: f1

data:
  train_csv: data/customer_churn/train.csv
  test_csv: data/customer_churn/test.csv
  sample_submission_csv: data/customer_churn/sample_submission.csv

feature_hints:
  numeric_with_missing: [age, monthly_spend]
  categorical: [plan, country]
  text_free: [support_notes]

class_labels:
  "0": "Retained"
  "1": "Churned"

success_criterion:
  metric: f1
  threshold: 0.75
  direction: maximize
```

User cases written by the Cases page are a **source bundle**: `problem_statement`, paths under `data.sources`, optional `train_csv` / `test_csv` / `sample_submission_csv`, and optional user hints (suspected target/ID). `data.test_csv` and `data.sample_submission_csv` may be omitted.

Field meanings:

| Field | Meaning |
|---|---|
| `case_id` | Stable case identifier. Use lowercase letters, digits, and underscores. |
| `kaggle_competition` | Kaggle competition slug, or empty for a hosted user case. |
| `problem_statement` | What must be predicted, what one row represents, when prediction occurs, and why it matters. |
| `problem_type` | `classification`, `binary_classification`, `multiclass_classification`, or `regression`. |
| `target_column` | Suspected target header; agents may override from evidence. |
| `id_column` | Suspected identifier header. |
| `evaluation_metric` | Metric used during model validation. |
| `data.sources` | Raw uploaded files (user cases). |
| `data.train_csv` / `test_csv` / `sample_submission_csv` | Required for demos; test and sample are optional for user cases. |
| `feature_hints` | Optional domain hints. |
| `class_labels` | Human-readable labels for encoded classes. |
| `success_criterion` | Threshold used by Evaluation / Loop C. |

## 7. Describe the case well

A good `problem_statement` should answer:

1. What entity does one row represent?
2. What value must be predicted?
3. At what point in time is the prediction made?
4. Which metric determines success?
5. What business or scientific decision uses the prediction?
6. Which fields are unavailable at prediction time or could leak the answer?

## 8. Layout

Hosted user case:

```text
data/users/<id>/cases/<case_id>/
├── case.yaml
└── raw/
    ├── observations.csv
    └── notes.md            # optional
```

Demo / operator install:

```text
configs/customer_churn.yaml
data/customer_churn/train.csv
data/customer_churn/test.csv
data/customer_churn/sample_submission.csv
```

## 9. Local MAADS smoke test

Demo:

```bash
.venv/bin/python -c "from pathlib import Path; from maads.config import load_case_config; print(load_case_config(Path('configs/titanic.yaml')))"
.venv/bin/python -m maads run --config configs/titanic.yaml
```

User case (after creating it on the Cases page, or with an explicit YAML):

```bash
.venv/bin/python -m maads run --config data/users/<id>/cases/<case_id>/case.yaml
```

A complete run should produce a per-run artifact directory containing prepared data, evaluation evidence, a final report, and a `submission.csv` when predictions are part of the goal.

## 10. Frequent failure causes

- Target missing from the labelled table, misspelled, or containing null values when the user named it.
- Train and test predictor names differ by capitalization or whitespace (demos).
- Duplicate or null IDs in a demo package.
- Sample submission IDs sorted differently from demo test IDs.
- Regression target stored as formatted text such as `$125,000`.
- Delimiter is semicolon or tab while the file is named `.csv` (inspect reports the sniffed delimiter; agents can still parse it).
- A post-outcome or future-information column leaks the target.

Passing structural checks makes a case loadable; it does not guarantee a useful model.
