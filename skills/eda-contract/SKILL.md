# EDA Contract

Deterministic profiling owns measured facts. Agents interpret; they do not invent counts.

## Required exploration fields (2.3)

| Field | Meaning |
|---|---|
| `n_rows` | Train row count |
| `target` | Target column name |
| `target_distribution` | Class/value counts when available |
| `correlations` | Top numeric correlations with target (optional) |
| `constant_columns` | Columns with ≤1 unique value |
| `source` | Must cite `deterministic explore_report` when tools ran |

## Quality (2.4)

- `blockers`: missing target, constant predictors, duplicate IDs, undocumented ≥60% missingness
- `tolerable`: columns in `na_means_absent` or `high_missing`, mild missingness
- Never mark `na_means_absent` or `high_missing` columns as blockers
