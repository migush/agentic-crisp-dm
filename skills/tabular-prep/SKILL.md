# Tabular Data Preparation

Implemented by `maads.capabilities.ml_tools` (agents orchestrate; tools execute).

- Profile dtypes and missingness before transforming.
- Impute numeric with median; categorical with mode or "Unknown".
- `na_means_absent`: fill as structural absence, not a quality blocker.
- Encode categoricals consistently across train and test (OneHot inside model pipeline).
- Align train/test columns; test may lack target column.
- Drop high-cardinality IDs from train features; keep ID in test for submission only.
- Optional derived: `{col}_missing` for `high_missing`, product of first two `numeric_with_missing`.
- Persist parquet stages: clean → constructed → integrated → final train/test.
