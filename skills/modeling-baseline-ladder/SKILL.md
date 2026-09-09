# Modeling Baseline Ladder

Prefer deterministic `ml_tools` baselines over freeform scripts.

## Technique order (config-driven)

1. If `feature_hints.representation_options` is set — try those in order (budget: first 3).
2. Else if `feature_hints.text_free` — `tfidf_logreg` first.
3. Else classification — `logistic_regression` → `random_forest` → `hist_gradient_boosting`.
4. Else regression — `ridge` → `hist_gradient_boosting`.

## Escalation

| Stage | When |
|---|---|
| Baseline | Always run first; persist joblib artifact |
| Alternate ladder step | Prior CV fails contract or misses threshold |
| Limited HPO | Only if YAML opts in later (`model_search`) — not default |
| Authored code | Novel transforms only; not for standard CV |

Select with direction-aware comparison (minimize RMSE/MAE; maximize accuracy/F1/AUC).
Deploy the exact artifact digest from 4.3 — never rebuild from technique name alone.
