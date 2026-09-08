# Kaggle Submission Contract

When a sample submission file exists:

- Match `sample_submission.csv` columns exactly (order and names).
- Row count must equal test set rows.
- ID column: correct dtype (usually int).
- Prediction column: correct dtype (int for encoded class ids, string for original class names, float for regression).
- Write CSV without index column.

When no sample submission is provided, the Developer owns the schema they write (typically identifier plus prediction). Classification labels may be strings or integers of any cardinality; encode for fitting and decode for the file if the destination expects original labels.

