# NLP Data Preparation

Implemented by `maads.capabilities.ml_tools` when `feature_hints.text_free` is set.

- Primary signal is free text; keyword/location may be sparse or missing.
- Test set typically has no target column.
- TF-IDF + linear model is the default baseline (`tfidf_logreg`); honor `representation_options` order.
- Vectorizer + `.ravel()` handling lives in the tool — do not re-author ColumnTransformer TF-IDF glue.
- Use the same vocabulary fit on train when transforming test (persisted pipeline artifact).
- Handle empty strings; lowercase/normalize inside the vectorizer path.
- Do not fine-tune transformers unless a future config explicitly requires it.
