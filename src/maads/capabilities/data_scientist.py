"""Data Scientist capabilities — CRISP-DM-independent execution API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from maads.codegen import run_authored_code
from maads.deltas import StateDelta
from maads.capabilities.shared import (
    abspath as _abspath,
    execution_or_llm,
    has_keys as _has_keys,
)
from maads.state import CrispDMState, EvaluationBundle, ModelRun, coerce_evaluation_bundle
from maads.success_criterion import normalize_assessment


def _train_schema_context(dataset_train: str, id_col: str) -> tuple[list[str], str]:
    import pandas as pd

    cols = list(pd.read_parquet(dataset_train).columns)
    id_present = id_col in cols
    note = (
        f"TRAIN_PARQUET columns (measured): {cols}. "
        f"ID_COL ({id_col!r}) is {'present' if id_present else 'absent'} in train "
        "(train parquet intentionally drops identifier columns after prep 3.5; "
        "ID_COL is kept in test only). "
        "Drop features with: "
        "X = train.drop(columns=[c for c in (TARGET, ID_COL) if c in train.columns])."
    )
    return cols, note


def _primary_text_column_name(state: CrispDMState, columns: list[str] | None = None) -> str | None:
    hints = (state.config.feature_hints or {}).get("text_free") or []
    cols = set(columns or [])
    for col in hints:
        if not cols or col in cols:
            return str(col)
    if not cols or "text" in cols:
        return "text"
    return None


def _text_modeling_hint(state: CrispDMState) -> str:
    text_free = (state.config.feature_hints or {}).get("text_free") or []
    if not text_free:
        return ""
    return (
        " Text modeling: if using TfidfVectorizer inside ColumnTransformer, "
        "ColumnTransformer passes 2-D arrays — add "
        "FunctionTransformer(lambda x: x.ravel().astype(str), validate=False) "
        "before TfidfVectorizer, or vectorize train[text_col].astype(str) directly. "
        "Combine text TF-IDF with numeric derived columns via ColumnTransformer when present."
    )


_TEXT_HEADER_HELPERS = """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

def drop_feature_columns(df):
    return df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])

def text_vectorizer_pipeline(**tfidf_kwargs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    def _to_1d_str(x):
        return np.asarray(x).ravel().astype(str)

    return Pipeline([
        ("to_str", FunctionTransformer(_to_1d_str, validate=False)),
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
    ])

def classification_target_values(series):
    import pandas as pd
    s = pd.Series(series)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all() and (numeric == numeric.round()).all():
        y = numeric.astype(int).values
        classes = sorted(set(y.tolist()))
        return y, classes, None
    le = LabelEncoder()
    y = le.fit_transform(s.astype(str).values)
    return y, list(le.classes_), le

def logistic_solver_for(n_classes):
    return "liblinear" if n_classes <= 2 else "lbfgs"
"""


_TEXT_BUILD_MODEL_BASELINE = """
import json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, make_scorer, get_scorer

train = pd.read_parquet(TRAIN_PARQUET)
X = drop_feature_columns(train)
y, class_values, _le = classification_target_values(train[TARGET])
n_classes = len(set(y.tolist()))
train_cols = json.loads(TRAIN_COLUMNS)
primary_text = PRIMARY_TEXT_COL if PRIMARY_TEXT_COL in X.columns else None
text_cols = [
    c for c in train_cols
    if c in X.columns and (str(X[c].dtype) == "object" or str(X[c].dtype).startswith("string"))
]
if not primary_text:
    primary_text = next((c for c in text_cols if c == "text"), None)
if not primary_text and text_cols:
    primary_text = text_cols[0]
num_cols = X.select_dtypes(include="number").columns.tolist()
transformers = []
if primary_text:
    transformers.append((
        primary_text,
        text_vectorizer_pipeline(ngram_range=(1, 2), max_features=50000, min_df=2),
        [primary_text],
    ))
if num_cols:
    transformers.append((
        "num",
        Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
        num_cols,
    ))
pre = ColumnTransformer(transformers, remainder="drop")
technique = "tfidf_logreg"
clf = LogisticRegression(max_iter=2000, solver=logistic_solver_for(n_classes), class_weight="balanced", random_state=42)
pipe = Pipeline([("pre", pre), ("clf", clf)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
if str(METRIC).lower() == "f1":
    if n_classes <= 2 and set(class_values) <= {0, 1}:
        scorer = make_scorer(f1_score, pos_label=1)
    else:
        scorer = make_scorer(f1_score, average="macro")
else:
    scorer = get_scorer(METRIC)
scores = cross_validate(pipe, X, y, cv=cv, scoring={"metric": scorer}, return_train_score=False)
cv_mean = float(scores["test_metric"].mean())
cv_std = float(scores["test_metric"].std())
pipe.fit(X, y)
try:
    n_features = int(pipe.named_steps["pre"].transform(X.iloc[:1]).shape[1])
except Exception:
    n_features = len(num_cols) + (1 if primary_text else 0)
print(json.dumps({"technique": technique, "cv_score": cv_mean, "cv_std": cv_std, "n_features": n_features}))
"""


_TEXT_ASSESS_MODEL_BASELINE = """
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, f1_score,
)
from sklearn.base import clone

train = pd.read_parquet(TRAIN_PARQUET)
X = drop_feature_columns(train)
y, class_values, _le = classification_target_values(train[TARGET])
n_classes = len(set(y.tolist()))
train_cols = json.loads(TRAIN_COLUMNS)
primary_text = PRIMARY_TEXT_COL if PRIMARY_TEXT_COL in X.columns else None
text_cols = [
    c for c in train_cols
    if c in X.columns and (str(X[c].dtype) == "object" or str(X[c].dtype).startswith("string"))
]
if not primary_text:
    primary_text = next((c for c in text_cols if c == "text"), None)
if not primary_text and text_cols:
    primary_text = text_cols[0]
num_cols = X.select_dtypes(include="number").columns.tolist()
transformers = []
if primary_text:
    transformers.append((
        primary_text,
        text_vectorizer_pipeline(ngram_range=(1, 2), max_features=50000, min_df=2),
        [primary_text],
    ))
if num_cols:
    transformers.append((
        "num",
        Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
        num_cols,
    ))
pre = ColumnTransformer(transformers, remainder="drop")
clf = LogisticRegression(max_iter=2000, solver=logistic_solver_for(n_classes), class_weight="balanced", random_state=42)
pipe = Pipeline([("pre", pre), ("clf", clf)])
technique = MODEL_TECHNIQUE or "tfidf_logreg"
figures_dir = FIGURES_DIR
os.makedirs(figures_dir, exist_ok=True)
class_labels_map = json.loads(CLASS_LABELS) if isinstance(CLASS_LABELS, str) else (CLASS_LABELS or {})
cm_labels = sorted(set(y.tolist()))
binary_01 = n_classes <= 2 and _le is None and set(class_values) <= {0, 1}
label_names = []
for l in cm_labels:
    if _le is not None:
        label_names.append(str(class_values[l]))
    else:
        label_names.append(class_labels_map.get(str(l), str(l)))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y), dtype=int)
cv_scores = []
for tr_idx, va_idx in cv.split(X, y):
    est = clone(pipe)
    est.fit(X.iloc[tr_idx], y[tr_idx])
    preds = est.predict(X.iloc[va_idx])
    oof_pred[va_idx] = preds
    if binary_01:
        cv_scores.append(float(f1_score(y[va_idx], preds, pos_label=1)))
    else:
        cv_scores.append(float(f1_score(y[va_idx], preds, average="macro")))

cm = confusion_matrix(y, oof_pred, labels=cm_labels).tolist()
prec, rec, f1, support = precision_recall_fscore_support(
    y, oof_pred, labels=cm_labels, average=None, zero_division=0,
)
if binary_01 and 1 in cm_labels:
    metrics_f1 = float(f1[cm_labels.index(1)])
else:
    metrics_f1 = float(f1_score(y, oof_pred, average="macro"))
metrics = {
    "accuracy": float(accuracy_score(y, oof_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
    "f1": metrics_f1,
}
for i, lbl in enumerate(cm_labels):
    name = label_names[i]
    metrics[f"precision_{name}"] = float(prec[i])
    metrics[f"recall_{name}"] = float(rec[i])
    metrics[f"f1_{name}"] = float(f1[i])

fig_paths = []
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(cm_labels)))
ax.set_yticks(range(len(cm_labels)))
ax.set_xticklabels(label_names)
ax.set_yticklabels(label_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion matrix (OOF)")
for i in range(len(cm)):
    for j in range(len(cm[i])):
        ax.text(j, i, str(cm[i][j]), ha="center", va="center")
p_cm = os.path.join(figures_dir, "confusion_matrix.png")
fig.tight_layout()
fig.savefig(p_cm, dpi=150)
plt.close(fig)
fig_paths.append(p_cm)

evaluation_bundle = {
    "problem_type": PROBLEM_TYPE,
    "metrics": metrics,
    "confusion_matrix": cm,
    "class_labels": class_labels_map,
    "cv": {"mean": float(np.mean(cv_scores)), "std": float(np.std(cv_scores)), "n_folds": 5},
    "figures": fig_paths,
    "warnings": [],
}
print(json.dumps({
    "evaluation_bundle": evaluation_bundle,
    "assessment": f"OOF evaluation via {technique} baseline",
}))
"""


def _is_text_modeling_case(state: CrispDMState) -> bool:
    return bool((state.config.feature_hints or {}).get("text_free"))


def _run_text_model_baseline(pyexec, header_vars: dict[str, Any]) -> dict[str, Any]:
    from maads.codegen import _header, _last_json_line

    header = _header(header_vars, helpers=_TEXT_HEADER_HELPERS)
    res = pyexec.run(header + _TEXT_BUILD_MODEL_BASELINE, label="data_scientist_43_text_fallback")
    if not res.ok:
        raise RuntimeError((res.stderr or "text model baseline failed").strip()[-500:])
    payload = _last_json_line(res.stdout)
    if not payload:
        raise RuntimeError("text model baseline printed no JSON")
    return payload


def _model_technique_from_state(state: CrispDMState) -> str:
    if state.md.models:
        best = max(state.md.models, key=lambda m: m.cv_score or 0.0)
        return best.technique or state.md.modeling_technique or "tfidf_logreg"
    return state.md.modeling_technique or "tfidf_logreg"


def _run_text_assess_baseline(pyexec, header_vars: dict[str, Any]) -> dict[str, Any]:
    from maads.codegen import _header, _last_json_line

    header = _header(header_vars, helpers=_TEXT_HEADER_HELPERS)
    res = pyexec.run(header + _TEXT_ASSESS_MODEL_BASELINE, label="data_scientist_44_text_fallback")
    if not res.ok:
        raise RuntimeError((res.stderr or "text assess baseline failed").strip()[-500:])
    payload = _last_json_line(res.stdout)
    if not payload:
        raise RuntimeError("text assess baseline printed no JSON")
    return payload


def _evaluation_bundle_contract(
    payload: dict[str, Any],
    *,
    problem_type: str,
    class_labels: dict[str, str],
) -> list[str]:
    key_errors = _has_keys(payload, "evaluation_bundle")
    if key_errors:
        return key_errors
    raw = payload.get("evaluation_bundle")
    if not isinstance(raw, dict):
        return ["evaluation_bundle must be an object"]
    try:
        EvaluationBundle.model_validate(
            coerce_evaluation_bundle(
                raw,
                problem_type=problem_type,
                class_labels=class_labels,
            ),
        )
    except ValidationError as exc:
        return [str(exc).split("\n")[0]]
    return []


def execution_evidence(
    pyexec,
    state: CrispDMState,
    substep: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Author and run code for DS-owned execution substeps (no baseline fallback)."""
    train = _abspath(state.config.data.train_csv)
    target = state.resolved_target()

    if substep == "2.3":
        res = run_authored_code(
            pyexec=pyexec, agent_name="data_scientist", state=state,
            instruction="CRISP-DM 2.3 Explore Data: probe the training data to inform "
                        "modeling — target balance, feature/target relationships, "
                        "notable distributions. Compute from the data, do not guess.",
            header_vars={"TRAIN_CSV": train, "TARGET": target},
            contract=lambda p: _has_keys(p, "n_rows", "target"),
            contract_hint="Required keys: n_rows (int), target (str); add any findings "
                          "(e.g. target_distribution, correlations).",
            artifact_dir=artifact_dir,
        )
        return {"data_exploration_report": res.payload}

    if substep == "4.3":
        dataset_train = state.dp.dataset.get("train")
        if not dataset_train:
            return {}
        idc = state.config.id_column
        metric = state.config.evaluation_metric
        problem_type = state.config.problem_type
        technique_hint = state.md.modeling_technique or "your choice"
        train_columns, schema_note = _train_schema_context(dataset_train, idc)
        header_vars = {
            "TRAIN_PARQUET": dataset_train,
            "TRAIN_COLUMNS": json.dumps(train_columns),
            "TARGET": target,
            "ID_COL": idc,
            "METRIC": metric,
            "PROBLEM_TYPE": problem_type,
            "PRIMARY_TEXT_COL": _primary_text_column_name(state, train_columns) or "text",
        }
        text_case = _is_text_modeling_case(state)

        def _text_fallback() -> dict[str, Any]:
            return _run_text_model_baseline(pyexec, header_vars)

        res = run_authored_code(
            pyexec=pyexec, agent_name="data_scientist", state=state,
            instruction="CRISP-DM 4.3 Build Model: write and execute Python that reads "
                        "TRAIN_PARQUET, builds a scikit-learn pipeline appropriate for "
                        "PROBLEM_TYPE, and evaluates it with cross-validation scored on "
                        f"METRIC. Prefer the technique from 4.1 ({technique_hint}) when "
                        "suitable. "
                        f"{schema_note}"
                        f"{_text_modeling_hint(state)} "
                        "You must train and score the model in code — do not claim results "
                        "you did not run.",
            header_vars=header_vars,
            header_helpers=_TEXT_HEADER_HELPERS if text_case else "",
            fallback=_text_fallback if text_case else None,
            fallback_code="text_tfidf_logreg_baseline",
            contract=lambda p: (
                _has_keys(p, "technique", "cv_score", "cv_std")
                or ([] if isinstance(p.get("cv_score"), (int, float)) and isinstance(p.get("cv_std"), (int, float))
                    else ["cv_score and cv_std must be numeric"])
            ),
            contract_hint="Required keys: technique (str), cv_score (float), cv_std (float), n_features (int).",
            artifact_dir=artifact_dir,
        )
        p = res.payload
        return {
            "model_run": {
                "technique": p.get("technique") or "unspecified",
                "cv_score": p.get("cv_score"),
                "cv_std": p.get("cv_std"),
                "description": f"{p.get('n_features', '?')} features, CV",
                "parameter_settings": p.get("parameter_settings") or {},
            },
        }

    if substep == "4.4":
        dataset_train = state.dp.dataset.get("train")
        if not dataset_train:
            return {}
        idc = state.config.id_column
        problem_type = state.config.problem_type
        figures_dir = str((artifact_dir / "figures").resolve())
        class_labels_map = state.config.class_labels or {}
        train_columns, schema_note = _train_schema_context(dataset_train, idc)
        model_technique = _model_technique_from_state(state)
        text_case = _is_text_modeling_case(state)
        header_vars = {
            "TRAIN_PARQUET": dataset_train,
            "TRAIN_COLUMNS": json.dumps(train_columns),
            "TARGET": target,
            "ID_COL": idc,
            "PROBLEM_TYPE": problem_type,
            "FIGURES_DIR": figures_dir,
            "CLASS_LABELS": json.dumps(class_labels_map),
            "MODEL_TECHNIQUE": model_technique,
            "PRIMARY_TEXT_COL": _primary_text_column_name(state, train_columns) or "text",
        }
        bundle_contract = lambda p: _evaluation_bundle_contract(
            p,
            problem_type=problem_type,
            class_labels=class_labels_map,
        )

        def _text_fallback() -> dict[str, Any]:
            return _run_text_assess_baseline(pyexec, header_vars)

        res = run_authored_code(
            pyexec=pyexec, agent_name="data_scientist", state=state,
            instruction="CRISP-DM 4.4 Assess Model: write and execute Python that reads "
                        "TRAIN_PARQUET, rebuilds the pipeline for MODEL_TECHNIQUE from 4.3 "
                        "(pipelines are not persisted between substeps — reconstruct in code), "
                        "produces out-of-fold predictions via stratified CV, computes "
                        "problem-type-appropriate metrics (for classification of any "
                        "cardinality: accuracy, balanced accuracy, per-class precision/"
                        "recall/F1, confusion matrix over the observed labels — do not "
                        "hardcode [0, 1]; for regression: error metrics matching EVAL_METRIC), "
                        "saves figures under FIGURES_DIR using matplotlib only (no seaborn), "
                        "and prints evaluation_bundle. Use CLASS_LABELS for human-readable names. "
                        "evaluation_bundle must include problem_type, metrics (flat float map), "
                        "confusion_matrix, class_labels, figures (list of paths), and optional cv. "
                        f"{schema_note}"
                        f"{_text_modeling_hint(state)}",
            header_vars=header_vars,
            header_helpers=_TEXT_HEADER_HELPERS if text_case else "",
            fallback=_text_fallback if text_case else None,
            fallback_code="text_tfidf_logreg_assess_baseline",
            contract=bundle_contract,
            contract_hint=(
                "Required key: evaluation_bundle with problem_type, metrics (numbers only), "
                "confusion_matrix, figures (list of path strings). "
                "Do not nest per_class metrics or use a dict for figures."
            ),
            artifact_dir=artifact_dir,
        )
        p = res.payload
        return {
            "evaluation_bundle": p.get("evaluation_bundle"),
            "assessment": p.get("assessment"),
        }
    return {}


_DS_EXECUTION_AUTHORITY_KEYS: dict[str, tuple[str, ...]] = {
    "2.3": ("data_exploration_report",),
    "4.3": ("model_run",),
    "4.4": ("evaluation_bundle",),
}


def _execution_authoritative(execution: dict[str, Any], substep: str) -> bool:
    return any(
        execution.get(key) is not None
        for key in _DS_EXECUTION_AUTHORITY_KEYS.get(substep, ())
    )


def apply_response(
    data: dict,
    state: CrispDMState,
    substep: str,
    execution: dict[str, Any],
) -> StateDelta:
    """Map data-scientist JSON (or execution evidence) into shared state."""
    from maads.output_contracts import validate_agent_output

    if not _execution_authoritative(execution, substep):
        schema_errors = validate_agent_output("data_scientist", data, substep=substep)
        if schema_errors:
            return StateDelta(
                notes=f"DS {substep}: schema-invalid response: {schema_errors[0]}",
                failed=True,
            )

    su = (data or {}).get("state_updates") or {}
    du = su.get("du") or {}
    md = su.get("md") or {}
    ev = su.get("ev") or {}
    fields: list[str] = []

    if substep == "2.3":
        report = execution_or_llm(execution, du, "data_exploration_report")
        if not report:
            desc = state.du.data_description_report or {}
            report = {"n_rows": desc.get("n_rows"), "target": state.resolved_target()}
        state.du.data_exploration_report = report
        fields.append("du.data_exploration_report")
        inferred = (report or {}).get("target") if isinstance(report, dict) else None
        if state.adopt_target_if_blank(inferred if isinstance(inferred, str) else None):
            fields.append("config.target_column")
    elif substep == "4.1":
        state.md.modeling_technique = md.get("modeling_technique") or "to be chosen at 4.3"
        state.md.modeling_assumptions = md.get("modeling_assumptions") or [
            "tabular features", "no leakage (pipeline fit on train only)",
        ]
        fields.extend(["md.modeling_technique", "md.modeling_assumptions"])
    elif substep == "4.2":
        state.md.test_design = md.get("test_design") or {
            "cv": "stratified_5fold",
            "metric": state.config.evaluation_metric,
        }
        fields.append("md.test_design")
    elif substep == "4.3":
        run = dict(execution.get("model_run") or {})
        llm_run = md.get("model_run") or {}
        if llm_run.get("description") and not run.get("description"):
            run["description"] = llm_run["description"]
        if not run:
            return StateDelta(notes="DS 4.3: no model execution evidence")
        technique = run.get("technique") or "unspecified"
        state.md.models.append(ModelRun(
            technique=technique,
            cv_score=run.get("cv_score"),
            cv_std=run.get("cv_std"),
            description=run.get("description") or "model run",
            parameter_settings=run.get("parameter_settings") or {},
        ))
        state.md.modeling_technique = technique
        fields.extend(["md.models", "md.modeling_technique"])
    elif substep == "4.4":
        bundle_raw = execution.get("evaluation_bundle")
        if state.md.models:
            best = max(state.md.models, key=lambda m: m.cv_score or 0.0)
            chosen = md.get("chosen_model_technique")
            if chosen:
                for m in state.md.models:
                    if m.technique == chosen:
                        best = m
                        break
            best.assessment = (
                execution.get("assessment")
                or md.get("assessment")
                or "selected: best CV score"
            )
            if bundle_raw:
                try:
                    best.evaluation_bundle = EvaluationBundle.model_validate(
                        coerce_evaluation_bundle(
                            bundle_raw,
                            problem_type=state.config.problem_type,
                            class_labels=state.config.class_labels or {},
                        ),
                    )
                except ValidationError as exc:
                    return StateDelta(
                        notes=f"DS 4.4: invalid evaluation_bundle: {exc}",
                        failed=True,
                    )
            state.md.chosen_model = best
            fields.append("md.chosen_model")
        elif not bundle_raw:
            return StateDelta(notes="DS 4.4: no evaluation_bundle from execution", failed=True)
    elif substep == "5.1":
        cv = state.md.chosen_model.cv_score if state.md.chosen_model else None
        sc = state.config.success_criterion
        raw = ev.get("assessment_of_dm_results") or {
            "cv_score": cv,
            "threshold": sc.threshold,
        }
        state.ev.assessment_of_dm_results = normalize_assessment(
            raw,
            metric=sc.metric,
            threshold=sc.threshold,
            direction=sc.direction,
            cv_score=cv,
        )
        fields.append("ev.assessment_of_dm_results")
        if state.md.chosen_model:
            state.ev.approved_models = [state.md.chosen_model]
            fields.append("ev.approved_models")

    summary = (data or {}).get("summary", "")
    return StateDelta(fields, notes=summary or f"DS completed {substep}")
