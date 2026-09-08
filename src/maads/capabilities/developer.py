"""Developer capabilities — deployment and submission."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from maads.codegen import run_authored_code
from maads.capabilities.data_scientist import _TEXT_HEADER_HELPERS
from maads.capabilities.shared import abspath as _abspath, has_keys as _has_keys
from maads.deltas import StateDelta
from maads.knowledge_setup import append_experience_to_knowledge
from maads.state import CrispDMState
from maads.tools import FileIO, PythonExec

_SUBMISSION_INSTRUCTION = (
    "CRISP-DM 6.1 Build Submission: refit the chosen model approach on the full "
    "training set, generate predictions for the prepared test set, and write "
    "OUTPUT_PATH. When SAMPLE_SUBMISSION is a real file, load it as the "
    "authoritative schema template — column names, dtypes, and row count must "
    "match exactly before writing. When SAMPLE_SUBMISSION is empty, invent a "
    "schema (typically ID_COL plus TARGET or a prediction column) and own that "
    "schema. "
    "Parse CHOSEN_MODEL with load_chosen_model() (or json.loads when it is a string) "
    "in code (pipelines are not persisted from Phase 4). Respect PROBLEM_TYPE and "
    "EVAL_METRIC (e.g. log-transform the target when the metric name contains 'log'). "
    "Classification labels may be strings or integers of any cardinality — encode "
    "for fitting and inverse-transform for the submission when needed. "
    "When TEXT_COLUMN is non-empty, treat this as NLP-primary and use that column "
    "as the main feature (parse FEATURE_HINTS for weak categoricals if needed). "
    "Join predictions to ID_COL from the test records; never reorder or drop rows. "
    "Never treat the sample submission as ground-truth labels."
)

_TEXT_SUBMISSION_BASELINE = """
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_parquet(TRAIN_PARQUET)
test = pd.read_parquet(TEST_PARQUET)
X_train = drop_feature_columns(train)
y, class_values, label_encoder = classification_target_values(train[TARGET])
n_classes = len(set(y.tolist()))
X_test = drop_feature_columns(test)
primary_text = PRIMARY_TEXT_COL if PRIMARY_TEXT_COL in X_train.columns else None
text_cols = [
    c for c in X_train.columns
    if str(X_train[c].dtype) == "object" or str(X_train[c].dtype).startswith("string")
]
if not primary_text:
    primary_text = next((c for c in text_cols if c == "text"), None)
if not primary_text and TEXT_COLUMN and TEXT_COLUMN in X_train.columns:
    primary_text = TEXT_COLUMN
if not primary_text and text_cols:
    primary_text = text_cols[0]
num_cols = X_train.select_dtypes(include="number").columns.tolist()
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
pipe.fit(X_train, y)
encoded_preds = pipe.predict(X_test)
if label_encoder is not None:
    preds = label_encoder.inverse_transform(encoded_preds)
else:
    preds = encoded_preds
if SAMPLE_SUBMISSION:
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    idc = ID_COL if ID_COL in sample.columns else sample.columns[0]
    target_col = TARGET if TARGET in sample.columns else sample.columns[1]
    if ID_COL in test.columns:
        sub = pd.DataFrame({idc: test[ID_COL].values, target_col: preds})
    else:
        sub = pd.DataFrame({idc: sample[idc].values, target_col: preds})
    sub = sub.reindex(columns=list(sample.columns))
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)
else:
    idc = ID_COL if ID_COL in test.columns else (test.columns[0] if len(test.columns) else "id")
    target_col = TARGET or "prediction"
    ids = test[idc].values if idc in test.columns else range(len(preds))
    sub = pd.DataFrame({idc: ids, target_col: preds})
sub.to_csv(OUTPUT_PATH, index=False)
print(json.dumps({"submission_path": OUTPUT_PATH, "rows": int(len(sub))}))
"""

_DEVELOPER_HEADER_HELPERS = """
def load_chosen_model():
    import json
    raw = CHOSEN_MODEL
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else {}
    return raw or {}
"""


def _submission_contract(sample_csv: str) -> callable:
    sample_path = Path(sample_csv)
    sample_cols: list[str] | None = None
    expected_rows: int | None = None
    if sample_path.is_file():
        sample = pd.read_csv(sample_path)
        sample_cols = list(sample.columns)
        expected_rows = len(sample)

    def contract(payload: dict) -> list[str]:
        errors = _has_keys(payload, "submission_path", "rows")
        if errors:
            return errors
        path = Path(str(payload["submission_path"]))
        if not path.is_file():
            return [f"submission file not found: {path}"]
        try:
            sub = pd.read_csv(path)
        except Exception as exc:
            return [f"submission not readable: {exc}"]
        if sample_cols is not None and list(sub.columns) != sample_cols:
            return [f"columns {list(sub.columns)} != sample {sample_cols}"]
        if expected_rows is not None and len(sub) != expected_rows:
            return [f"row count {len(sub)} != sample {expected_rows}"]
        rows = payload.get("rows")
        if not isinstance(rows, int) or rows != len(sub):
            return ["rows must be an int matching the written file"]
        return []

    return contract


def _primary_text_column(feature_hints: dict) -> str:
    for key in ("text_free", "text"):
        cols = feature_hints.get(key)
        if isinstance(cols, list) and cols:
            return str(cols[0])
    return ""


def _is_text_modeling_case(state: CrispDMState) -> bool:
    return bool(_primary_text_column(state.config.feature_hints or {}))


def _run_text_submission_baseline(pyexec: PythonExec, header_vars: dict[str, str]) -> dict:
    from maads.capabilities.data_scientist import _TEXT_HEADER_HELPERS
    from maads.codegen import _header, _last_json_line

    header = _header(header_vars, helpers=_TEXT_HEADER_HELPERS)
    res = pyexec.run(header + _TEXT_SUBMISSION_BASELINE, label="developer_61_text_fallback")
    if not res.ok:
        raise RuntimeError((res.stderr or "text submission baseline failed").strip()[-500:])
    payload = _last_json_line(res.stdout)
    if not payload:
        raise RuntimeError("text submission baseline printed no JSON")
    return payload


def build_submission(
    pyexec: PythonExec,
    state: CrispDMState,
    artifact_dir: Path,
) -> StateDelta:
    dataset_train = state.dp.dataset.get("train")
    dataset_test = state.dp.dataset.get("test")
    if not dataset_train or not dataset_test:
        raise RuntimeError("6.1 requires prepared dataset train and test parquet paths")

    out = str((artifact_dir / "submission.csv").resolve())
    sample = _abspath(state.config.data.sample_submission_csv)
    chosen = state.md.chosen_model.model_dump() if state.md.chosen_model else {}
    feature_hints = state.config.feature_hints or {}
    text_col = _primary_text_column(feature_hints)
    text_case = _is_text_modeling_case(state)
    header_vars = {
        "TRAIN_PARQUET": dataset_train,
        "TEST_PARQUET": dataset_test,
        "TARGET": state.config.target_column,
        "ID_COL": state.config.id_column,
        "PROBLEM_TYPE": state.config.problem_type,
        "EVAL_METRIC": state.config.evaluation_metric,
        "SAMPLE_SUBMISSION": sample,
        "OUTPUT_PATH": out,
        "CHOSEN_MODEL": json.dumps(chosen),
        "FEATURE_HINTS": json.dumps(feature_hints),
        "TEXT_COLUMN": text_col,
        "PRIMARY_TEXT_COL": text_col or "text",
    }

    def _text_fallback() -> dict:
        return _run_text_submission_baseline(pyexec, header_vars)

    res = run_authored_code(
        pyexec=pyexec,
        agent_name="developer",
        state=state,
        instruction=_SUBMISSION_INSTRUCTION,
        header_vars=header_vars,
        header_helpers=(
            (_TEXT_HEADER_HELPERS + "\n" + _DEVELOPER_HEADER_HELPERS) if text_case else _DEVELOPER_HEADER_HELPERS
        ),
        fallback=_text_fallback if text_case else None,
        fallback_code="text_tfidf_logreg_submission_baseline",
        contract=_submission_contract(sample),
        contract_hint=(
            "Required keys: submission_path (str, absolute path to written CSV), "
            "rows (int, must match file and sample submission row count)."
        ),
        artifact_dir=artifact_dir,
    )

    state.dep.submission_path = res.payload["submission_path"]
    state.dep.deployment_plan = (
        f"Agent-authored submission from {chosen.get('technique', 'chosen model')}; "
        f"validated against sample template."
    )
    return StateDelta(["dep.submission_path", "dep.deployment_plan"])


def plan_monitoring(state: CrispDMState) -> StateDelta:
    state.dep.monitoring_and_maintenance_plan = "Re-run on data refresh; watch CV vs leaderboard gap."
    return StateDelta(["dep.monitoring_and_maintenance_plan"])


def experience_review(state: CrispDMState) -> StateDelta:
    loops = [le.label for le in state.loop_history]
    deg = state.degraded_flags
    experience = (
        f"# Experience — {state.case_id}\n\n"
        f"- Loops fired: {loops or 'none'}\n"
        f"- Degraded steps: {deg or 'none'}\n"
        f"- Chosen model: "
        f"{state.md.chosen_model.technique if state.md.chosen_model else 'n/a'}\n"
        f"- CV: {state.md.chosen_model.cv_score if state.md.chosen_model else 'n/a'}\n"
    )
    state.dep.experience_documentation = experience
    append_experience_to_knowledge(state.case_id, experience)
    return StateDelta(["dep.experience_documentation"])
