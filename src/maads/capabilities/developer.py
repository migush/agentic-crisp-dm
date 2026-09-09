"""Developer capabilities — deployment and submission."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from maads.capabilities import ml_tools
from maads.capabilities.shared import abspath as _abspath, has_keys as _has_keys, record_degraded
from maads.codegen import run_authored_code
from maads.deltas import StateDelta
from maads.knowledge_setup import append_experience_to_knowledge
from maads.state import CrispDMState
from maads.tools import FileIO, PythonExec

_SUBMISSION_INSTRUCTION = (
    "CRISP-DM 6.1 Build Submission: prefer loading CHOSEN_MODEL.artifact_path when "
    "present. Otherwise refit the chosen model on the full training set, generate "
    "predictions for the prepared test set, and write OUTPUT_PATH. When "
    "SAMPLE_SUBMISSION is a real file, load it as the authoritative schema template "
    "— column names, dtypes, and row count must match exactly before writing. When "
    "SAMPLE_SUBMISSION is empty, invent a schema (typically ID_COL plus TARGET). "
    "Parse CHOSEN_MODEL with load_chosen_model() when no artifact exists. Respect "
    "PROBLEM_TYPE and EVAL_METRIC. Join predictions to ID_COL; never reorder rows. "
    "Never treat the sample submission as ground-truth labels."
)

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


def _deterministic_submission_fallback(
    state: CrispDMState,
    *,
    train_parquet: str,
    test_parquet: str,
    sample: str,
    output_path: str,
    artifact_dir: Path,
) -> dict:
    technique = (
        state.md.chosen_model.technique
        if state.md.chosen_model
        else state.md.modeling_technique
        or ("tfidf_logreg" if _is_text_modeling_case(state) else "logistic_regression")
    )
    run = ml_tools.run_experiment(
        train_parquet,
        technique=technique,
        target=state.resolved_target(),
        id_column=state.config.id_column,
        metric=state.config.evaluation_metric,
        problem_type=state.config.problem_type,
        feature_hints=state.config.feature_hints or {},
        artifact_dir=artifact_dir,
    )
    artifact = run.get("artifact_path")
    if not artifact:
        raise RuntimeError("deterministic submission fallback produced no artifact")
    return ml_tools.predict_from_artifact(
        artifact,
        train_parquet=train_parquet,
        test_parquet=test_parquet,
        sample_submission=sample,
        output_path=output_path,
        target=state.resolved_target(),
        id_column=state.config.id_column,
    )


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
    contract = _submission_contract(sample)

    artifact_path = None
    if state.md.chosen_model:
        artifact_path = state.md.chosen_model.artifact_path or (
            (state.md.chosen_model.parameter_settings or {}).get("artifact_path")
        )
    if artifact_path and Path(str(artifact_path)).is_file():
        payload = ml_tools.predict_from_artifact(
            str(artifact_path),
            train_parquet=dataset_train,
            test_parquet=dataset_test,
            sample_submission=sample,
            output_path=out,
            target=state.resolved_target(),
            id_column=state.config.id_column,
        )
        errors = contract(payload)
        if not errors:
            state.dep.submission_path = payload["submission_path"]
            state.dep.deployment_plan = (
                f"Loaded exact artifact {artifact_path} "
                f"({chosen.get('technique', 'chosen model')}); validated against sample."
            )
            return StateDelta(["dep.submission_path", "dep.deployment_plan"])
        record_degraded(state, "6.1", "developer", f"artifact submission invalid: {errors[0]}")

    try:
        payload = _deterministic_submission_fallback(
            state,
            train_parquet=dataset_train,
            test_parquet=dataset_test,
            sample=sample,
            output_path=out,
            artifact_dir=artifact_dir,
        )
        errors = contract(payload)
        if not errors:
            state.dep.submission_path = payload["submission_path"]
            state.dep.deployment_plan = (
                f"Deterministic baseline submission "
                f"({chosen.get('technique', 'chosen model')}); validated against sample."
            )
            return StateDelta(["dep.submission_path", "dep.deployment_plan"])
        record_degraded(state, "6.1", "developer", f"deterministic submission invalid: {errors[0]}")
    except Exception as exc:  # noqa: BLE001
        record_degraded(state, "6.1", "developer", f"deterministic submission failed: {exc}")

    header_vars = {
        "TRAIN_PARQUET": dataset_train,
        "TEST_PARQUET": dataset_test,
        "TARGET": state.resolved_target(),
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

    res = run_authored_code(
        pyexec=pyexec,
        agent_name="developer",
        state=state,
        instruction=_SUBMISSION_INSTRUCTION,
        header_vars=header_vars,
        header_helpers=_DEVELOPER_HEADER_HELPERS,
        contract=contract,
        contract_hint=(
            "Required keys: submission_path (str, absolute path to written CSV), "
            "rows (int, must match file and sample submission row count)."
        ),
        artifact_dir=artifact_dir,
    )
    if res.degraded:
        record_degraded(
            state, "6.1", "developer",
            res.error or "authored submission fell back to baseline",
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
    artifact = state.md.chosen_model.artifact_path if state.md.chosen_model else "n/a"
    experience = (
        f"# Experience — {state.case_id}\n\n"
        f"- Loops fired: {loops or 'none'}\n"
        f"- Degraded steps: {deg or 'none'}\n"
        f"- Chosen model: "
        f"{state.md.chosen_model.technique if state.md.chosen_model else 'n/a'}\n"
        f"- CV: {state.md.chosen_model.cv_score if state.md.chosen_model else 'n/a'}\n"
        f"- Artifact: {artifact}\n"
    )
    state.dep.experience_documentation = experience
    append_experience_to_knowledge(state.case_id, experience)
    return StateDelta(["dep.experience_documentation"])
