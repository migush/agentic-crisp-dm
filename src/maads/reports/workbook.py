"""Deterministic Jupyter workbook for human data-scientist handoff."""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from maads.artifact_paths import RunPaths
from maads.conclusions import build_conclusions_summary
from maads.outcome import ml_outcome_deficits, ml_run_succeeded, workflow_complete
from maads.paths import repo_root
from maads.reports.execution_analysis import build_execution_analysis
from maads.state import SUBSTEP_NAMES, CrispDMState
from maads.success_criterion import criterion_direction

_INJECTION_HEADER = re.compile(
    r"^# --- injected by maads; do not redefine ---\n"
    r"(?:import json\n|from pathlib import Path\n|import pandas as pd\n)*"
    r"(?:[A-Z][A-Z0-9_]* = .*\n)+",
    re.MULTILINE,
)

_BROKEN_REPO_PATH = re.compile(r"'str\(REPO_ROOT\)/([^']*)'")
_BROKEN_RUN_PATH = re.compile(r"'str\(RUN_DIR\)/([^']*)'")


def _rewrite_quoted_paths(out: str, root_name: str, root_s: str) -> str:
    # Match the root path inside a string literal regardless of how its
    # separators are written in the source: POSIX "/", Windows native "\",
    # or repr-escaped "\\". The relative tail is normalised to "/" so the
    # generated ``ROOT / '...'`` literal is portable across platforms.
    sep = r"[\\/]+"
    parts = [re.escape(p) for p in root_s.replace("\\", "/").split("/") if p]
    # Allow an optional leading separator so POSIX absolute roots ("/home/...")
    # match as well as Windows drive roots ("C:\\...").
    pattern = re.compile(rf"""(['"])(?:{sep})?{sep.join(parts)}(?:{sep}([^'"]*))?\1""")

    def repl(match: re.Match[str]) -> str:
        rel = re.sub(sep, "/", match.group(2) or "")
        if rel:
            return f"str({root_name} / '{rel}')"
        return f"str({root_name})"

    return pattern.sub(repl, out)

# Substeps that may emit sandbox Python, in CRISP-DM order.
_CODE_SUBSTEPS: tuple[str, ...] = (
    "2.1", "2.2", "2.3", "2.4",
    "3.1", "3.2", "3.3", "3.4", "3.5",
    "4.3", "4.4",
    "6.1",
)


def _cell_md(text: str) -> dict[str, Any]:
    lines = text if text.endswith("\n") else text + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [lines]}


def _cell_code(text: str) -> dict[str, Any]:
    lines = text if text.endswith("\n") else text + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [lines],
    }


def _load_manifest_rows(paths: RunPaths) -> list[dict[str, Any]]:
    manifest = paths.sandbox_manifest()
    if not manifest.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _winning_scripts(paths: RunPaths) -> dict[str, dict[str, Any]]:
    """Last successful sandbox script per substep."""
    by_substep: dict[str, list[dict[str, Any]]] = {}
    for row in _load_manifest_rows(paths):
        sub = row.get("substep")
        if not sub:
            continue
        by_substep.setdefault(str(sub), []).append(row)
    winners: dict[str, dict[str, Any]] = {}
    for sub, rows in by_substep.items():
        winner = next((r for r in reversed(rows) if r.get("ok")), None)
        if winner:
            winners[sub] = winner
    return winners


def _failed_attempts(paths: RunPaths) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in _load_manifest_rows(paths):
        if row.get("ok"):
            continue
        failures.append({
            "substep": row.get("substep"),
            "label": row.get("label"),
            "script": row.get("script"),
            "return_code": row.get("return_code"),
        })
    return failures


def _conclusions_by_substep(conclusions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for phase in conclusions.get("phases") or []:
        for item in phase.get("items") or []:
            sub = item.get("id")
            if sub:
                out[str(sub)] = item
    return out


def _strip_injection_header(source: str) -> str:
    return _INJECTION_HEADER.sub("", source, count=1).lstrip("\n")


def _rewrite_paths(source: str, run_dir: Path, repo: Path) -> str:
    """Replace absolute run/repo paths with portable expressions."""
    out = source
    run_s = str(run_dir.resolve())
    repo_s = str(repo.resolve())

    # Repair paths broken by an earlier naive replace inside string literals.
    out = _BROKEN_REPO_PATH.sub(r"str(REPO_ROOT / '\1')", out)
    out = _BROKEN_RUN_PATH.sub(r"str(RUN_DIR / '\1')", out)

    out = _rewrite_quoted_paths(out, "REPO_ROOT", repo_s)
    out = _rewrite_quoted_paths(out, "RUN_DIR", run_s)

    out = _BROKEN_REPO_PATH.sub(r"str(REPO_ROOT / '\1')", out)
    out = _BROKEN_RUN_PATH.sub(r"str(RUN_DIR / '\1')", out)
    return out


def _parse_injection_vars(raw: str) -> dict[str, str]:
    """Extract ALL_CAPS assignments from the maads injection header."""
    if not raw.startswith("# --- injected by maads"):
        return {}
    vars_out: dict[str, str] = {}
    for line in raw.splitlines()[1:]:
        if not line.strip():
            continue
        if line.startswith("import ") or line.startswith("from "):
            continue
        match = re.match(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+)$", line)
        if match:
            vars_out[match.group(1)] = match.group(2)
            continue
        break
    return vars_out


def _defined_names(source: str) -> set[str]:
    return set(re.findall(r"^([A-Z][A-Z0-9_]*)\s*=", source, re.MULTILINE))


def _used_names(source: str) -> set[str]:
    return set(re.findall(r"\b([A-Z][A-Z0-9_]{1,})\b", source))


def _format_injection_binding(name: str, value: str, *, run_dir: Path, repo: Path) -> str:
    literal = value.strip()
    if name in {"CHOSEN_MODEL", "FEATURE_HINTS", "CLASS_LABELS", "DATASET_INSPECT_JSON"}:
        try:
            parsed = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            parsed = None
        if name == "CHOSEN_MODEL" and parsed is not None:
            data = json.loads(parsed) if isinstance(parsed, str) else parsed
            bundle = data.get("evaluation_bundle") or {}
            figures = bundle.get("figures") or []
            run_s = str(run_dir.resolve())
            bundle["figures"] = [
                Path(fig).relative_to(run_dir).as_posix()
                if isinstance(fig, str) and fig.startswith(run_s)
                else fig
                for fig in figures
            ]
            data["evaluation_bundle"] = bundle
            return f"{name} = {json.dumps(data)!r}"
        if isinstance(parsed, (dict, list)):
            return f"{name} = {json.dumps(parsed)!r}"
    expr = _rewrite_paths(f"{name} = {literal}\n", run_dir, repo).strip()
    return expr


def _prepend_missing_bindings(
    source: str,
    injection_vars: dict[str, str],
    *,
    run_dir: Path,
    repo: Path,
) -> str:
    """Re-introduce sandbox path constants removed with the injection header."""
    aliases = {
        "TRAIN_CSV": "DATA_TRAIN_CSV",
        "TEST_CSV": "DATA_TEST_CSV",
        "SOURCE_TRAIN": "DATA_TRAIN_CSV",
        "SAMPLE_SUBMISSION": "SAMPLE_SUBMISSION",
        "TRAIN_PARQUET": "TRAIN_PARQUET",
        "TEST_PARQUET": "TEST_PARQUET",
        "FIGURES_DIR": "FIGURES_DIR",
        "OUTPUT_PATH": "RUN_DIR / 'submission.csv'",
        "TARGET": "TARGET",
        "ID_COL": "ID_COL",
        "METRIC": "EVAL_METRIC",
        "PROBLEM_TYPE": "PROBLEM_TYPE",
        "EVAL_METRIC": "EVAL_METRIC",
        "FEATURE_HINTS": "FEATURE_HINTS",
    }
    defined = _defined_names(source)
    bindable = set(aliases) | set(injection_vars)
    needed = (_used_names(source) - defined) & bindable
    preamble: list[str] = []
    for name in sorted(needed):
        if name in aliases:
            preamble.append(f"{name} = {aliases[name]}")
        elif name in injection_vars:
            preamble.append(_format_injection_binding(
                name, injection_vars[name], run_dir=run_dir, repo=repo,
            ))
    if not preamble:
        return source
    return "\n".join(preamble) + "\n\n" + source.lstrip("\n")


def _read_winning_code(paths: RunPaths, row: dict[str, Any]) -> str | None:
    script = row.get("script")
    if not script:
        return None
    script_path = paths.sandbox_exec() / str(script)
    if not script_path.is_file():
        return None
    raw = script_path.read_text(encoding="utf-8")
    injection_vars = _parse_injection_vars(raw)
    cleaned = _strip_injection_header(raw)
    cleaned = _rewrite_paths(cleaned, paths.run_dir, repo_root())
    return _prepend_missing_bindings(
        cleaned, injection_vars, run_dir=paths.run_dir, repo=repo_root(),
    )


def _repo_relative_posix(path: str | Path | None) -> str:
    """Repo-relative path as a plain POSIX string for notebook code cells."""
    if not path:
        return ""
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(p.resolve())


def _setup_cell_source(state: CrispDMState) -> str:
    cfg = state.config
    sc = cfg.success_criterion
    dir_ = criterion_direction(sc.metric, sc.direction)
    feature_hints = json.dumps(cfg.feature_hints, indent=2)
    class_labels = json.dumps(cfg.class_labels, indent=2)
    train_csv = _repo_relative_posix(cfg.data.train_csv)
    test_csv = _repo_relative_posix(cfg.data.test_csv)
    sample_csv = _repo_relative_posix(cfg.data.sample_submission_csv)
    return f'''"""Portable paths and case constants for this MAADS run."""
from pathlib import Path
import json

# This notebook lives in <run_dir>/reports/ — resolve run and repo roots.
_cwd = Path.cwd()
RUN_DIR = _cwd.parent if _cwd.name == "reports" else _cwd
for _p in [RUN_DIR, *_cwd.parents]:
    if (_p / "pyproject.toml").is_file():
        REPO_ROOT = _p
        break
else:
    raise RuntimeError("Could not locate repository root (pyproject.toml)")

CASE_ID = {cfg.case_id!r}
PROBLEM_TYPE = {cfg.problem_type!r}
TARGET = {cfg.target_column!r}
ID_COL = {cfg.id_column!r}
EVAL_METRIC = {cfg.evaluation_metric!r}
SUCCESS_METRIC = {sc.metric!r}
SUCCESS_THRESHOLD = {sc.threshold!r}
SUCCESS_DIRECTION = {dir_!r}
FEATURE_HINTS = json.loads({feature_hints!r})
CLASS_LABELS = json.loads({class_labels!r})

DATA_TRAIN_CSV = REPO_ROOT / {train_csv!r}
DATA_TEST_CSV = REPO_ROOT / {test_csv!r}
SAMPLE_SUBMISSION = REPO_ROOT / {sample_csv!r}

TRAIN_PARQUET = RUN_DIR / "train.parquet"
TEST_PARQUET = RUN_DIR / "test.parquet"
PREP_DIR = RUN_DIR / "prep"
FIGURES_DIR = RUN_DIR / "figures"
NOTEBOOK_OUT = RUN_DIR / "notebook_outputs"
NOTEBOOK_OUT.mkdir(parents=True, exist_ok=True)

print(f"Case: {{CASE_ID}} ({{PROBLEM_TYPE}})")
print(f"Run directory: {{RUN_DIR}}")
'''


def _title_markdown(
    state: CrispDMState,
    paths: RunPaths,
    analysis: dict[str, Any],
) -> str:
    cfg = state.config
    sc = analysis.get("success_criterion") or {}
    cm = analysis.get("chosen_model") or {}
    met = sc.get("met")
    ml_ok = analysis.get("ml_success")
    lines = [
        f"# Case Workbook — {cfg.case_id}",
        "",
        f"*Generated {datetime.now(timezone.utc).isoformat()}*",
        "",
        "Handoff notebook for human data scientists to review agent work and continue the case.",
        "",
        "## Run summary",
        "",
        f"- **Run ID:** `{paths.run_dir.name}`",
        f"- **Problem type:** {cfg.problem_type}",
        f"- **Target:** `{cfg.target_column}`",
        f"- **Evaluation metric:** `{cfg.evaluation_metric}`",
        f"- **Workflow complete:** {workflow_complete(state)}",
        f"- **ML success:** {ml_ok}",
    ]
    deficits = ml_outcome_deficits(state)
    if deficits:
        lines.append(f"- **ML deficits:** {'; '.join(deficits)}")
    if cm:
        cv = cm.get("cv_score")
        if cv is not None:
            lines.append(f"- **Chosen model:** {cm.get('technique')} (CV {cv:.4f})")
    if sc:
        lines.append(
            f"- **Success criterion:** {sc.get('metric')} {sc.get('direction')} "
            f"{sc.get('threshold')} — {'met' if met else 'not met'}"
        )
    if state.ev.decision:
        lines.append(f"- **Decision:** {state.ev.decision}")
    lines += [
        "",
        "## Problem statement",
        "",
        cfg.problem_statement.strip(),
        "",
    ]
    if state.bu.data_mining_goals:
        lines += ["## Data mining goals", "", state.bu.data_mining_goals.strip(), ""]
    return "\n".join(lines)


def _phase_intro_markdown(phase: dict[str, Any]) -> str:
    name = phase.get("name") or f"Phase {phase.get('id', '?')}"
    return f"## {name}"


def _substep_markdown(item: dict[str, Any]) -> str:
    sub = item.get("id", "?")
    title = item.get("name") or SUBSTEP_NAMES.get(str(sub), str(sub))
    summary = item.get("summary", "")
    return f"### {sub} — {title}\n\n{summary}"


def _findings_markdown(state: CrispDMState, conclusions: dict[str, Any]) -> str:
    from maads.reports.report_format import format_assessment_lines

    lines = ["## Findings and assessment", ""]
    assessment = conclusions.get("assessment") or state.ev.assessment_of_dm_results or {}
    if assessment:
        lines.extend(format_assessment_lines(assessment))
        lines.append("")
    cm = conclusions.get("chosen_model")
    if cm and cm.get("evaluation_metrics"):
        lines.append("### Evaluation metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for key, val in sorted(cm["evaluation_metrics"].items()):
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")
        lines.append("")
    blockers = conclusions.get("data_quality_blockers") or []
    tolerable = conclusions.get("data_quality_tolerable") or []
    if blockers or tolerable:
        lines.append("### Data quality notes")
        lines.append("")
        if blockers:
            lines.append("**Blockers:**")
            for b in blockers:
                lines.append(f"- {b}")
            lines.append("")
        if tolerable:
            lines.append("**Tolerable:**")
            for t in tolerable[:10]:
                lines.append(f"- {t}")
            if len(tolerable) > 10:
                lines.append(f"- _…and {len(tolerable) - 10} more_")
            lines.append("")
    return "\n".join(lines)


def _failures_markdown(failures: list[dict[str, Any]], paths: RunPaths) -> str:
    if not failures:
        return ""
    lines = [
        "## Appendix — recovered sandbox failures",
        "",
        "These attempts failed during the automated run; later attempts succeeded.",
        "",
    ]
    for f in failures:
        sub = f.get("substep", "?")
        name = SUBSTEP_NAMES.get(str(sub), str(sub))
        lines.append(
            f"- **{sub} ({name})** — {f.get('label')} "
            f"(`{f.get('script')}`, rc={f.get('return_code')})"
        )
        script = f.get("script")
        if script:
            stderr_path = paths.sandbox_exec() / str(script).replace(".py", ".stderr.txt")
            if not stderr_path.is_file():
                stderr_path = paths.sandbox_exec() / f"{Path(str(script)).stem}.stderr.txt"
            # manifest stores separate stderr filename
            for row in _load_manifest_rows(paths):
                if row.get("script") == script and not row.get("ok"):
                    stderr_name = row.get("stderr")
                    if stderr_name:
                        stderr_path = paths.sandbox_exec() / str(stderr_name)
                    break
            if stderr_path.is_file():
                tail = stderr_path.read_text(encoding="utf-8").strip()[-500:]
                if tail:
                    lines.append(f"  ```\n  {tail}\n  ```")
    lines.append("")
    return "\n".join(lines)


def _deliverables_markdown(
    analysis: dict[str, Any],
    *,
    md_dir: Path | None = None,
    run_dir: Path | None = None,
) -> str:
    from maads.reports.md_paths import md_file_link

    deliverables = analysis.get("deliverables") or {}
    lines = ["## Appendix — artifact index", ""]
    for key, path in sorted(deliverables.items()):
        if key.endswith("_exists"):
            continue
        if path:
            if md_dir is not None and run_dir is not None:
                link = md_file_link(path, md_dir=md_dir, run_dir=run_dir)
                lines.append(f"- **{key}:** {link or path}")
            else:
                lines.append(f"- **{key}:** `{path}`")
    lines.append("")
    return "\n".join(lines)


def _continuation_markdown(state: CrispDMState) -> str:
    cfg = state.config
    problem = cfg.problem_type
    lines = [
        "## Continue from here",
        "",
        "Editable checklist for human data scientists:",
        "",
    ]
    common = [
        "- Re-run cells top-to-bottom after changing data or code.",
        "- Use the canonical pipeline cell above for CV-aligned fit and submission.",
        "- Load `notebook_outputs/model.joblib` for further experiments.",
        "- Document experiments in a new section below.",
    ]
    if problem == "regression":
        specific = [
            "- Inspect residual patterns on high-leverage / outlier segments.",
            "- Try target transforms, robust loss, or ensemble methods.",
            "- Add feature importance or SHAP for interpretability.",
        ]
    elif problem == "binary_classification":
        specific = [
            "- Review class balance and threshold trade-offs (precision/recall).",
            "- Inspect confusion-matrix errors by segment.",
            "- Try calibration and alternative classifiers or text representations.",
        ]
    else:
        specific = [
            f"- Extend modeling for problem type `{problem}`.",
            "- Add diagnostics appropriate to the evaluation metric.",
        ]
    if cfg.feature_hints.get("text_free"):
        specific.append(
            "- Text features detected — experiment with representations "
            f"from `FEATURE_HINTS` ({cfg.feature_hints.get('text_free')})."
        )
    if cfg.feature_hints.get("ordinal_string_encoded") or cfg.feature_hints.get("ordinal"):
        cols = (
            cfg.feature_hints.get("ordinal_string_encoded")
            or cfg.feature_hints.get("ordinal")
            or []
        )
        specific.append(f"- Consider ordinal encoding for: {cols}.")
    lines.extend(common)
    lines.extend(specific)
    lines += [
        "",
        "### Your notes",
        "",
        "_Add observations, hypotheses, and next experiments here._",
        "",
    ]
    return "\n".join(lines)


def _train_deploy_note(winners: dict[str, dict[str, Any]]) -> str:
    if "4.3" in winners and "6.1" in winners:
        return (
            "### Train vs deploy consistency\n\n"
            "Modeling (4.3/4.4) and submission (6.1) are separate generated scripts. "
            "Before shipping, verify that imputation, encoding, and feature columns "
            "match what was cross-validated. The **canonical pipeline** cell below "
            "uses one sklearn `Pipeline` for fit, persist, and predict.\n"
        )
    return ""


def _chosen_technique(state: CrispDMState) -> str:
    if state.md.chosen_model and state.md.chosen_model.technique:
        return state.md.chosen_model.technique
    return state.md.modeling_technique or "unspecified"


_TECHNIQUE_ALIASES: dict[str, str] = {
    "lgbm": "lightgbm",
    "xgb": "xgboost",
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
    "gb": "gradient_boosting",
    "gbc": "gradient_boosting",
    "gbr": "gradient_boosting",
    "rf": "random_forest",
    "svc": "svm",
    "hist_gb": "hist_gradient_boosting",
}


def _normalize_technique(name: str) -> str:
    normalized = name.lower().strip().replace("-", "_").replace(" ", "_")
    return _TECHNIQUE_ALIASES.get(normalized, normalized)


def _canonical_pipeline_markdown(state: CrispDMState, *, from_agent_script: bool) -> str:
    technique = _chosen_technique(state)
    if from_agent_script:
        return (
            "## Canonical pipeline (human continuation)\n\n"
            f"The cell below is the agent's **successful 4.3 Build Model** sandbox "
            f"script for **{technique}** — not a synthetic template. Review and "
            "adapt it for deployment and submission consistency.\n"
        )
    return (
        "## Canonical pipeline (human continuation)\n\n"
        f"No 4.3 sandbox script was captured for this run. The simplified sklearn "
        f"template below uses **{technique}** as a starting point only — it may "
        "not match what the agent chose or ran. Prefer any 4.3 script in the "
        "Modeling section above when present.\n"
    )


def _template_estimator_block(problem: str) -> tuple[str, str]:
    optional_boosters = '''\
try:
    import lightgbm as lgb
    _ESTIMATORS["lightgbm"] = (
        lgb.LGBMClassifier(random_state=42, verbosity=-1)
        if PROBLEM_TYPE != "regression"
        else lgb.LGBMRegressor(random_state=42, verbosity=-1)
    )
except ImportError:
    pass
try:
    import xgboost as xgb
    _ESTIMATORS["xgboost"] = (
        xgb.XGBClassifier(random_state=42, eval_metric="logloss")
        if PROBLEM_TYPE != "regression"
        else xgb.XGBRegressor(random_state=42)
    )
except ImportError:
    pass
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _ESTIMATORS["catboost"] = (
        CatBoostClassifier(random_state=42, verbose=0)
        if PROBLEM_TYPE != "regression"
        else CatBoostRegressor(random_state=42, verbose=0)
    )
except ImportError:
    pass
'''
    if problem == "regression":
        estimator_block = '''\
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge

_ESTIMATORS = {
    "ridge": Ridge(random_state=42),
    "lasso": Lasso(random_state=42, max_iter=5000),
    "elastic_net": ElasticNet(random_state=42, max_iter=5000),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
}
''' + optional_boosters
        predict_fn = '''\
y_pred = pipeline.predict(X_test)
if use_log_target:
    y_pred = np.expm1(y_pred)
    y_pred = np.maximum(y_pred, 0)
'''
    elif problem == "binary_classification":
        estimator_block = '''\
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

_ESTIMATORS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
    "svm": SVC(probability=True, random_state=42),
}
''' + optional_boosters
        predict_fn = '''\
y_pred = pipeline.predict(X_test)
'''
    else:
        estimator_block = '''\
from sklearn.ensemble import RandomForestClassifier

_ESTIMATORS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}
'''
        predict_fn = '''\
y_pred = pipeline.predict(X_test)
'''
    return estimator_block, predict_fn


def _canonical_pipeline_template_code(state: CrispDMState) -> str:
    raw_technique = (
        state.md.chosen_model.technique
        if state.md.chosen_model
        else state.md.modeling_technique or "gradient_boosting"
    )
    technique = _normalize_technique(raw_technique)
    problem = state.config.problem_type
    estimator_block, predict_fn = _template_estimator_block(problem)
    default = "ridge" if problem == "regression" else "logistic_regression"
    note = (
        "# NOTE: no 4.3 sandbox script captured; template may not match agent choice\n"
        if raw_technique != "unspecified"
        else "# NOTE: no 4.3 sandbox script captured; using simplified template\n"
    )
    return f'''\
"""Fit one sklearn Pipeline, persist model.joblib, and write a submission."""
{note}import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

{estimator_block}
TECHNIQUE = {technique!r}
_DEFAULT_TECHNIQUE = {default!r}
MODEL_PATH = NOTEBOOK_OUT / "model.joblib"
SUBMISSION_OUT = NOTEBOOK_OUT / "submission.csv"

def build_pipeline(technique: str):
    if technique not in _ESTIMATORS:
        print(
            f"WARNING: {{technique!r}} not in template registry; "
            f"using {{_DEFAULT_TECHNIQUE!r}}"
        )
        technique = _DEFAULT_TECHNIQUE
    estimator = _ESTIMATORS[technique]

    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    drop_cols = {{TARGET}}
    if ID_COL in train_df.columns:
        drop_cols.add(ID_COL)
    text_cols = set(FEATURE_HINTS.get("text_free") or [])
    drop_cols |= text_cols

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[TARGET]
    use_log_target = PROBLEM_TYPE == "regression" and "log" in EVAL_METRIC.lower()
    if use_log_target:
        y_train = np.log1p(y_train)

    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    if ID_COL in test_df.columns:
        test_ids = test_df[ID_COL].values
    else:
        test_ids = np.arange(len(X_test))
    return pipeline, X_test, use_log_target, test_ids

pipeline, X_test, use_log_target, test_ids = build_pipeline(TECHNIQUE)
joblib.dump({{"pipeline": pipeline, "technique": TECHNIQUE, "use_log_target": use_log_target}}, MODEL_PATH)
print(f"Saved {{MODEL_PATH}}")

{predict_fn}
sample = pd.read_csv(SAMPLE_SUBMISSION)
id_col = ID_COL if ID_COL in sample.columns else sample.columns[0]
target_col = [c for c in sample.columns if c != id_col][0]
out = pd.DataFrame({{id_col: test_ids, target_col: y_pred}})
out.to_csv(SUBMISSION_OUT, index=False)
print(f"Wrote {{SUBMISSION_OUT}} ({{len(out)}} rows)")
'''


def _canonical_pipeline_code(
    state: CrispDMState,
    scripts: dict[str, str] | None = None,
) -> str:
    if scripts and scripts.get("4.3"):
        return scripts["4.3"]
    return _canonical_pipeline_template_code(state)


def build_workbook_context(
    state: CrispDMState,
    paths: RunPaths,
    *,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured inputs for workbook rendering."""
    conclusions = build_conclusions_summary(state)
    analysis = analysis or build_execution_analysis(state, paths)
    winners = _winning_scripts(paths)
    scripts: dict[str, str] = {}
    for sub, row in winners.items():
        code = _read_winning_code(paths, row)
        if code:
            scripts[sub] = code
    return {
        "case_id": state.case_id,
        "run_id": paths.run_dir.name,
        "problem_type": state.config.problem_type,
        "conclusions": conclusions,
        "analysis": analysis,
        "winners": {k: {kk: vv for kk, vv in v.items() if kk != "stdout"} for k, v in winners.items()},
        "scripts": scripts,
        "failures": _failed_attempts(paths),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def render_workbook_ipynb(context: dict[str, Any], state: CrispDMState, paths: RunPaths) -> dict[str, Any]:
    """Build nbformat v4 notebook dict."""
    analysis = context["analysis"]
    conclusions = context["conclusions"]
    winners = context.get("winners") or {}
    scripts = context.get("scripts") or {}
    by_substep = _conclusions_by_substep(conclusions)

    cells: list[dict[str, Any]] = [
        _cell_md(_title_markdown(state, paths, analysis)),
        _cell_code(_setup_cell_source(state)),
    ]

    seen_phases: set[int] = set()
    for phase in conclusions.get("phases") or []:
        phase_id = phase.get("id")
        if phase_id not in seen_phases:
            cells.append(_cell_md(_phase_intro_markdown(phase)))
            seen_phases.add(phase_id)
        for item in phase.get("items") or []:
            sub = str(item.get("id", ""))
            cells.append(_cell_md(_substep_markdown(item)))
            if sub in scripts:
                cells.append(_cell_code(scripts[sub]))

    # Executable substeps with code but no conclusion item (edge case).
    for sub in _CODE_SUBSTEPS:
        if sub in scripts and sub not in by_substep:
            title = SUBSTEP_NAMES.get(sub, sub)
            cells.append(_cell_md(f"### {sub} — {title}\n\n_(executable artifact; no summary stored)_"))
            cells.append(_cell_code(scripts[sub]))

    train_deploy = _train_deploy_note(winners)
    if train_deploy:
        cells.append(_cell_md(train_deploy))

    if state.md.chosen_model or "4.3" in scripts or (paths.run_dir / "train.parquet").is_file():
        from_agent = "4.3" in scripts
        cells.append(_cell_md(_canonical_pipeline_markdown(state, from_agent_script=from_agent)))
        cells.append(_cell_code(_canonical_pipeline_code(state, scripts)))

    cells.append(_cell_md(_findings_markdown(state, conclusions)))

    failures_md = _failures_markdown(context.get("failures") or [], paths)
    if failures_md:
        cells.append(_cell_md(failures_md))

    cells.append(_cell_md(_deliverables_markdown(analysis, md_dir=paths.reports, run_dir=paths.run_dir)))
    cells.append(_cell_md(_continuation_markdown(state)))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
            "maads": {
                "case_id": context.get("case_id"),
                "run_id": context.get("run_id"),
                "problem_type": context.get("problem_type"),
                "generated_at": context.get("generated_at"),
            },
        },
        "cells": cells,
    }


def write_case_workbook(
    state: CrispDMState,
    paths: RunPaths,
    *,
    analysis: dict[str, Any] | None = None,
) -> Path:
    """Write ``reports/case_workbook.ipynb`` and ``reports/workbook_context.json``."""
    paths.reports.mkdir(parents=True, exist_ok=True)
    context = build_workbook_context(state, paths, analysis=analysis)
    notebook = render_workbook_ipynb(context, state, paths)
    nb_path = paths.reports / "case_workbook.ipynb"
    ctx_path = paths.reports / "workbook_context.json"
    nb_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    ctx_path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    return nb_path
