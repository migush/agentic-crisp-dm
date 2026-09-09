"""Deterministic ML/data tools for DE/DS orchestration.

Agents select and interpret; these functions execute standard CRISP-DM ops
without freeform ``run_authored_code``. Case variance comes only from
``feature_hints`` / problem_type / metric — never ``case_id``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from maads.success_criterion import criterion_direction

# ── IO helpers ──────────────────────────────────────────────────────────────


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"table not found: {p}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _write_parquet(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return str(path.resolve())


def _is_text_dtype(series: pd.Series) -> bool:
    return str(series.dtype) == "object" or str(series.dtype).startswith("string")


def _feature_hints(hints: dict[str, Any] | None) -> dict[str, Any]:
    return hints if isinstance(hints, dict) else {}


# ── Profiling / EDA ─────────────────────────────────────────────────────────


def profile_dataset(
    train_path: str | Path,
    test_path: str | Path | None = None,
    *,
    target: str | None = None,
    id_column: str | None = None,
    na_means_absent: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Full measured profile used by DU substeps 2.1–2.4 / 2.3."""
    train = _read_table(train_path)
    if max_rows is not None and len(train) > max_rows:
        train = train.head(max_rows)

    n_rows, n_cols = int(len(train)), int(train.shape[1])
    columns = list(train.columns)
    dtypes = {c: str(train[c].dtype) for c in columns}
    missing = {c: int(train[c].isna().sum()) for c in columns}
    cardinality = {
        c: int(train[c].nunique(dropna=True)) for c in columns
    }
    constant_cols = [c for c, k in cardinality.items() if k <= 1]
    duplicate_id_count = 0
    if id_column and id_column in train.columns:
        duplicate_id_count = int(train[id_column].duplicated().sum())

    target_info: dict[str, Any] = {}
    if target and target in train.columns:
        vc = train[target].value_counts(dropna=False)
        target_info = {
            "name": target,
            "missing": int(train[target].isna().sum()),
            "n_unique": int(train[target].nunique(dropna=True)),
            "distribution": {str(k): int(v) for k, v in vc.head(20).items()},
        }
        if pd.api.types.is_numeric_dtype(train[target]):
            target_info["mean"] = float(train[target].mean())
            target_info["std"] = float(train[target].std(ddof=0) or 0.0)

    correlations: dict[str, float] = {}
    if target and target in train.columns and pd.api.types.is_numeric_dtype(train[target]):
        num = train.select_dtypes(include="number")
        if target in num.columns and len(num.columns) > 1:
            corr = num.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
            for col, val in corr.dropna().abs().sort_values(ascending=False).head(10).items():
                correlations[str(col)] = float(corr[col])

    out: dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing,
        "cardinality": cardinality,
        "constant_columns": constant_cols,
        "duplicate_id_count": duplicate_id_count,
        "target": target_info or (target or None),
        "correlations_with_target": correlations,
        "na_means_absent": list(na_means_absent or []),
        "source": "deterministic profile_dataset",
    }

    if test_path and Path(test_path).is_file():
        test = _read_table(test_path)
        if max_rows is not None and len(test) > max_rows:
            test = test.head(max_rows)
        tr_set, te_set = set(columns), set(test.columns)
        out["test_rows"] = int(len(test))
        out["test_columns"] = list(test.columns)
        out["columns_only_in_train"] = sorted(tr_set - te_set)
        out["columns_only_in_test"] = sorted(te_set - tr_set)
        out["columns_shared"] = sorted(tr_set & te_set)
    return out


def collect_report(
    train_path: str,
    test_path: str = "",
    *,
    source_paths: list[Any] | None = None,
) -> dict[str, Any]:
    train = _read_table(train_path)
    report: dict[str, Any] = {
        "train_rows": int(len(train)),
        "columns": list(train.columns),
        "source": "deterministic collect_report",
    }
    if test_path and Path(test_path).is_file():
        test = _read_table(test_path)
        report["test_rows"] = int(len(test))
        report["test_columns"] = list(test.columns)
    if source_paths:
        report["source_paths"] = source_paths
    return report


def describe_report_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_rows": profile["n_rows"],
        "n_cols": profile["n_cols"],
        "columns": list(profile["columns"]),
        "dtypes": dict(profile["dtypes"]),
        "missing": dict(profile["missing"]),
        "cardinality": dict(profile.get("cardinality") or {}),
        "source": "deterministic describe_report",
    }


def explore_report_from_profile(profile: dict[str, Any], target: str) -> dict[str, Any]:
    tgt = profile.get("target")
    tgt_name = tgt.get("name") if isinstance(tgt, dict) else target
    return {
        "n_rows": profile["n_rows"],
        "target": tgt_name or target,
        "target_distribution": (tgt or {}).get("distribution") if isinstance(tgt, dict) else {},
        "correlations": dict(profile.get("correlations_with_target") or {}),
        "constant_columns": list(profile.get("constant_columns") or []),
        "missing": dict(profile.get("missing") or {}),
        "source": "deterministic explore_report",
    }


def quality_report_from_profile(
    profile: dict[str, Any],
    *,
    target: str,
    na_means_absent: Sequence[str] | None = None,
) -> dict[str, Any]:
    absent = set(na_means_absent or profile.get("na_means_absent") or [])
    blockers: list[str] = []
    tolerable: list[str] = []
    n_rows = max(int(profile.get("n_rows") or 1), 1)
    missing = profile.get("missing") or {}

    tgt = profile.get("target")
    if target and (not isinstance(tgt, dict) or tgt.get("name") != target):
        if target not in (profile.get("columns") or []):
            blockers.append(f"missing target column '{target}'")
    elif isinstance(tgt, dict) and int(tgt.get("missing") or 0) > 0:
        blockers.append(f"target '{target}' has missing values")

    for col in profile.get("constant_columns") or []:
        if col == target:
            continue
        blockers.append(f"constant column '{col}'")

    dup = int(profile.get("duplicate_id_count") or 0)
    if dup > 0:
        blockers.append(f"duplicate id rows: {dup}")

    for col, count in missing.items():
        if col == target or not isinstance(count, int) or count <= 0:
            continue
        rate = count / n_rows
        if col in absent:
            tolerable.append(f"{col}: structural absence (no feature) — {rate:.0%} NA")
        elif rate >= 0.6:
            blockers.append(f"undocumented high missingness on '{col}' ({rate:.0%})")
        elif rate >= 0.05:
            tolerable.append(f"{col}: {rate:.0%} missing")

    return {
        "blockers": blockers,
        "tolerable": tolerable,
        "source": "deterministic quality_report",
    }


# ── Config-driven prep ──────────────────────────────────────────────────────


def _fill_na_means_absent(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and _is_text_dtype(out[col]):
            out[col] = out[col].fillna("__ABSENT__").astype(str)
        elif col in out.columns:
            out[col] = out[col].fillna(-1)
    return out


def _impute_frame(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    *,
    target: str,
    feature_hints: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any], dict[str, Any]]:
    absent = list(feature_hints.get("na_means_absent") or [])
    tr = _fill_na_means_absent(train, absent)
    te = _fill_na_means_absent(test, absent) if test is not None else None

    missing_before = {c: int(train[c].isna().sum()) for c in train.columns}
    medians: dict[str, float] = {}
    modes: dict[str, Any] = {}

    for col in tr.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(tr[col]):
            med = float(tr[col].median()) if tr[col].notna().any() else 0.0
            medians[col] = med
            tr[col] = tr[col].fillna(med)
            if te is not None and col in te.columns:
                te[col] = te[col].fillna(med)
        elif _is_text_dtype(tr[col]) or str(tr[col].dtype) == "category":
            mode = tr[col].mode(dropna=True)
            fill = str(mode.iloc[0]) if len(mode) else "Unknown"
            modes[col] = fill
            tr[col] = tr[col].fillna(fill).astype(str)
            if te is not None and col in te.columns:
                te[col] = te[col].fillna(fill).astype(str)

    missing_after = {c: int(tr[c].isna().sum()) for c in tr.columns}
    return tr, te, missing_before, missing_after


def clean_tables(
    train_in: str,
    test_in: str,
    outdir: str | Path,
    *,
    target: str,
    feature_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = _feature_hints(feature_hints)
    train = _read_table(train_in)
    test = _read_table(test_in) if test_in and Path(test_in).is_file() else None
    tr, te, before, after = _impute_frame(train, test, target=target, feature_hints=hints)
    wd = Path(outdir)
    train_out = _write_parquet(tr, wd / "train_clean.parquet")
    test_out = (
        _write_parquet(te, wd / "test_clean.parquet")
        if te is not None
        else ""
    )
    return {
        "train_out": train_out,
        "test_out": test_out,
        "missing_before": before,
        "missing_after": after,
        "operations": ["impute_numeric_median", "impute_categorical_mode", "na_means_absent_fill"],
        "source": "deterministic clean_tables",
    }


def construct_tables(
    train_in: str,
    test_in: str,
    outdir: str | Path,
    *,
    target: str,
    feature_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = _feature_hints(feature_hints)
    train = _read_table(train_in)
    test = _read_table(test_in) if test_in and Path(test_in).is_file() else None
    derived: list[str] = []

    high_missing = list(hints.get("high_missing") or [])
    for col in high_missing:
        if col not in train.columns:
            continue
        name = f"{col}_missing"
        train[name] = train[col].isna().astype(int)
        if test is not None:
            test[name] = test[col].isna().astype(int) if col in test.columns else 0
        derived.append(name)

    # Light numeric interactions only when both columns exist (config-agnostic).
    numeric_hint = list(hints.get("numeric_with_missing") or [])
    if len(numeric_hint) >= 2:
        a, b = numeric_hint[0], numeric_hint[1]
        if a in train.columns and b in train.columns:
            name = f"{a}_x_{b}"
            train[name] = pd.to_numeric(train[a], errors="coerce").fillna(0) * pd.to_numeric(
                train[b], errors="coerce",
            ).fillna(0)
            if test is not None and a in test.columns and b in test.columns:
                test[name] = pd.to_numeric(test[a], errors="coerce").fillna(0) * pd.to_numeric(
                    test[b], errors="coerce",
                ).fillna(0)
            derived.append(name)

    wd = Path(outdir)
    train_out = _write_parquet(train, wd / "train_constructed.parquet")
    test_out = (
        _write_parquet(test, wd / "test_constructed.parquet")
        if test is not None
        else ""
    )
    return {
        "train_out": train_out,
        "test_out": test_out,
        "derived": derived,
        "source": "deterministic construct_tables",
    }


def integrate_tables(
    train_in: str,
    test_in: str,
    outdir: str | Path,
    *,
    target: str,
) -> dict[str, Any]:
    train = _read_table(train_in)
    test = _read_table(test_in) if test_in and Path(test_in).is_file() else pd.DataFrame()

    if len(test):
        # Keep all train columns (including TARGET); align test to shared features only.
        train_out_df = train
        feature_shared = [c for c in train.columns if c in test.columns]
        test_out_df = test.reindex(columns=feature_shared)
    else:
        train_out_df = train
        test_out_df = test

    if target and target not in train_out_df.columns:
        raise RuntimeError(f"TARGET '{target}' missing during integrate")

    wd = Path(outdir)
    train_out = _write_parquet(train_out_df, wd / "train_integrated.parquet")
    test_out = (
        _write_parquet(test_out_df, wd / "test_integrated.parquet")
        if len(test_out_df)
        else ""
    )
    return {
        "train_out": train_out,
        "test_out": test_out,
        "train_rows": int(len(train_out_df)),
        "test_rows": int(len(test_out_df)),
        "columns_train": list(train_out_df.columns),
        "columns_test": list(test_out_df.columns) if len(test_out_df) else [],
        "source": "deterministic integrate_tables",
    }


def format_tables(
    train_in: str,
    test_in: str,
    outdir: str | Path,
    *,
    target: str,
    id_column: str,
    feature_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = _feature_hints(feature_hints)
    train = _read_table(train_in)
    test = _read_table(test_in) if test_in and Path(test_in).is_file() else None

    if test is None or not len(test):
        # Modelling holdout from train when no test file.
        from sklearn.model_selection import train_test_split

        strat = train[target] if target in train.columns and train[target].nunique() < 50 else None
        train, test = train_test_split(
            train, test_size=0.2, random_state=42, stratify=strat,
        )
        # Holdout still has labels; drop target from test for submission-like shape.
        if target in test.columns:
            test = test.drop(columns=[target])

    drop_from_train = {id_column} if id_column else set()
    # Keep primary free-text only for NLP-oriented configs; tabular cases drop
    # high-cardinality text (Name/Ticket/…) so OneHot does not explode.
    text_free = list(hints.get("text_free") or [])
    primary_text = text_free[0] if text_free else None
    tabular_hints = any(
        hints.get(k)
        for k in (
            "categorical",
            "numeric_with_missing",
            "ordinal",
            "ordinal_string_encoded",
        )
    )
    nlp_primary = bool(hints.get("representation_options")) or (
        bool(text_free) and not tabular_hints
    )
    for col in text_free:
        if nlp_primary and col == primary_text:
            continue
        if col not in train.columns:
            continue
        nunique = int(train[col].nunique(dropna=True))
        if nunique > max(50, int(0.3 * len(train))) or not nlp_primary:
            drop_from_train.add(col)

    dropped = sorted(c for c in drop_from_train if c in train.columns)
    train_fmt = train.drop(columns=dropped, errors="ignore")
    test_fmt = test.copy()
    # Keep id in test for submission; drop other train-only drops except id.
    for col in dropped:
        if col != id_column and col in test_fmt.columns:
            test_fmt = test_fmt.drop(columns=[col])

    if target and target not in train_fmt.columns:
        raise RuntimeError(f"TARGET '{target}' missing during format")

    # Align feature columns: test gets train features minus target, plus id if present.
    train_feats = [c for c in train_fmt.columns if c != target]
    test_cols = []
    if id_column and id_column in test.columns:
        test_cols.append(id_column)
    for c in train_feats:
        if c == id_column:
            continue
        if c in test_fmt.columns:
            test_cols.append(c)
        else:
            test_fmt[c] = np.nan
            test_cols.append(c)
    test_fmt = test_fmt.reindex(columns=list(dict.fromkeys(test_cols)))

    wd = Path(outdir)
    train_path = _write_parquet(train_fmt, wd / "train.parquet")
    test_path = _write_parquet(test_fmt, wd / "test.parquet")
    derived = [c for c in train_fmt.columns if c.endswith("_missing") or "_x_" in c]
    return {
        "train": train_path,
        "test": test_path,
        "n_train": int(len(train_fmt)),
        "n_test": int(len(test_fmt)),
        "derived": derived,
        "dropped": dropped,
        "source": "deterministic format_tables",
    }


def lint_prepared_features(
    train_parquet: str,
    *,
    target: str,
    id_column: str,
) -> list[str]:
    """Programmatic leakage / prep lint before modeling."""
    findings: list[str] = []
    df = _read_table(train_parquet)
    if target and target not in df.columns:
        findings.append(f"target '{target}' absent from prepared train")
        return findings
    if id_column and id_column in df.columns:
        findings.append(f"id column '{id_column}' still present as a train predictor risk")
    for col in df.columns:
        if col == target:
            continue
        if df[col].nunique(dropna=True) <= 1:
            findings.append(f"constant predictor '{col}'")
    if target in df.columns and df[target].isna().any():
        findings.append(f"target '{target}' has NA after prep")
    return findings


# ── Model selection ─────────────────────────────────────────────────────────


def select_best_model(
    models: Sequence[Any],
    *,
    metric: str,
    direction: str | None = None,
) -> Any:
    """Direction-aware best-model selection (minimize RMSE, maximize AUC, …)."""
    if not models:
        raise ValueError("no models to select")
    dir_ = criterion_direction(metric, direction)
    scored = []
    for m in models:
        score = getattr(m, "cv_score", None)
        if score is None and isinstance(m, dict):
            score = m.get("cv_score")
        if score is None or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            continue
        scored.append((m, float(score)))
    if not scored:
        raise ValueError("no models with finite cv_score")
    reverse = dir_ == "maximize"
    scored.sort(key=lambda item: item[1], reverse=reverse)
    return scored[0][0]


def baseline_techniques_for(
    *,
    problem_type: str,
    feature_hints: dict[str, Any] | None = None,
    preferred: str | None = None,
) -> list[str]:
    hints = _feature_hints(feature_hints)
    options = hints.get("representation_options")
    if isinstance(options, list) and options:
        techniques = [str(x) for x in options]
    elif hints.get("text_free"):
        techniques = ["tfidf_logreg"]
    elif problem_type == "regression":
        techniques = ["ridge", "hist_gradient_boosting"]
    else:
        techniques = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    if preferred and preferred not in techniques:
        techniques = [preferred, *techniques]
    elif preferred:
        techniques = [preferred, *[t for t in techniques if t != preferred]]
    normalized: list[str] = []
    for tech in techniques:
        name = tech.lower()
        if name.startswith("openai"):
            name = "tfidf_logreg"
        if name not in normalized:
            normalized.append(name)
    return normalized[:3]


# ── Modeling primitives ─────────────────────────────────────────────────────


def _drop_xy(
    df: pd.DataFrame,
    *,
    target: str,
    id_column: str,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    y = df[target].values if target in df.columns else None
    drop = [c for c in (target, id_column) if c and c in df.columns]
    return df.drop(columns=drop), y


def _classification_y(series: pd.Series) -> tuple[np.ndarray, list[Any], Any]:
    from sklearn.preprocessing import LabelEncoder

    s = pd.Series(series)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all() and (numeric == numeric.round()).all():
        y = numeric.astype(int).values
        return y, sorted(set(y.tolist())), None
    le = LabelEncoder()
    y = le.fit_transform(s.astype(str).values)
    return y, list(le.classes_), le


def _to_1d_str(x):
    return np.asarray(x).ravel().astype(str)


def _build_text_pipeline(
    X: pd.DataFrame,
    *,
    primary_text: str | None,
    problem_type: str,
    n_classes: int,
) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    text_pipe = Pipeline([
        ("to_str", FunctionTransformer(_to_1d_str, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
    ])
    transformers = []
    if primary_text and primary_text in X.columns:
        transformers.append((primary_text, text_pipe, [primary_text]))
    num_cols = X.select_dtypes(include="number").columns.tolist()
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
            num_cols,
        ))
    if not transformers:
        raise RuntimeError("no usable text/numeric columns for tfidf_logreg")
    pre = ColumnTransformer(transformers, remainder="drop")
    solver = "liblinear" if n_classes <= 2 else "lbfgs"
    clf = LogisticRegression(
        max_iter=2000, solver=solver, class_weight="balanced", random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _build_tabular_pipeline(
    X: pd.DataFrame,
    *,
    technique: str,
    problem_type: str,
) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))
    if not transformers:
        raise RuntimeError("no feature columns for tabular baseline")
    pre = ColumnTransformer(transformers, remainder="drop")

    is_reg = problem_type == "regression"
    tech = technique.lower()
    if tech in ("ridge", "linear_regression") or (is_reg and tech == "logistic_regression"):
        est = Ridge(alpha=1.0)
        technique_name = "ridge"
    elif tech in ("random_forest", "rf"):
        if is_reg:
            from sklearn.ensemble import RandomForestRegressor

            est = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        else:
            est = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        technique_name = "random_forest"
    elif tech in ("hist_gradient_boosting", "hgb", "gradient_boosting", "lightgbm"):
        if is_reg:
            est = HistGradientBoostingRegressor(random_state=42)
        else:
            est = HistGradientBoostingClassifier(random_state=42)
        technique_name = "hist_gradient_boosting"
    elif is_reg:
        est = Ridge(alpha=1.0)
        technique_name = "ridge"
    else:
        est = LogisticRegression(max_iter=2000, random_state=42)
        technique_name = "logistic_regression"

    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe._maads_technique = technique_name  # type: ignore[attr-defined]
    return pipe


def _scorer_for(metric: str, *, n_classes: int, class_values: list[Any], is_reg: bool):
    from sklearn.metrics import f1_score, get_scorer, make_scorer, mean_squared_error

    m = (metric or "").lower()
    if is_reg:
        if "log" in m:
            def _rmse_log(y_true, y_pred):
                yt = np.log1p(np.clip(np.asarray(y_true, dtype=float), a_min=0, a_max=None))
                yp = np.log1p(np.clip(np.asarray(y_pred, dtype=float), a_min=0, a_max=None))
                return float(np.sqrt(mean_squared_error(yt, yp)))
            return make_scorer(_rmse_log, greater_is_better=False)
        if m in ("rmse", "mse", "mae"):
            return get_scorer("neg_root_mean_squared_error" if m != "mae" else "neg_mean_absolute_error")
        return get_scorer("neg_root_mean_squared_error")
    if m == "f1":
        if n_classes <= 2 and set(class_values) <= {0, 1}:
            return make_scorer(f1_score, pos_label=1)
        return make_scorer(f1_score, average="macro")
    try:
        return get_scorer(m)
    except Exception:
        return get_scorer("accuracy")


def _primary_text_col(feature_hints: dict[str, Any], columns: list[str]) -> str | None:
    for col in feature_hints.get("text_free") or []:
        if col in columns:
            return str(col)
    if "text" in columns:
        return "text"
    return None


def run_experiment(
    train_parquet: str,
    *,
    technique: str,
    target: str,
    id_column: str,
    metric: str,
    problem_type: str,
    feature_hints: dict[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Fit/CV one baseline pipeline and optionally persist a joblib artifact."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

    hints = _feature_hints(feature_hints)
    train = _read_table(train_parquet)
    X, y_raw = _drop_xy(train, target=target, id_column=id_column)
    if y_raw is None:
        raise RuntimeError(f"target '{target}' missing from train parquet")

    is_reg = problem_type == "regression"
    label_encoder = None
    class_values: list[Any] = []
    if is_reg:
        y = np.asarray(y_raw, dtype=float)
        use_log = "log" in (metric or "").lower()
        y_fit = np.log1p(np.clip(y, a_min=0, a_max=None)) if use_log else y
    else:
        y, class_values, label_encoder = _classification_y(pd.Series(y_raw))
        y_fit = y
        use_log = False

    text_col = _primary_text_col(hints, list(X.columns))
    tech = technique.lower()
    if tech.startswith("tfidf") or (hints.get("text_free") and tech in ("tfidf_logreg", "openai_embeddings_logreg")):
        if tech.startswith("openai"):
            # Embeddings path not available offline — fall back to TF-IDF.
            tech = "tfidf_logreg"
        n_classes = len(set(y.tolist())) if not is_reg else 0
        pipe = _build_text_pipeline(X, primary_text=text_col, problem_type=problem_type, n_classes=max(n_classes, 2))
        technique_name = "tfidf_logreg"
    else:
        pipe = _build_tabular_pipeline(X, technique=tech, problem_type=problem_type)
        technique_name = getattr(pipe, "_maads_technique", tech)

    if is_reg:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scorer = _scorer_for(metric, n_classes=0, class_values=[], is_reg=True)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scorer = _scorer_for(metric, n_classes=len(set(y.tolist())), class_values=class_values, is_reg=False)

    scores = cross_validate(pipe, X, y_fit, cv=cv, scoring={"metric": scorer}, return_train_score=False)
    raw_scores = scores["test_metric"]
    # sklearn neg_* scorers: convert back to positive error for minimize metrics.
    dir_ = criterion_direction(metric)
    if dir_ == "minimize" and float(np.mean(raw_scores)) < 0:
        raw_scores = -raw_scores
    cv_mean = float(np.mean(raw_scores))
    cv_std = float(np.std(raw_scores))

    fitted = clone(pipe)
    fitted.fit(X, y_fit)
    try:
        n_features = int(fitted.named_steps["pre"].transform(X.iloc[:1]).shape[1])
    except Exception:
        n_features = int(X.shape[1])

    artifact_path = ""
    if artifact_dir is not None:
        import joblib

        out_dir = Path(artifact_dir) / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str((out_dir / f"{technique_name}.joblib").resolve())
        joblib.dump(
            {
                "pipeline": fitted,
                "technique": technique_name,
                "use_log_target": use_log,
                "label_encoder": label_encoder,
                "target": target,
                "id_column": id_column,
                "metric": metric,
                "problem_type": problem_type,
                "feature_columns": list(X.columns),
            },
            artifact_path,
        )

    return {
        "technique": technique_name,
        "cv_score": cv_mean,
        "cv_std": cv_std,
        "n_features": n_features,
        "parameter_settings": {
            "artifact_path": artifact_path,
            "use_log_target": use_log,
            "cv_folds": cv_folds,
        },
        "artifact_path": artifact_path,
        "source": "deterministic run_experiment",
    }


def compare_experiments(
    train_parquet: str,
    techniques: Sequence[str],
    *,
    target: str,
    id_column: str,
    metric: str,
    problem_type: str,
    feature_hints: dict[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    runs = []
    for tech in techniques:
        try:
            runs.append(
                run_experiment(
                    train_parquet,
                    technique=tech,
                    target=target,
                    id_column=id_column,
                    metric=metric,
                    problem_type=problem_type,
                    feature_hints=feature_hints,
                    artifact_dir=artifact_dir,
                ),
            )
        except Exception as exc:  # noqa: BLE001 — skip failed ladder step
            runs.append({
                "technique": tech,
                "cv_score": None,
                "cv_std": None,
                "error": str(exc)[:300],
            })
    viable = [r for r in runs if isinstance(r.get("cv_score"), (int, float))]
    if not viable:
        raise RuntimeError(f"all baseline techniques failed: {runs}")
    best = select_best_model(viable, metric=metric, direction=direction)
    return {"candidates": runs, "best": best}


def assess_fitted_experiment(
    train_parquet: str,
    *,
    technique: str,
    target: str,
    id_column: str,
    metric: str,
    problem_type: str,
    feature_hints: dict[str, Any] | None = None,
    artifact_path: str | None = None,
    figures_dir: str | Path | None = None,
    class_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """OOF evaluation; prefer loading the exact joblib artifact when present."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        mean_squared_error,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import KFold, StratifiedKFold

    hints = _feature_hints(feature_hints)
    train = _read_table(train_parquet)
    X, y_raw = _drop_xy(train, target=target, id_column=id_column)
    if y_raw is None:
        raise RuntimeError(f"target '{target}' missing")

    is_reg = problem_type == "regression"
    bundle_warnings: list[str] = []
    pipe = None
    use_log = "log" in (metric or "").lower()
    label_encoder = None

    if artifact_path and Path(artifact_path).is_file():
        import joblib

        blob = joblib.load(artifact_path)
        pipe = blob.get("pipeline")
        use_log = bool(blob.get("use_log_target", use_log))
        label_encoder = blob.get("label_encoder")
        technique = str(blob.get("technique") or technique)

    if is_reg:
        y = np.asarray(y_raw, dtype=float)
        y_fit = np.log1p(np.clip(y, a_min=0, a_max=None)) if use_log else y
        class_values: list[Any] = []
    else:
        y, class_values, le = _classification_y(pd.Series(y_raw))
        if label_encoder is None:
            label_encoder = le
        y_fit = y

    if pipe is None:
        text_col = _primary_text_col(hints, list(X.columns))
        tech = technique.lower()
        if tech.startswith("tfidf") or hints.get("text_free"):
            n_classes = len(set(y.tolist())) if not is_reg else 2
            pipe = _build_text_pipeline(
                X, primary_text=text_col, problem_type=problem_type, n_classes=n_classes,
            )
            technique = "tfidf_logreg"
        else:
            pipe = _build_tabular_pipeline(X, technique=tech, problem_type=problem_type)
            technique = getattr(pipe, "_maads_technique", tech)

    fig_paths: list[str] = []
    if figures_dir:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

    if is_reg:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y_fit), dtype=float)
        fold_scores = []
        for tr_idx, va_idx in cv.split(X):
            est = clone(pipe)
            est.fit(X.iloc[tr_idx], y_fit[tr_idx])
            pred = est.predict(X.iloc[va_idx])
            oof[va_idx] = pred
            yt, yp = y_fit[va_idx], pred
            fold_scores.append(float(np.sqrt(mean_squared_error(yt, yp))))
        if use_log:
            oof_report = np.expm1(oof)
            y_report = y
            rmse = float(np.sqrt(mean_squared_error(np.log1p(y_report), np.log1p(np.clip(oof_report, 0, None)))))
        else:
            oof_report = oof
            rmse = float(np.mean(fold_scores))
        metrics = {"rmse": rmse, "rmse_log" if use_log else "rmse_raw": rmse}
        evaluation_bundle = {
            "problem_type": problem_type,
            "metrics": metrics,
            "confusion_matrix": [],
            "class_labels": class_labels or {},
            "cv": {"mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores)), "n_folds": 5},
            "figures": fig_paths,
            "warnings": bundle_warnings,
        }
        return {
            "evaluation_bundle": evaluation_bundle,
            "assessment": f"OOF evaluation via {technique} artifact",
            "technique": technique,
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=int)
    cv_scores = []
    n_classes = len(set(y.tolist()))
    binary_01 = n_classes <= 2 and label_encoder is None and set(class_values) <= {0, 1}
    for tr_idx, va_idx in cv.split(X, y):
        est = clone(pipe)
        est.fit(X.iloc[tr_idx], y[tr_idx])
        preds = est.predict(X.iloc[va_idx])
        oof_pred[va_idx] = preds
        if binary_01:
            cv_scores.append(float(f1_score(y[va_idx], preds, pos_label=1)))
        else:
            cv_scores.append(float(f1_score(y[va_idx], preds, average="macro")))

    cm_labels = sorted(set(y.tolist()))
    class_labels_map = class_labels or {}
    label_names = []
    for lbl in cm_labels:
        if label_encoder is not None:
            label_names.append(str(class_values[lbl]))
        else:
            label_names.append(class_labels_map.get(str(lbl), str(lbl)))

    cm = confusion_matrix(y, oof_pred, labels=cm_labels).tolist()
    prec, rec, f1, _support = precision_recall_fscore_support(
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

    if figures_dir:
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
        p_cm = str(Path(figures_dir) / "confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(p_cm, dpi=150)
        plt.close(fig)
        fig_paths.append(p_cm)

    evaluation_bundle = {
        "problem_type": problem_type,
        "metrics": metrics,
        "confusion_matrix": cm,
        "class_labels": class_labels_map,
        "cv": {"mean": float(np.mean(cv_scores)), "std": float(np.std(cv_scores)), "n_folds": 5},
        "figures": fig_paths,
        "warnings": bundle_warnings,
    }
    return {
        "evaluation_bundle": evaluation_bundle,
        "assessment": f"OOF evaluation via {technique} artifact",
        "technique": technique,
    }


def predict_from_artifact(
    artifact_path: str,
    *,
    train_parquet: str,
    test_parquet: str,
    sample_submission: str,
    output_path: str,
    target: str,
    id_column: str,
) -> dict[str, Any]:
    """Load persisted pipeline and write a Kaggle-shaped submission."""
    import joblib

    blob = joblib.load(artifact_path)
    pipe = blob["pipeline"]
    use_log = bool(blob.get("use_log_target"))
    label_encoder = blob.get("label_encoder")
    train = _read_table(train_parquet)
    test = _read_table(test_parquet)
    X_train, y_raw = _drop_xy(train, target=target, id_column=id_column)
    X_test, _ = _drop_xy(test, target=target, id_column=id_column)
    X_test = X_test.reindex(columns=list(X_train.columns), fill_value=np.nan)

    # Refit on full train for deployment parity with stored feature schema.
    if y_raw is None:
        raise RuntimeError("cannot refit: target missing")
    is_class = label_encoder is not None or not pd.api.types.is_numeric_dtype(pd.Series(y_raw))
    if blob.get("problem_type") == "regression" or (
        not is_class and pd.api.types.is_numeric_dtype(pd.Series(y_raw))
    ):
        y = np.asarray(y_raw, dtype=float)
        y_fit = np.log1p(np.clip(y, a_min=0, a_max=None)) if use_log else y
        pipe.fit(X_train, y_fit)
        preds = pipe.predict(X_test)
        if use_log:
            preds = np.expm1(preds)
    else:
        y, _classes, le = _classification_y(pd.Series(y_raw))
        if label_encoder is None:
            label_encoder = le
        pipe.fit(X_train, y)
        encoded = pipe.predict(X_test)
        if label_encoder is not None:
            preds = label_encoder.inverse_transform(encoded)
        else:
            preds = encoded

    sample_path = Path(sample_submission) if sample_submission else None
    if sample_path and sample_path.is_file():
        sample = pd.read_csv(sample_path)
        idc = id_column if id_column in sample.columns else sample.columns[0]
        target_col = target if target in sample.columns else sample.columns[1]
        if id_column in test.columns:
            sub = pd.DataFrame({idc: test[id_column].values, target_col: preds})
        else:
            sub = pd.DataFrame({idc: sample[idc].values, target_col: preds})
        sub = sub.reindex(columns=list(sample.columns))
    else:
        idc = id_column if id_column in test.columns else "id"
        ids = test[idc].values if idc in test.columns else range(len(preds))
        sub = pd.DataFrame({idc: ids, target: preds})

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    return {"submission_path": str(out.resolve()), "rows": int(len(sub)), "source": "artifact"}
