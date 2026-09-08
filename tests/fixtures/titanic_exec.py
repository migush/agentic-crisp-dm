"""Titanic-oriented codegen stubs for tests (replaces removed DE/DS baselines)."""
from __future__ import annotations

from maads.state import CrispDMState

_COLLECT = '''```python
import pandas as pd, json
tr = pd.read_csv(TRAIN_CSV)
if TEST_CSV:
    te = pd.read_csv(TEST_CSV)
    test_rows = int(len(te))
else:
    test_rows = 0
print(json.dumps({"train_rows": int(len(tr)), "test_rows": test_rows, "columns": list(tr.columns)}))
```'''

_DESCRIBE = '''```python
import pandas as pd, json
df = pd.read_csv(TRAIN_CSV)
print(json.dumps({
    "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
    "columns": list(df.columns),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "missing": {c: int(df[c].isna().sum()) for c in df.columns},
}))
```'''

_QUALITY = '''```python
import pandas as pd, json
df = pd.read_csv(TRAIN_CSV)
target = TARGET
blockers = []; tolerable = []
for c in df.columns:
    miss = float(df[c].isna().mean())
    if miss > 0.4:
        blockers.append(c + ": %.0f%% missing" % (miss * 100))
    elif miss > 0:
        tolerable.append(c + ": %.0f%% missing (imputable)" % (miss * 100))
if target in df.columns and bool(df[target].isna().any()):
    blockers.append(target + ": target has missing values")
print(json.dumps({"blockers": blockers, "tolerable": tolerable}))
```'''

_EXPLORE = '''```python
import pandas as pd, json
df = pd.read_csv(TRAIN_CSV)
target = TARGET
out = {"n_rows": int(len(df)), "target": target}
if target in df.columns:
    out["target_distribution"] = {str(k): int(v) for k, v in df[target].value_counts().items()}
print(json.dumps(out))
```'''

_CLEAN = '''```python
import pandas as pd, json, os
def read_table(path):
    if not path:
        return None
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
os.makedirs(OUTDIR, exist_ok=True)
tr = read_table(TRAIN_IN); te = read_table(TEST_IN)
if te is None:
    te = tr.iloc[0:0].copy()
    if TARGET and TARGET in te.columns:
        te = te.drop(columns=[TARGET])
missing_before = {c: int(tr[c].isna().sum()) for c in tr.columns}
if "Age" in tr.columns:
    med = float(tr["Age"].median())
    tr["Age"] = tr["Age"].fillna(med)
    if "Age" in te.columns:
        te["Age"] = te["Age"].fillna(med)
if "Fare" in tr.columns:
    med = float(tr["Fare"].median())
    tr["Fare"] = tr["Fare"].fillna(med)
    if "Fare" in te.columns:
        te["Fare"] = te["Fare"].fillna(med)
if "Embarked" in tr.columns:
    mode = tr["Embarked"].mode()
    fill = str(mode.iloc[0]) if len(mode) else "S"
    tr["Embarked"] = tr["Embarked"].fillna(fill)
    if "Embarked" in te.columns:
        te["Embarked"] = te["Embarked"].fillna(fill)
missing_after = {c: int(tr[c].isna().sum()) for c in tr.columns}
tp = os.path.join(OUTDIR, "train_clean.parquet")
sp = os.path.join(OUTDIR, "test_clean.parquet")
tr.to_parquet(tp); te.to_parquet(sp)
print(json.dumps({"train_out": tp, "test_out": sp, "missing_before": missing_before,
                  "missing_after": missing_after, "operations": ["median/mode imputation"]}))
```'''

_CONSTRUCT = '''```python
import pandas as pd, json, os
def read_table(path):
    if not path:
        return None
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
os.makedirs(OUTDIR, exist_ok=True)
tr = read_table(TRAIN_IN); te = read_table(TEST_IN)
if te is None:
    te = tr.iloc[0:0].copy()
    if TARGET and TARGET in te.columns:
        te = te.drop(columns=[TARGET])
derived = []
if "SibSp" in tr.columns and "Parch" in tr.columns:
    for df in (tr, te):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    derived = ["FamilySize", "IsAlone"]
tp = os.path.join(OUTDIR, "train_constructed.parquet")
sp = os.path.join(OUTDIR, "test_constructed.parquet")
tr.to_parquet(tp); te.to_parquet(sp)
print(json.dumps({"train_out": tp, "test_out": sp, "derived": derived}))
```'''

_INTEGRATE = '''```python
import pandas as pd, json, os
def read_table(path):
    if not path:
        return None
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
os.makedirs(OUTDIR, exist_ok=True)
tr = read_table(TRAIN_IN); te = read_table(TEST_IN)
if te is None:
    te = tr.iloc[0:0].copy()
    if TARGET and TARGET in te.columns:
        te = te.drop(columns=[TARGET])
tp = os.path.join(OUTDIR, "train_integrated.parquet")
sp = os.path.join(OUTDIR, "test_integrated.parquet")
tr.to_parquet(tp); te.to_parquet(sp)
print(json.dumps({"train_out": tp, "test_out": sp,
                  "train_rows": int(len(tr)), "test_rows": int(len(te)),
                  "columns_train": list(tr.columns), "columns_test": list(te.columns)}))
```'''

_FORMAT = '''```python
import pandas as pd, json, os
def read_table(path):
    if not path:
        return None
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
tr = read_table(TRAIN_IN); te = read_table(TEST_IN)
if te is None:
    te = tr.iloc[0:0].copy()
    if TARGET and TARGET in te.columns:
        te = te.drop(columns=[TARGET])
target = TARGET; idc = ID_COL
dropped = []
for col in ("Name", "Ticket", "Cabin"):
    if col in tr.columns:
        dropped.append(col)
        tr = tr.drop(columns=[col])
        te = te.drop(columns=[col], errors="ignore")
os.makedirs(OUTDIR, exist_ok=True)
tp = os.path.join(OUTDIR, "train.parquet")
sp = os.path.join(OUTDIR, "test.parquet")
tr.to_parquet(tp); te.to_parquet(sp)
src = read_table(SOURCE_TRAIN)
derived = [c for c in tr.columns if src is not None and c not in src.columns and c not in (target, idc)]
print(json.dumps({"train": tp, "test": sp, "n_train": int(len(tr)), "n_test": int(len(te)),
                  "derived": derived, "dropped": dropped}))
```'''

_TRAIN = '''```python
import pandas as pd, json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
train = pd.read_parquet(TRAIN_PARQUET)
target = TARGET; idc = ID_COL
y = train[target]
X = train.drop(columns=[c for c in (target, idc) if c in train.columns])
num = X.select_dtypes(include="number").columns.tolist()
cat = [c for c in X.select_dtypes(exclude="number").columns if X[c].nunique() <= 20]
if not num and not cat:
    cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
], remainder="drop")
pipe = Pipeline([("pre", pre), ("gb", GradientBoostingClassifier(random_state=0))])
feats = num + cat
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X[feats], y, cv=cv, scoring="accuracy")
print(json.dumps({"technique": "gradient_boosting", "cv_score": float(scores.mean()),
                  "cv_std": float(scores.std()), "n_features": int(len(feats))}))
```'''

_ASSESS = '''```python
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
train = pd.read_parquet(TRAIN_PARQUET)
target = TARGET; idc = ID_COL
problem_type = PROBLEM_TYPE
figures_dir = FIGURES_DIR
os.makedirs(figures_dir, exist_ok=True)
class_labels = json.loads(CLASS_LABELS)
y = train[target]
X = train.drop(columns=[c for c in (target, idc) if c in train.columns])
num = X.select_dtypes(include="number").columns.tolist()
cat = [c for c in X.select_dtypes(exclude="number").columns if X[c].nunique() <= 20]
if not num and not cat:
    cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
], remainder="drop")
pipe = Pipeline([("pre", pre), ("gb", GradientBoostingClassifier(random_state=0))])
feats = num + cat
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof_pred = np.empty(len(y), dtype=object)
cv_scores = []
for tr_idx, va_idx in cv.split(X[feats], y):
    pipe.fit(X[feats].iloc[tr_idx], y.iloc[tr_idx])
    preds = pipe.predict(X[feats].iloc[va_idx])
    oof_pred[va_idx] = preds
    cv_scores.append(float(accuracy_score(y.iloc[va_idx], preds)))
labels_sorted = list(pd.unique(y))
cm = confusion_matrix(y, oof_pred, labels=labels_sorted).tolist()
prec, rec, f1, support = precision_recall_fscore_support(y, oof_pred, labels=labels_sorted, average=None, zero_division=0)
metrics = {"accuracy": float(accuracy_score(y, oof_pred)), "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred))}
for i, lbl in enumerate(labels_sorted):
    key = str(lbl)
    name = class_labels.get(key, class_labels.get(str(int(lbl)) if str(lbl).lstrip("-").isdigit() else key, key))
    metrics[f"precision_{name}"] = float(prec[i])
    metrics[f"recall_{name}"] = float(rec[i])
    metrics[f"f1_{name}"] = float(f1[i])
fig_paths = []
fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts()
names = [class_labels.get(str(k), str(k)) for k in counts.index]
ax.bar([str(n) for n in names], counts.values)
p1 = os.path.join(figures_dir, "class_distribution.png")
fig.tight_layout(); fig.savefig(p1); plt.close(fig)
fig_paths.append("figures/class_distribution.png")
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
p2 = os.path.join(figures_dir, "confusion_matrix.png")
fig.tight_layout(); fig.savefig(p2); plt.close(fig)
fig_paths.append("figures/confusion_matrix.png")
bundle = {
    "problem_type": problem_type,
    "metrics": metrics,
    "confusion_matrix": cm,
    "class_labels": class_labels,
    "cv": {"mean": float(np.mean(cv_scores)), "std": float(np.std(cv_scores)), "n_folds": 5},
    "figures": fig_paths,
    "warnings": [],
}
print(json.dumps({"evaluation_bundle": bundle, "assessment": "OOF evaluation"}))
```'''

_SUBMIT = '''```python
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_parquet(TRAIN_PARQUET)
test = pd.read_parquet(TEST_PARQUET)
target = TARGET
idc = ID_COL
y = train[target]
X = train.drop(columns=[c for c in (target, idc) if c in train.columns])
num = X.select_dtypes(include="number").columns.tolist()
cat = [c for c in X.select_dtypes(exclude="number").columns if X[c].nunique() <= 20]
if not num and not cat:
    cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
], remainder="drop")
pipe = Pipeline([("pre", pre), ("gb", GradientBoostingClassifier(random_state=0))])
feats = num + cat
pipe.fit(X[feats], y)
Xt = test.reindex(columns=feats, fill_value=0)
preds = pipe.predict(Xt)
id_use = idc if idc in test.columns else (test.columns[0] if len(test.columns) else "id")
target_use = target or "prediction"
ids = test[id_use] if id_use in test.columns else range(len(preds))
sub = pd.DataFrame({id_use: ids, target_use: preds})
if SAMPLE_SUBMISSION:
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    sub = sub.reindex(columns=list(sample.columns))
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)
sub.to_csv(OUTPUT_PATH, index=False)
print(json.dumps({"submission_path": OUTPUT_PATH, "rows": int(len(sub))}))
```'''

_DE_MAP = {
    "2.1": _COLLECT,
    "2.2": _DESCRIBE,
    "2.4": _QUALITY,
    "3.2": _CLEAN,
    "3.3": _CONSTRUCT,
    "3.4": _INTEGRATE,
    "3.5": _FORMAT,
}

_DS_MAP = {
    "2.3": _EXPLORE,
    "4.3": _TRAIN,
    "4.4": _ASSESS,
}


def fake_run_text_task(agent_name: str, instruction: str, state: CrispDMState, **kwargs) -> str:
    """Return executable Python for DE/DS substeps in integration tests."""
    sub = state.substep
    if agent_name == "data_engineer" and sub in _DE_MAP:
        return _DE_MAP[sub]
    if agent_name == "data_scientist" and sub in _DS_MAP:
        return _DS_MAP[sub]
    if agent_name == "developer" and sub == "6.1":
        return _SUBMIT
    if agent_name == "developer":
        return kwargs.get("expected_output", "") or ""
    return ""
