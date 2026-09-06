"""Plain-language conclusions per run, for non-technical readers.

Everything here is deterministic (computed from the recorded metrics), so the
text is trustworthy and reproducible — no extra LLM calls.
"""
from __future__ import annotations

from typing import Any

_MINIMIZE = ("rmse", "rmsle", "mae", "mse", "mape", "loss", "error")


def _lower_better(metric: str | None) -> bool:
    m = (metric or "").lower()
    return any(t in m for t in _MINIMIZE)


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _recall(cm: list[list[int]] | None) -> float | None:
    """Recall of the positive (last) class for a binary confusion matrix."""
    if not cm or len(cm) != 2 or len(cm[0]) != 2:
        return None
    fn, tp = cm[1][0], cm[1][1]
    denom = tp + fn
    return tp / denom if denom else None


def build_run_insight(row: dict[str, Any]) -> dict[str, Any]:
    """Return {'summary': str, 'flags': [str]} for one run row."""
    parts: list[str] = []
    flags: list[str] = []
    metric = row.get("score_metric")
    metrics = row.get("metrics") or {}
    score = metrics.get(metric) if metric else row.get("score")
    thr = row.get("success_threshold")

    if score is not None and thr is not None:
        verb = "Met the goal" if row.get("meets_threshold") else "Fell short of the goal"
        parts.append(f"{verb} — {metric} {score:.3f} vs target {thr}.")
    elif score is not None:
        parts.append(f"{metric} {score:.3f}.")

    cm = row.get("confusion_matrix")
    labels = row.get("class_labels") or {}
    if cm and len(cm) == 2 and len(cm[0]) == 2:
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        pos = labels.get("1") or "the positive class"
        actual_pos = tp + fn
        pred_pos = tp + fp
        recall = tp / actual_pos if actual_pos else 0.0
        precision = tp / pred_pos if pred_pos else 0.0
        parts.append(
            f"Out of {actual_pos} real “{pos}”, it correctly found "
            f"{tp} ({_pct(recall)}) and missed {fn}. Of everything it labelled "
            f"“{pos}”, {_pct(precision)} were actually correct."
        )
        if recall < 0.7:
            flags.append(f"Misses {_pct(1 - recall)} of “{pos}”")
        if precision < 0.6:
            flags.append("Many false alarms")
    elif cm:
        total = sum(sum(r) for r in cm)
        correct = sum(cm[i][i] for i in range(len(cm)))
        if total:
            parts.append(
                f"Correctly classified {_pct(correct / total)} "
                f"({correct}/{total}) across {len(cm)} classes."
            )

    if row.get("workflow_complete") is False:
        parts.append("The run did not finish cleanly — treat the result as partial.")
        flags.append("Did not finish")

    return {
        "summary": " ".join(parts) or "No evaluation recorded for this run yet.",
        "flags": flags,
    }


def attach_badges(rows: list[dict[str, Any]]) -> None:
    """Add cross-run 'badges' (best score / cheapest / best recall) in place."""
    for r in rows:
        r["badges"] = []

    def _primary_metric(r: dict[str, Any]) -> float | None:
        m = r.get("score_metric")
        if not m:
            return None
        v = (r.get("metrics") or {}).get(m)
        return float(v) if isinstance(v, (int, float)) else None

    scored = [r for r in rows if _primary_metric(r) is not None]
    if scored:
        metric = scored[0].get("score_metric")
        best = (min if _lower_better(metric) else max)(
            scored, key=lambda r: _primary_metric(r) or 0.0
        )
        best["badges"].append("Best score")

    costed = [r for r in rows if isinstance(r.get("total_tokens"), (int, float))]
    if costed:
        cheapest = min(costed, key=lambda r: r["total_tokens"])
        cheapest["badges"].append("Most efficient")

    recalls = [(r, _recall(r.get("confusion_matrix"))) for r in rows]
    recalls = [(r, v) for r, v in recalls if v is not None]
    if recalls:
        top = max(recalls, key=lambda rv: rv[1])[0]
        top["badges"].append("Best at catching positives")
