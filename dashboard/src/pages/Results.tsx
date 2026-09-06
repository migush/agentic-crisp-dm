import { useEffect, useState } from "react";
import { Loading } from "../components/Loading";
import { useCaseResults, useCases } from "../hooks/useCasePolling";
import { formatDuration, prettyCase } from "../shared/format";
import { openStyledReport } from "../shared/reportPdf";
import { useTheme } from "../shared/theme";
import type { RunResult } from "../shared/types";

// Metrics where a *lower* value is better.
const LOWER_BETTER = /rmse|rmsle|\bmae\b|\bmse\b|mape|loss|error/i;
// Preferred display order; unknown metrics fall after these, alphabetically.
const METRIC_ORDER = [
  "accuracy",
  "balanced_accuracy",
  "precision",
  "recall",
  "f1",
  "roc_auc",
  "rmse",
  "mae",
  "mse",
  "r2",
  "mape",
];

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1 Score",
  roc_auc: "ROC AUC",
  rmse: "RMSE",
  mae: "MAE",
  mse: "MSE",
  r2: "R²",
  mape: "MAPE",
  log_loss: "Log Loss",
};

function metricLabel(k: string): string {
  return (
    METRIC_LABELS[k] ??
    k.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase())
  );
}

// Plain-words explanation + formula, shown on hover over a metric.
const METRIC_INFO: Record<string, string> = {
  accuracy:
    "Share of ALL predictions that were correct. Formula: (TP + TN) / everything. Higher is better.",
  balanced_accuracy:
    "Average recall across the classes — fairer than accuracy when one class is rarer. Higher is better.",
  precision:
    "Of everything the model flagged as positive, how many really were. Formula: TP / (TP + FP). Higher is better.",
  recall:
    "Of all the real positives, how many the model actually found. Formula: TP / (TP + FN). Higher is better.",
  f1: "Single score balancing precision and recall (their harmonic mean). Formula: 2·P·R / (P + R). Higher is better.",
  roc_auc:
    "How well the model ranks positives above negatives across all thresholds. 0.5 = random, 1.0 = perfect.",
  rmse: "Typical prediction error in the target's units. Formula: sqrt(mean((pred − actual)²)). Lower is better.",
  mae: "Average size of the error. Formula: mean(|pred − actual|). Lower is better.",
  mse: "Average squared error — punishes big misses harder. Lower is better.",
  r2: "Share of the variation the model explains. 1.0 = perfect, 0 = no better than the mean. Higher is better.",
  mape: "Average error as a percentage of the true value. Lower is better.",
};

function metricInfo(k: string): string {
  return METRIC_INFO[k] ?? "Model evaluation metric.";
}

// How a derived feature is typically built, shown on hover.
const FEATURE_INFO: Record<string, string> = {
  familysize:
    "Family size aboard = SibSp (siblings/spouses) + Parch (parents/children) + 1 (the passenger).",
  isalone: "1 if the passenger travelled alone (family size = 1), otherwise 0.",
  farebin: "The ticket fare grouped into ranges (bins) instead of a raw number.",
  agebin: "Age grouped into ranges (e.g. child / adult / senior) instead of a raw number.",
  title: "Honorific pulled from the name (Mr, Mrs, Miss, Master…), a proxy for age/sex/status.",
  deck: "Deck letter taken from the cabin code.",
  cabindeck: "Deck letter taken from the cabin code.",
  fareperperson: "Fare divided by family size — the cost per traveller.",
};

function featureInfo(f: string): string {
  return (
    FEATURE_INFO[f.toLowerCase()] ??
    "Feature the model engineered from the raw columns."
  );
}

const MODEL_LABELS: Record<string, string> = {
  random_forest: "Random Forest",
  extra_trees: "Extra Trees",
  decision_tree: "Decision Tree",
  logistic_regression: "Logistic Regression",
  linear_regression: "Linear Regression",
  ridge: "Ridge Regression",
  lasso: "Lasso Regression",
  elastic_net: "Elastic Net",
  gradient_boosting: "Gradient Boosting",
  hist_gradient_boosting: "Hist Gradient Boosting",
  ada_boost: "AdaBoost",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  svm: "SVM",
  svc: "SVM",
  knn: "K-Nearest Neighbors",
  naive_bayes: "Naive Bayes",
  mlp: "Neural Network (MLP)",
};

function modelLabel(k: string | null): string {
  if (!k) return "—";
  return (
    MODEL_LABELS[k.toLowerCase()] ??
    k.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase())
  );
}

function fmt(n: number): string {
  if (Math.abs(n) >= 1000)
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return n.toFixed(4);
}

function fmtWhen(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function metricKeys(runs: RunResult[]): string[] {
  const all = new Set<string>();
  runs.forEach((r) => Object.keys(r.metrics ?? {}).forEach((k) => all.add(k)));
  return [...all].sort((a, b) => {
    const ia = METRIC_ORDER.indexOf(a);
    const ib = METRIC_ORDER.indexOf(b);
    if (ia !== -1 || ib !== -1)
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    return a.localeCompare(b);
  });
}

function bestValue(runs: RunResult[], key: string): number | null {
  const vals = runs
    .map((r) => r.metrics?.[key])
    .filter((v): v is number => typeof v === "number");
  if (!vals.length) return null;
  return LOWER_BETTER.test(key) ? Math.min(...vals) : Math.max(...vals);
}

function statusInfo(r: RunResult) {
  if (r.status === "running")
    return { label: "Running", dot: "bg-status-running", active: true };
  if (r.status === "halted" && !r.workflow_complete)
    return { label: "Halted", dot: "bg-status-halted", active: false };
  return { label: "Complete", dot: "bg-status-complete", active: false };
}

function ConfusionMatrix({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: Record<string, string>;
}) {
  const names = matrix.map((_, i) => labels[String(i)] ?? `C${i}`);
  return (
    <div
      className="inline-grid gap-0.5"
        style={{ gridTemplateColumns: `repeat(${matrix.length}, minmax(0,1fr))` }}
      >
        {matrix.flatMap((rowArr, i) =>
          rowArr.map((v, j) => {
            const correct = i === j;
            return (
              <div
                key={`${i}-${j}`}
                title={`actual ${names[i]} → predicted ${names[j]}: ${v}`}
                className={`rounded px-2 py-1.5 text-center text-xs font-semibold tabular-nums ${
                  correct
                    ? "bg-status-complete/20 text-status-complete"
                    : v > 0
                    ? "bg-status-halted/15 text-status-halted"
                    : "bg-surface text-slate-500"
                }`}
              >
                {v}
              </div>
            );
          }),
        )}
    </div>
  );
}

export function Results() {
  const { clean } = useTheme();
  const { data: cases } = useCases();
  const [caseId, setCaseId] = useState<string>("");

  // Reverse-alphabetical so titanic comes first.
  const sortedCases = cases
    ? [...cases].sort((a, b) => b.case_id.localeCompare(a.case_id))
    : [];

  useEffect(() => {
    if (!caseId && sortedCases.length) setCaseId(sortedCases[0].case_id);
  }, [sortedCases, caseId]);

  const { data: results, isLoading } = useCaseResults(caseId || null);
  const runs = results ?? [];
  const keys = metricKeys(runs);

  // Derived features NOT shared by every run — these are what set the models
  // apart and explain the differing metrics.
  const featCount = new Map<string, number>();
  runs.forEach((r) =>
    (r.derived_features ?? []).forEach((f) =>
      featCount.set(f, (featCount.get(f) ?? 0) + 1),
    ),
  );
  const diffFeatures = new Set(
    [...featCount.entries()]
      .filter(([, c]) => c < runs.length)
      .map(([f]) => f),
  );
  const cols = `170px repeat(${runs.length}, minmax(180px, 1fr))`;

  const labelCell =
    "border-b border-r border-surface-border/50 px-3 py-2 text-xs font-semibold text-slate-400 bg-surface/40";
  const cell =
    "border-b border-surface-border/50 px-3 py-2 text-sm text-slate-200";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-100">
            {clean("📊 Results — model comparison")}
          </h2>
          <p className="text-sm text-slate-400">
            Language models side by side on the same dataset — compare each row
            at a glance. Best value per metric is highlighted.
          </p>
        </div>
        <select
          value={caseId}
          onChange={(e) => setCaseId(e.target.value)}
          className="rounded-full border border-surface-border bg-surface px-4 py-1.5 text-sm font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-accent/40"
        >
          {!sortedCases.length && (
            <option value="">{clean("No datasets 😴")}</option>
          )}
          {sortedCases.map((c) => (
            <option key={c.case_id} value={c.case_id}>
              {prettyCase(c.case_id)}
            </option>
          ))}
        </select>
      </div>

      {isLoading && !results && <Loading label="Loading results…" />}

      {results && runs.length === 0 && (
        <p className="text-slate-500">
          {clean("No runs for this dataset yet — launch one to compare. 🚀")}
        </p>
      )}

      {runs.length > 0 && (
        <div className="overflow-x-auto rounded-2xl border border-surface-border bg-surface-raised glow-card">
          <div className="min-w-fit" style={{ display: "grid", gridTemplateColumns: cols }}>
            {/* Header: model names */}
            <div className={`${labelCell} flex items-end`}>{clean("Language model")}</div>
            {runs.map((r) => {
              const s = statusInfo(r);
              return (
                <div
                  key={r.run_id}
                  className="border-b border-surface-border/50 px-3 py-2"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-2 w-2 shrink-0 rounded-full ${s.dot} ${
                        s.active ? "node-active" : ""
                      }`}
                    />
                    <span className="break-words font-mono text-sm font-bold text-slate-100">
                      {r.llm_model}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[11px] text-slate-500">
                    {s.label}
                    {r.problem_type ? ` · ${r.problem_type}` : ""}
                  </div>
                  {Boolean(
                    (r.badges?.length ?? 0) + (r.insight?.flags.length ?? 0),
                  ) && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {r.badges?.map((b) => (
                        <span
                          key={b}
                          className="rounded-full bg-status-complete/15 px-2 py-0.5 text-[10px] font-semibold text-status-complete"
                        >
                          {b}
                        </span>
                      ))}
                      {r.insight?.flags.map((f) => (
                        <span
                          key={f}
                          className="rounded-full bg-amber-400/15 px-2 py-0.5 text-[10px] font-semibold text-amber-400"
                        >
                          {f}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Success threshold (case goal) */}
            <div className={labelCell}>
              {clean("Success threshold")}
              {runs[0]?.score_metric ? ` (${runs[0].score_metric})` : ""}
              <div className="mt-0.5 font-normal normal-case text-[10px] text-slate-500">
                goal from case config
              </div>
            </div>
            {runs.map((r) => (
              <div key={r.run_id} className={cell}>
                {r.success_threshold != null ? (
                  <span className="font-semibold tabular-nums text-slate-400">
                    ≥ {fmt(r.success_threshold)}
                  </span>
                ) : (
                  "—"
                )}
              </div>
            ))}

            {/* Metrics section */}
            {keys.length > 0 && (
              <div
                className="border-b border-surface-border/50 bg-surface/60 px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500"
                style={{ gridColumn: `1 / span ${runs.length + 1}` }}
              >
                Metrics
              </div>
            )}
            {keys.map((k) => {
              const best = bestValue(runs, k);
              return (
                <div key={`row-${k}`} style={{ display: "contents" }}>
                  <div className={`${labelCell} cursor-help`} title={metricInfo(k)}>
                    {metricLabel(k)}
                    <span className="text-slate-500">
                      {LOWER_BETTER.test(k) ? " ↓" : " ↑"}
                    </span>
                  </div>
                  {runs.map((r) => {
                    const v = r.metrics?.[k];
                    const isBest =
                      typeof v === "number" && best != null && v === best;
                    return (
                      <div
                        key={r.run_id}
                        className={`${cell} tabular-nums ${
                          isBest
                            ? "bg-status-complete/15 font-bold text-status-complete"
                            : ""
                        }`}
                      >
                        {typeof v === "number" ? fmt(v) : "—"}
                      </div>
                    );
                  })}
                </div>
              );
            })}

            {/* Confusion matrix (classification) */}
            {runs.some((r) => r.confusion_matrix) && (
              <>
                <div className={labelCell}>{clean("Confusion matrix")}</div>
                {runs.map((r) => (
                  <div key={r.run_id} className={cell}>
                    {r.confusion_matrix ? (
                      <ConfusionMatrix
                        matrix={r.confusion_matrix}
                        labels={r.class_labels ?? {}}
                      />
                    ) : (
                      "—"
                    )}
                  </div>
                ))}
              </>
            )}

            {/* Chosen model */}
            <div className={labelCell}>{clean("Chosen model")}</div>
            {runs.map((r) => (
              <div key={r.run_id} className={cell}>
                {modelLabel(r.chosen_model)}
              </div>
            ))}

            {/* Derived features */}
            <div className={labelCell}>{clean("Derived features")}</div>
            {runs.map((r) => (
              <div key={r.run_id} className={`${cell} text-slate-300`}>
                {r.derived_summary ?? "—"}
              </div>
            ))}

            {/* What sets the models apart */}
            {diffFeatures.size > 0 && (
              <>
                <div className={labelCell}>
                  {clean("Key differences")}
                  <div className="mt-0.5 font-normal normal-case text-[10px] text-slate-500">
                    what drives the metric gap
                  </div>
                </div>
                {runs.map((r) => {
                  const has = r.derived_features ?? [];
                  const extras = has.filter((f) => diffFeatures.has(f));
                  const lacks = [...diffFeatures].filter(
                    (f) => !has.includes(f),
                  );
                  return (
                    <div key={r.run_id} className={cell}>
                      <div className="flex flex-wrap gap-1">
                        {extras.map((f) => (
                          <span
                            key={f}
                            title={`${f} — ${featureInfo(f)} (this model added it; not all did)`}
                            className="cursor-help rounded-md bg-status-complete/15 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-status-complete"
                          >
                            + {f}
                          </span>
                        ))}
                        {lacks.map((f) => (
                          <span
                            key={f}
                            title={`${f} — ${featureInfo(f)} (other models used it; this one didn't)`}
                            className="cursor-help rounded-md bg-surface px-1.5 py-0.5 font-mono text-[11px] text-slate-500 line-through"
                          >
                            {f}
                          </span>
                        ))}
                        {extras.length === 0 && lacks.length === 0 && (
                          <span className="text-xs text-slate-500">
                            same as others
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </>
            )}

            {/* Missing handled */}
            <div className={labelCell}>{clean("Handled missing")}</div>
            {runs.map((r) => (
              <div key={r.run_id} className={`${cell} text-slate-300`}>
                {r.missing_summary ?? "—"}
              </div>
            ))}

            {/* Execution */}
            <div
              className="border-b border-surface-border/50 bg-surface/60 px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500"
              style={{ gridColumn: `1 / span ${runs.length + 1}` }}
            >
              Execution
            </div>
            <div className={labelCell}>{clean("Duration")}</div>
            {runs.map((r) => {
              const min = Math.min(
                ...runs
                  .map((x) => x.duration_ms ?? Infinity)
                  .filter((n) => Number.isFinite(n)),
              );
              const isFastest =
                r.duration_ms != null && r.duration_ms === min;
              return (
                <div
                  key={r.run_id}
                  className={`${cell} tabular-nums ${
                    isFastest ? "font-bold text-status-complete" : ""
                  }`}
                >
                  {r.duration_ms != null
                    ? formatDuration(r.duration_ms)
                    : r.status === "running"
                      ? "running…"
                      : "—"}
                </div>
              );
            })}
            <div className={labelCell}>{clean("Started")}</div>
            {runs.map((r) => (
              <div key={r.run_id} className={`${cell} text-slate-300`}>
                {fmtWhen(r.started_at)}
              </div>
            ))}
            <div className={labelCell}>{clean("Ended")}</div>
            {runs.map((r) => (
              <div key={r.run_id} className={`${cell} text-slate-300`}>
                {r.status === "running" ? "—" : fmtWhen(r.ended_at)}
              </div>
            ))}

            {/* Tokens */}
            <div className={labelCell}>{clean("Total tokens")}</div>
            {runs.map((r) => {
              const min = Math.min(
                ...runs
                  .map((x) => x.total_tokens ?? Infinity)
                  .filter((n) => Number.isFinite(n)),
              );
              const isMin = r.total_tokens != null && r.total_tokens === min;
              return (
                <div
                  key={r.run_id}
                  className={`${cell} tabular-nums ${
                    isMin ? "font-bold text-status-complete" : ""
                  }`}
                >
                  {r.total_tokens != null ? r.total_tokens.toLocaleString() : "—"}
                </div>
              );
            })}

            {/* Outcome / halt */}
            <div className={labelCell}>{clean("Outcome")}</div>
            {runs.map((r) => (
              <div key={r.run_id} className={`${cell} text-slate-400`}>
                {r.halt_reason ?? (r.workflow_complete ? "completed" : "—")}
              </div>
            ))}

            {/* Final report download */}
            <div className={`${labelCell} border-b-0`}>{clean("Report")}</div>
            {runs.map((r) => {
              const canGet = r.workflow_complete || r.chosen_model;
              return (
                <div key={r.run_id} className="border-surface-border/50 px-3 py-2">
                  {canGet ? (
                    <button
                      type="button"
                      onClick={() =>
                        openStyledReport(
                          caseId,
                          r.run_id,
                          prettyCase(caseId),
                          r.llm_model,
                        )
                      }
                      className="inline-flex items-center gap-1.5 rounded-lg border border-surface-border px-3 py-1.5 text-xs font-semibold text-accent transition-colors hover:bg-surface-border focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
                    >
                      <span aria-hidden>↓</span> Report (PDF)
                    </button>
                  ) : (
                    <span className="text-xs text-slate-500">not ready</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
