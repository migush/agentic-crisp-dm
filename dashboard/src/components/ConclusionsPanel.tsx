import type { ReactNode } from "react";
import { BASE } from "../shared/api";
import type {
  ConclusionItem,
  ConclusionPhase,
  ProcessConclusions,
  ProcessDeliverable,
} from "../shared/types";

function deliverableHref(url: string): string {
  return url.startsWith("/") ? `${BASE}${url}` : url;
}

interface Props {
  conclusions: ProcessConclusions | undefined;
  deliverables: ProcessDeliverable[] | undefined;
  config: { problem_statement?: string; evaluation_metric?: string; target_column?: string } | undefined;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-slate-300">{title}</h3>
      <div className="text-sm text-slate-400">{children}</div>
    </div>
  );
}

function ConclusionItemCard({ item }: { item: ConclusionItem }) {
  return (
    <div className="rounded-lg border border-surface-border/60 bg-surface/40 p-3 space-y-1.5">
      <p className="text-xs font-mono text-accent-muted">
        §{item.id} — {item.name}
      </p>
      <p className="text-slate-300">{item.summary}</p>
      {(item.highlights?.length ?? 0) > 0 && (
        <ul className="text-xs space-y-0.5 pt-1">
          {item.highlights!.map((h) => (
            <li key={h.label}>
              <span className="text-slate-500">{h.label}: </span>
              <span className="text-slate-400">{h.value}</span>
            </li>
          ))}
        </ul>
      )}
      {item.artifact_paths && Object.keys(item.artifact_paths).length > 0 && (
        <ul className="text-xs font-mono space-y-0.5 pt-1">
          {Object.entries(item.artifact_paths).map(([role, path]) => (
            <li key={role} className="text-slate-500 truncate">
              {role}: <span className="text-slate-400">{path}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PhaseConclusions({ phase }: { phase: ConclusionPhase }) {
  return (
    <Section title={phase.name}>
      <div className="space-y-2">
        {phase.items.map((item) => (
          <ConclusionItemCard key={item.id} item={item} />
        ))}
      </div>
    </Section>
  );
}

export function ConclusionsPanel({ conclusions, deliverables, config }: Props) {
  if (!conclusions) {
    return <p className="text-sm text-slate-500">Waiting for process data…</p>;
  }

  const phaseItems = conclusions.phases ?? [];
  const hasPhases = phaseItems.some((p) => p.items.length > 0);
  const hasData =
    (conclusions.data_quality_blockers?.length ?? 0) > 0 ||
    (conclusions.data_quality_tolerable?.length ?? 0) > 0 ||
    Object.keys(conclusions.dataset_paths ?? {}).length > 0 ||
    !!conclusions.dataset_description;
  const hasModels = (conclusions.models?.length ?? 0) > 0;
  const hasEval = conclusions.assessment || conclusions.decision;
  const hasDeploy =
    conclusions.submission_path || conclusions.final_report_path;

  if (
    !hasPhases &&
    !hasData &&
    !hasModels &&
    !hasEval &&
    !hasDeploy &&
    !config?.problem_statement
  ) {
    return (
      <p className="text-sm text-slate-500">
        No conclusions recorded yet — agents are still working through early phases.
      </p>
    );
  }

  return (
    <div className="space-y-5">
      {config?.problem_statement && (
        <Section title="Problem">
          <p>{config.problem_statement}</p>
          {config.evaluation_metric && (
            <p className="text-xs mt-1 text-slate-500">
              Metric: {config.evaluation_metric}
              {config.target_column && ` · Target: ${config.target_column}`}
            </p>
          )}
        </Section>
      )}

      {phaseItems.map((phase) => (
        <PhaseConclusions key={phase.id} phase={phase} />
      ))}

      {hasData && !phaseItems.some((p) => p.id === 2 || p.id === 3) && (
        <Section title="Data">
          {(conclusions.data_quality_blockers?.length ?? 0) > 0 && (
            <ul className="list-disc list-inside text-amber-400/90">
              {conclusions.data_quality_blockers!.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          )}
          {(conclusions.data_quality_tolerable?.length ?? 0) > 0 && (
            <ul className="list-disc list-inside text-slate-400 mt-1">
              {conclusions.data_quality_tolerable!.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ul>
          )}
          {conclusions.dataset_description && (
            <p className="mt-1">{conclusions.dataset_description}</p>
          )}
          {Object.entries(conclusions.dataset_paths ?? {}).map(([role, path]) => (
            <p key={role} className="font-mono text-xs mt-1">
              {role}: {path}
            </p>
          ))}
        </Section>
      )}

      {hasModels &&
        !phaseItems.some((p) => p.id === 4 && p.items.some((i) => i.id === "4.3")) && (
        <Section title="Modeling">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 text-left">
                <th className="pb-1">Technique</th>
                <th className="pb-1">CV</th>
                <th className="pb-1">Assessment</th>
              </tr>
            </thead>
            <tbody>
              {conclusions.models!.map((m, i) => (
                <tr key={i} className="border-t border-surface-border">
                  <td className="py-1">{m.technique}</td>
                  <td className="py-1 font-mono">
                    {m.cv_score != null ? m.cv_score.toFixed(4) : "—"}
                  </td>
                  <td className="py-1 text-slate-500">{m.assessment ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {conclusions.chosen_model && (
            <p className="mt-2 text-accent-muted">
              Chosen: {conclusions.chosen_model.technique}
              {conclusions.chosen_model.cv_score != null &&
                ` (CV ${conclusions.chosen_model.cv_score.toFixed(4)})`}
            </p>
          )}
        </Section>
      )}

      {hasEval &&
        !phaseItems.some((p) => p.id === 5 && p.items.some((i) => i.id === "5.1")) && (
        <Section title="Evaluation">
          {conclusions.assessment && (
            <p>
              CV {conclusions.assessment.cv_score?.toFixed(4) ?? "?"}{" "}
              {conclusions.assessment.meets ? (
                <span className="text-emerald-400">meets threshold</span>
              ) : (
                <span className="text-amber-400">below threshold</span>
              )}
              {conclusions.assessment.threshold != null &&
                ` (${conclusions.assessment.threshold})`}
            </p>
          )}
          {conclusions.decision && (
            <p className="mt-1">
              <span className="text-slate-500">Decision: </span>
              {conclusions.decision}
            </p>
          )}
        </Section>
      )}

      {(hasDeploy || (deliverables?.length ?? 0) > 0) && (
        <Section title="Deployment">
          <ul className="space-y-1">
            {deliverables?.map((d) => (
              <li key={d.path} className="font-mono text-xs flex items-center gap-2 flex-wrap">
                <span className={d.exists ? "text-emerald-400" : "text-slate-600"}>
                  {d.exists ? "●" : "○"}
                </span>
                <span className="text-slate-500">{d.label}:</span>
                {d.url ? (
                  <a
                    href={deliverableHref(d.url)}
                    className="text-accent hover:underline truncate"
                    download
                  >
                    {d.label === "Case workbook"
                      ? "Open notebook"
                      : d.label === "Standard handoff"
                        ? "handoff_standard.zip"
                        : d.path}
                  </a>
                ) : (
                  <span className="text-slate-400 truncate">{d.path}</span>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  );
}
