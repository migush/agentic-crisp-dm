import { BASE } from "../shared/api";
import { useSelectedRun } from "../shared/selectedRun";

interface Props {
  caseId: string;
  show: boolean;
  className?: string;
}

export function HandoffDownloadLink({ caseId, show, className = "" }: Props) {
  const { runId } = useSelectedRun();
  if (!show) {
    return null;
  }

  // A plain download anchor, so no auth header is possible — the hosted
  // deployment authenticates it via the session cookie.
  const href = `${BASE}/api/cases/${caseId}/reports/handoff_standard.zip${
    runId ? `?run_id=${encodeURIComponent(runId)}` : ""
  }`;

  return (
    <div className={className}>
      <a
        href={href}
        download
        className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-fuchsia-500 to-pink-500 px-4 py-2.5 text-sm font-semibold text-white shadow-md transition-all hover:scale-[1.03] hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-accent/50"
      >
        <span aria-hidden className="text-base leading-none">
          ↓
        </span>
        Download Handoff Package
      </a>
      <p className="text-xs text-slate-500 mt-2">
        Portable bundle for an external data scientist — data, reports &amp; notebook.
      </p>
    </div>
  );
}
