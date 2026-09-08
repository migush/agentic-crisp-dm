import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { HelpStepper, CASE_UPLOAD_HELP } from "../components/HelpStepper";
import {
  CaseDetail,
  CaseListItem,
  createCase,
  deleteCase,
  getCase,
  InspectReport,
  listCases,
  updateCase,
} from "../lib/api";

const PROBLEM_TYPES = [
  { id: "classification", label: "Classification (any number of labels)" },
  { id: "regression", label: "Regression (numeric target)" },
  { id: "clustering", label: "Clustering / segmentation (not runnable yet)" },
  { id: "association", label: "Association / dependency (not runnable yet)" },
  { id: "image", label: "Images or audio (not runnable yet)" },
];

function statusClass(status: string): string {
  if (status === "ready") return "text-emerald-400";
  if (status === "unsupported") return "text-amber-400";
  return "text-slate-400";
}

function InspectTables({ inspect }: { inspect: InspectReport }) {
  return (
    <div className="space-y-4">
      {!inspect.runnable && inspect.unsupported_reason && (
        <p className="rounded border border-amber-700 bg-amber-950/40 p-3 text-sm text-amber-200">
          {inspect.unsupported_reason}
        </p>
      )}
      {inspect.files.map((file) => (
        <section key={file.filename} className="rounded border border-slate-800 p-3">
          <h3 className="text-sm font-medium text-slate-200">
            {file.original_filename}{" "}
            <span className="text-slate-500">
              ({file.kind}
              {file.encoding ? `, ${file.encoding}` : ""}
              {file.delimiter ? `, delimiter ${JSON.stringify(file.delimiter)}` : ""})
            </span>
          </h3>
          {file.kind === "csv" && (
            <p className="mt-1 text-xs text-slate-400">
              {file.n_rows} rows × {file.n_cols} columns
            </p>
          )}
          {file.parse_warnings.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-xs text-amber-300">
              {file.parse_warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
          {file.columns.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-1 pr-3">Column</th>
                    <th className="py-1 pr-3">Type</th>
                    <th className="py-1 pr-3">Missing</th>
                    <th className="py-1 pr-3">Unique</th>
                    <th className="py-1">Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {file.columns.map((col) => (
                    <tr key={col.name} className="border-b border-slate-900">
                      <td className="py-1 pr-3 font-mono">{col.name}</td>
                      <td className="py-1 pr-3">{col.dtype}</td>
                      <td className="py-1 pr-3">{col.n_missing}</td>
                      <td className="py-1 pr-3">{col.n_unique}</td>
                      <td className="py-1 text-slate-500">{col.sample_values.join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ))}
    </div>
  );
}

export function CasesPage() {
  const [cases, setCases] = useState<CaseListItem[]>([]);
  const [mode, setMode] = useState<"list" | "new" | "review">("list");
  const [step, setStep] = useState(0);
  const [title, setTitle] = useState("");
  const [problemStatement, setProblemStatement] = useState("");
  const [csvFiles, setCsvFiles] = useState<FileList | null>(null);
  const [document, setDocument] = useState<File | null>(null);
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [problemType, setProblemType] = useState("classification");
  const [targetColumn, setTargetColumn] = useState("");
  const [idColumn, setIdColumn] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function refreshList() {
    listCases().then((items) => setCases(items.filter((c) => c.kind === "user")));
  }

  useEffect(() => {
    refreshList();
  }, []);

  function resetWizard() {
    setMode("list");
    setStep(0);
    setTitle("");
    setProblemStatement("");
    setCsvFiles(null);
    setDocument(null);
    setDetail(null);
    setProblemType("classification");
    setTargetColumn("");
    setIdColumn("");
    setStatus(null);
  }

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    if (!csvFiles || csvFiles.length === 0) {
      setStatus("Upload at least one CSV.");
      return;
    }
    const form = new FormData();
    form.append("title", title);
    form.append("problem_statement", problemStatement);
    form.append("problem_type", problemType);
    for (const file of Array.from(csvFiles)) {
      form.append("files", file);
    }
    if (document) {
      form.append("document", document);
    }
    setBusy(true);
    try {
      const created = await createCase(form);
      setDetail(created);
      setTargetColumn(created.target_column || "");
      setIdColumn(created.id_column || "");
      setProblemType(created.problem_type || "classification");
      setMode("review");
      refreshList();
    } catch (err) {
      setStatus((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function openCase(caseId: string) {
    setStatus(null);
    const loaded = await getCase(caseId);
    setDetail(loaded);
    setTitle(loaded.display_name);
    setProblemStatement(loaded.problem_statement || "");
    setTargetColumn(loaded.target_column || "");
    setIdColumn(loaded.id_column || "");
    setProblemType(loaded.problem_type || "classification");
    setMode("review");
  }

  async function onConfirm(markReady: boolean) {
    if (!detail) return;
    setBusy(true);
    setStatus(null);
    try {
      const updated = await updateCase(detail.case_id, {
        display_name: title || detail.display_name,
        problem_statement: problemStatement,
        problem_type: problemType,
        target_column: targetColumn,
        id_column: idColumn,
        mark_ready: markReady,
      });
      setDetail(updated);
      refreshList();
      if (updated.status === "ready") {
        setStatus("Case is ready. Launch it from Tasks with any stored provider and model.");
      } else if (updated.status === "unsupported") {
        setStatus(updated.unsupported_reason || "Not runnable yet in V1.");
      } else {
        setStatus("Draft saved.");
      }
    } catch (err) {
      setStatus((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function onDelete(caseId: string) {
    if (!window.confirm(`Delete case ${caseId}? This cannot be undone.`)) return;
    await deleteCase(caseId);
    resetWizard();
    refreshList();
  }

  const userCases = cases;

  return (
    <div className="mx-auto mt-16 max-w-3xl space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Cases</h1>
        {mode !== "list" && (
          <button className="text-sm text-slate-400" onClick={resetWizard} type="button">
            Back to list
          </button>
        )}
      </div>

      {mode === "list" && (
        <>
          <p className="text-sm text-slate-400">
            Describe a problem, upload raw CSVs, and let MAADS agents understand and prepare the data
            during the CRISP-DM run. Ready cases appear in{" "}
            <Link className="text-sky-400" to="/tasks">
              Tasks
            </Link>
            .
          </p>
          <HelpStepper title="What should I provide?" steps={CASE_UPLOAD_HELP} />
          <button
            className="rounded bg-sky-600 px-4 py-2 text-sm font-medium hover:bg-sky-500"
            type="button"
            onClick={() => {
              setMode("new");
              setStep(0);
            }}
          >
            New case
          </button>
          <div className="space-y-2">
            {userCases.length === 0 && <p className="text-sm text-slate-500">No cases yet.</p>}
            {userCases.map((c) => (
              <div key={c.case_id} className="flex items-center justify-between rounded border border-slate-800 p-3 text-sm">
                <button className="text-left" type="button" onClick={() => openCase(c.case_id)}>
                  <span className="font-medium">{c.display_name}</span>{" "}
                  <span className="text-slate-500">({c.case_id})</span>{" "}
                  <span className={`uppercase ${statusClass(c.status)}`}>{c.status}</span>
                </button>
                <button className="text-slate-500 hover:text-red-400" type="button" onClick={() => onDelete(c.case_id)}>
                  Delete
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      {mode === "new" && (
        <div className="space-y-6">
          {step === 0 && (
            <div className="space-y-4">
              <HelpStepper title="What to provide" steps={CASE_UPLOAD_HELP} />
              <button
                className="rounded bg-sky-600 px-4 py-2 text-sm font-medium hover:bg-sky-500"
                type="button"
                onClick={() => setStep(1)}
              >
                Continue
              </button>
            </div>
          )}
          {step === 1 && (
            <form
              className="space-y-3"
              onSubmit={(e) => {
                e.preventDefault();
                setStep(2);
              }}
            >
              <label className="block text-sm">
                Title
                <input
                  className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                />
              </label>
              <label className="block text-sm">
                Problem statement
                <textarea
                  className="mt-1 min-h-[8rem] w-full rounded border border-slate-700 bg-slate-900 p-2"
                  value={problemStatement}
                  onChange={(e) => setProblemStatement(e.target.value)}
                  placeholder="What should be predicted, what one row represents, and why it matters."
                  required
                />
              </label>
              <label className="block text-sm">
                Optional PDF or markdown
                <input
                  className="mt-1 w-full text-sm"
                  type="file"
                  accept=".pdf,.md,.markdown,.txt"
                  onChange={(e) => setDocument(e.target.files?.[0] ?? null)}
                />
              </label>
              <button className="w-full rounded bg-sky-600 p-2 font-medium hover:bg-sky-500" type="submit">
                Continue to upload
              </button>
            </form>
          )}
          {step === 2 && (
            <form className="space-y-3" onSubmit={onCreate}>
              <label className="block text-sm">
                CSV files
                <input
                  className="mt-1 w-full text-sm"
                  type="file"
                  accept=".csv,.tsv,.txt,text/csv"
                  multiple
                  onChange={(e) => setCsvFiles(e.target.files)}
                  required
                />
              </label>
              <p className="text-xs text-slate-500">
                Per-file size is capped at 50 MB. Messy encodings and missing test/submission files are allowed.
              </p>
              {status && <p className="text-sm text-slate-300">{status}</p>}
              <button
                className="w-full rounded bg-sky-600 p-2 font-medium hover:bg-sky-500 disabled:opacity-50"
                type="submit"
                disabled={busy}
              >
                {busy ? "Profiling…" : "Upload and inspect"}
              </button>
            </form>
          )}
        </div>
      )}

      {mode === "review" && detail && (
        <div className="space-y-6">
          <h2 className="text-lg font-medium">
            {detail.display_name}{" "}
            <span className={`text-sm uppercase ${statusClass(detail.status)}`}>{detail.status}</span>
          </h2>
          <InspectTables inspect={detail.inspect} />
          <form
            className="space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              void onConfirm(true);
            }}
          >
            <label className="block text-sm">
              Problem type
              <select
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
                value={problemType}
                onChange={(e) => setProblemType(e.target.value)}
              >
                {PROBLEM_TYPES.map((pt) => (
                  <option key={pt.id} value={pt.id}>
                    {pt.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              Suspected target column (optional)
              <input
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                placeholder="Leave blank for agents to decide"
              />
            </label>
            <label className="block text-sm">
              Suspected ID column (optional)
              <input
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
                value={idColumn}
                onChange={(e) => setIdColumn(e.target.value)}
              />
            </label>
            <label className="block text-sm">
              Problem statement
              <textarea
                className="mt-1 min-h-[6rem] w-full rounded border border-slate-700 bg-slate-900 p-2"
                value={problemStatement}
                onChange={(e) => setProblemStatement(e.target.value)}
              />
            </label>
            {status && <p className="text-sm text-slate-300">{status}</p>}
            {["clustering", "association", "image"].includes(problemType) && (
              <p className="text-sm text-amber-300">This problem type is not runnable yet in V1.</p>
            )}
            <div className="flex gap-3">
              <button
                className="flex-1 rounded border border-slate-700 p-2 text-sm"
                type="button"
                disabled={busy}
                onClick={() => void onConfirm(false)}
              >
                Save draft
              </button>
              <button
                className="flex-1 rounded bg-sky-600 p-2 text-sm font-medium hover:bg-sky-500 disabled:opacity-50"
                type="submit"
                disabled={busy || ["clustering", "association", "image"].includes(problemType)}
              >
                Mark ready to run
              </button>
            </div>
          </form>
          {detail.status === "ready" && (
            <p className="text-sm text-slate-400">
              Launch from{" "}
              <Link className="text-sky-400" to="/tasks">
                Tasks
              </Link>{" "}
              with any stored provider and model.
            </p>
          )}
        </div>
      )}
    </div>
  );
}