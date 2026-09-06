import { useEffect, useMemo, useState } from "react";
import type { TabId } from "./shared/types";
import { BIZ_THEME_NAME, useTheme } from "./shared/theme";
import type { ThemeId } from "./shared/theme";
import { useCaseRuns, useCases } from "./hooks/useCasePolling";
import { useSelectedRun } from "./shared/selectedRun";
import { prettyCase } from "./shared/format";
import { Overview } from "./pages/Overview";
import { Process } from "./pages/Process";
import { Inspect } from "./pages/Inspect";
import { Architecture } from "./pages/Architecture";
import { Framework } from "./pages/Framework";
import { Prompts } from "./pages/Prompts";
import { StateShape } from "./pages/StateShape";
import { FailureModes } from "./pages/FailureModes";
import { Launch } from "./pages/Launch";
import { Home } from "./pages/Home";
import { Results } from "./pages/Results";
import { MaadsLogo } from "./components/MaadsLogo";

// `needsCase` tabs are disabled until a case is selected, so the user never
// lands on a "select a case" dead end.
// Launch & Overview are reachable from the Home page's buttons, so they're
// kept out of the top nav to keep it lean. State + Communications are merged
// into a single "Inspect" tab with an internal switch.
const TABS: { id: TabId; label: string; needsCase?: boolean }[] = [
  { id: "home", label: "🏠 Home" },
  { id: "architecture", label: "🦋 Architecture", needsCase: true },
  { id: "state_shape", label: "🏗️ State Shape" },
  { id: "process", label: "🌸 Process", needsCase: true },
  { id: "inspect", label: "🔎 Inspect", needsCase: true },
  { id: "results", label: "📊 Results" },
];

// Tabs whose content depends on the selected case/run. The case & run pickers
// are only shown on these; static pages (Home, State Shape, …) hide them.
const CASE_TABS: TabId[] = ["overview", "process", "inspect", "architecture"];

function ThemeToggle({
  theme,
  onChange,
}: {
  theme: ThemeId;
  onChange: (t: ThemeId) => void;
}) {
  const base =
    "rounded-full px-3 py-1 transition-all whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50";
  const active =
    "bg-gradient-to-r from-fuchsia-500 to-pink-500 text-white shadow";
  const idle = "text-slate-400 hover:text-slate-200";
  return (
    <div
      role="group"
      aria-label="Theme"
      title="Switch the look of the dashboard"
      className="flex items-center gap-1 rounded-full border border-surface-border bg-surface p-1 text-xs font-semibold shadow-sm"
    >
      <button
        type="button"
        onClick={() => onChange("biz")}
        aria-pressed={theme === "biz"}
        className={`${base} ${theme === "biz" ? active : idle}`}
      >
        {BIZ_THEME_NAME}
      </button>
      <button
        type="button"
        onClick={() => onChange("pink")}
        aria-pressed={theme === "pink"}
        className={`${base} ${theme === "pink" ? active : idle}`}
      >
        {theme === "biz" ? "Pink" : "🌸 Pink"}
      </button>
    </div>
  );
}

export default function App() {
  const { data: cases, isLoading, error } = useCases();
  const [caseId, setCaseId] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("home");
  const [promptAgent, setPromptAgent] = useState<string | undefined>();
  const { theme, setTheme, clean } = useTheme();
  const { runId, setRunId } = useSelectedRun();
  const { data: runs } = useCaseRuns(caseId);

  const openPrompt = (agentId: string) => {
    setPromptAgent(agentId);
    setTab("prompts");
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get("case");
    if (fromUrl) setCaseId(fromUrl);
  }, []);

  useEffect(() => {
    if (!caseId && cases?.length) {
      setCaseId(cases[0].case_id);
    }
  }, [cases, caseId]);

  // When the case changes, follow its active run until the user picks one.
  useEffect(() => {
    setRunId(null);
  }, [caseId, setRunId]);

  const activeCase = useMemo(
    () => cases?.find((c) => c.case_id === caseId),
    [cases, caseId],
  );
  const selectedRun = useMemo(
    () => runs?.find((r) => r.run_id === runId),
    [runs, runId],
  );
  const shownModel = selectedRun?.model ?? activeCase?.model ?? null;
  const showRunControls = CASE_TABS.includes(tab);

  if (error) {
    return (
      <div className="p-8 text-center">
        <p className="text-status-halted font-semibold">
          {clean("😢 Cannot reach API — is the dashboard server running?")}
        </p>
        <p className="text-sm text-slate-500 mt-2">
          Run: <code className="text-accent-muted">python -m maads dashboard --no-open</code>
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-surface-border bg-surface-raised/80 backdrop-blur px-6 py-4 shadow-sm">
        <div className="flex flex-wrap items-center gap-4 justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <MaadsLogo className="h-8 w-8 shrink-0 text-accent" />
            <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-fuchsia-500 via-pink-500 to-purple-500 bg-clip-text text-transparent">
              {clean("🌸 MAADS Trace ✨")}
            </h1>
          </div>

          <div className="flex items-center gap-2">
            {showRunControls && (
              <>
            <select
              value={caseId ?? ""}
              onChange={(e) => setCaseId(e.target.value)}
              disabled={isLoading}
              className="rounded-full border border-surface-border bg-surface px-4 py-1.5 text-sm font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-accent/40"
            >
              {!cases?.length && (
                <option value="">{clean("No cases 😴")}</option>
              )}
              {cases?.map((c) => (
                <option key={c.case_id} value={c.case_id}>
                  {clean(`🎀 ${prettyCase(c.case_id)} (${c.status})`)}
                </option>
              ))}
            </select>

            {/* Run picker — hovering it reveals which LLM that run used. */}
            <div className="group relative">
              <select
                value={runId ?? ""}
                onChange={(e) => setRunId(e.target.value || null)}
                disabled={!runs?.length}
                title="Pick which run of this case to view"
                className="rounded-full border border-surface-border bg-surface px-4 py-1.5 text-sm font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-accent/40 max-w-[16rem]"
              >
                <option value="">{clean("⚡ active run")}</option>
                {runs?.map((r, i) => (
                  <option key={r.run_id} value={r.run_id}>
                    {`#${(runs.length - i)} · ${r.model ?? "default"} · ${r.status}`}
                  </option>
                ))}
              </select>
              <span
                className="pointer-events-none absolute left-0 top-full z-20 mt-1 whitespace-nowrap rounded-full border border-surface-border bg-surface px-3 py-0.5 text-xs font-mono text-slate-400 opacity-0 shadow-sm transition-opacity duration-200 group-hover:opacity-100"
                title="LLM this run was launched with"
              >
                {clean("🧠 ")}{shownModel ?? "default (.env)"}
              </span>
            </div>
              </>
            )}

            <ThemeToggle theme={theme} onChange={setTheme} />
          </div>
        </div>

        <nav className="flex gap-2 mt-4 max-w-7xl mx-auto overflow-x-auto no-scrollbar -mx-1 px-1">
          {TABS.map((t) => {
            const locked = Boolean(t.needsCase) && !caseId;
            return (
              <button
                key={t.id}
                type="button"
                disabled={locked}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}
                title={locked ? "Select a case first" : undefined}
                className={`shrink-0 whitespace-nowrap rounded-full px-4 py-2 text-sm font-semibold transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
                  tab === t.id
                    ? "bg-gradient-to-r from-fuchsia-500 to-pink-500 text-white shadow-md scale-105"
                    : locked
                    ? "text-slate-600 cursor-not-allowed"
                    : "text-slate-400 hover:bg-surface-border hover:text-slate-200"
                }`}
              >
                {clean(t.label)}
              </button>
            );
          })}
        </nav>
      </header>

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        {tab === "home" ? (
          <Home
            onLaunch={() => setTab("launch")}
            onExplore={() => setTab(caseId ? "overview" : "launch")}
            datasetCount={cases?.length}
            experimentCount={cases?.reduce(
              (sum, c) => sum + (c.run_count ?? 0),
              0,
            )}
          />
        ) : tab === "launch" ? (
          <Launch onLaunched={(id) => setCaseId(id)} />
        ) : tab === "results" ? (
          <Results />
        ) : tab === "framework" ? (
          <Framework />
        ) : tab === "prompts" ? (
          <Prompts initialAgent={promptAgent} />
        ) : tab === "state_shape" ? (
          <StateShape />
        ) : tab === "failure_modes" ? (
          <FailureModes />
        ) : !caseId ? (
          <p className="text-slate-500">
            {clean("💫 Select a case or start one from the Launch tab!")}
          </p>
        ) : (
          <>
            {tab === "overview" && <Overview caseId={caseId} />}
            {tab === "process" && <Process caseId={caseId} />}
            {tab === "inspect" && <Inspect caseId={caseId} />}
            {tab === "architecture" && (
              <Architecture caseId={caseId} onOpenPrompt={openPrompt} />
            )}
          </>
        )}
      </main>
    </div>
  );
}
