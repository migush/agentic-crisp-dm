import type {
  CaseConfig,
  CaseSummary,
  CommunicationRecord,
  CommunicationsSummary,
  CrispDMStatePayload,
  GraphPayload,
  LiveSummary,
  ModelCatalog,
  ModelInfo,
  ProcessView,
  RagView,
  RunResult,
  RunSummary,
  StatusPayload,
  TraceSummary,
} from "./types";

/**
 * Path prefix this dashboard is served under.
 *
 * Two deployments share one build: `python -m maads dashboard` serves it from
 * `/`, and the hosted account app mounts it at `/dashboard/` (DASHBOARD_MOUNT
 * in webapp/backend/app.py). Resolved at runtime rather than baked in at build
 * time so `npm run build` doesn't have to be told which one it's for.
 */
export const MOUNT_PREFIX = "/dashboard";

export const BASE = window.location.pathname.startsWith(MOUNT_PREFIX) ? MOUNT_PREFIX : "";

/** True when running inside the hosted account app (per-user, authenticated). */
export const isHosted = BASE !== "";

const API = `${BASE}/api`;

/** Session token shared with the account app (same origin, same localStorage). */
const TOKEN_STORAGE_KEY = "maads_access_token";

export function authHeaders(): Record<string, string> {
  const token = localStorage.getItem(TOKEN_STORAGE_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** A 401 only happens in the hosted deployment; locally there is no auth. */
function redirectToLogin(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  window.location.href = "/login";
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: authHeaders() });
  if (res.status === 401) {
    redirectToLogin();
    throw new Error("Not authenticated");
  }
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/** Append a run_id query param (and optional extras) to a case-scoped path. */
function withRun(
  path: string,
  runId?: string | null,
  extra?: Record<string, string>,
): string {
  const params = new URLSearchParams(extra);
  if (runId) params.set("run_id", runId);
  const qs = params.toString();
  return `${API}${path}${qs ? `?${qs}` : ""}`;
}

export function fetchCases(): Promise<CaseSummary[]> {
  return fetchJson(`${API}/cases`);
}

export function fetchCaseRuns(caseId: string): Promise<RunSummary[]> {
  return fetchJson(`${API}/cases/${encodeURIComponent(caseId)}/runs`);
}

export function fetchCaseResults(caseId: string): Promise<RunResult[]> {
  return fetchJson(`${API}/cases/${encodeURIComponent(caseId)}/results`);
}

export function fetchLiveSummary(caseId: string, runId?: string | null): Promise<LiveSummary> {
  return fetchJson(withRun(`/cases/${encodeURIComponent(caseId)}/live_summary`, runId));
}

export function fetchStatus(caseId: string, runId?: string | null): Promise<StatusPayload> {
  return fetchJson(withRun(`/cases/${encodeURIComponent(caseId)}/status`, runId));
}

export function fetchTraceSummary(caseId: string, runId?: string | null): Promise<TraceSummary> {
  return fetchJson(
    withRun(`/cases/${encodeURIComponent(caseId)}/trace/summary`, runId, { limit: "50" }),
  );
}

export function fetchCommunications(
  caseId: string,
  opts?: { sinceId?: string; limit?: number },
  runId?: string | null,
): Promise<CommunicationRecord[]> {
  const extra: Record<string, string> = {};
  if (opts?.sinceId) extra.since_id = opts.sinceId;
  if (opts?.limit) extra.limit = String(opts.limit);
  return fetchJson(
    withRun(`/cases/${encodeURIComponent(caseId)}/communications`, runId, extra),
  );
}

export function fetchCommunicationsSummary(
  caseId: string,
  runId?: string | null,
): Promise<CommunicationsSummary> {
  return fetchJson(
    withRun(`/cases/${encodeURIComponent(caseId)}/communications/summary`, runId),
  );
}

export function fetchGraph(caseId: string, runId?: string | null): Promise<GraphPayload> {
  return fetchJson(withRun(`/cases/${encodeURIComponent(caseId)}/graph`, runId));
}

export function fetchProcess(caseId: string, runId?: string | null): Promise<ProcessView> {
  return fetchJson(withRun(`/cases/${encodeURIComponent(caseId)}/process`, runId));
}

export function fetchState(caseId: string, runId?: string | null): Promise<CrispDMStatePayload> {
  return fetchJson(withRun(`/cases/${encodeURIComponent(caseId)}/state`, runId));
}

export function fetchRag(caseId: string, runId?: string | null): Promise<RagView> {
  return fetchJson(withRun(`/cases/${encodeURIComponent(caseId)}/rag`, runId));
}

export function fetchConfigs(): Promise<CaseConfig[]> {
  return fetchJson(`${API}/configs`);
}

export function fetchModels(): Promise<ModelCatalog> {
  return fetchJson(`${API}/models`);
}

export function fetchModelInfo(model?: string, probe = false): Promise<ModelInfo> {
  const params = new URLSearchParams();
  if (model) params.set("model", model);
  if (probe) params.set("probe", "true");
  const qs = params.toString();
  return fetchJson(`${API}/models/info${qs ? `?${qs}` : ""}`);
}

export async function postStartRun(
  caseId: string,
  model?: string,
): Promise<{ status: string; case_id: string; model: string | null; pid: number }> {
  const res = await fetch(`${API}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ case_id: caseId, model: model ?? null }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}
