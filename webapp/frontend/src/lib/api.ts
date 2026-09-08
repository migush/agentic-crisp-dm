const TOKEN_STORAGE_KEY = "maads.accessToken";

export type LiveModel = { id: string; label: string };

export type StoredKey = {
  provider: string;
  selected_model: string;
  ciphertext_b64: string;
  iv_b64: string;
  kdf_salt_b64: string;
  kdf_params_json: string;
  updated_at: string;
};

export type TaskSummary = {
  id: number;
  case_name: string;
  provider: string;
  model_id: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
};

export type TokenSpendEvent = {
  agent: string;
  provider: string;
  model_id: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

type TokenResponse = {
  access_token: string;
  token_type: string;
};

type StoreKeyRequest = Omit<StoredKey, "updated_at">;

type LaunchTaskRequest = {
  case_name: string;
  provider: string;
  model_id: string;
  decrypted_api_key: string;
};

export function getToken(): string | null {
  return window.localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearToken(): void {
  window.localStorage.removeItem(TOKEN_STORAGE_KEY);
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) {
        message = Array.isArray(body.detail) ? body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join(", ") : String(body.detail);
      }
    } catch {
      // Keep the HTTP status fallback when the server did not return JSON.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function register(username: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

export function login(username: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

export async function logout(): Promise<void> {
  clearToken();
  try {
    await apiFetch<void>("/api/auth/logout", { method: "POST" });
  } catch {
    // Local logout must succeed even when the cookie-clearing request fails.
  }
}

export function fetchLiveModels(provider: string, decryptedApiKey: string): Promise<LiveModel[]> {
  return apiFetch<LiveModel[]>("/api/models", {
    method: "POST",
    body: JSON.stringify({ provider, decrypted_api_key: decryptedApiKey }),
  });
}

export function listStoredKeys(): Promise<StoredKey[]> {
  return apiFetch<StoredKey[]>("/api/keys");
}

export function upsertStoredKey(body: StoreKeyRequest): Promise<StoredKey> {
  return apiFetch<StoredKey>("/api/keys", {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export function listTasks(): Promise<TaskSummary[]> {
  return apiFetch<TaskSummary[]>("/api/tasks");
}

export function launchTask(body: LaunchTaskRequest): Promise<TaskSummary> {
  return apiFetch<TaskSummary>("/api/tasks", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fetchTaskSpend(taskId: number): Promise<TokenSpendEvent[]> {
  return apiFetch<TokenSpendEvent[]>(`/api/tasks/${taskId}/spend`);
}

export type CaseKind = "demo" | "user";
export type CaseStatus = "draft" | "ready" | "unsupported";

export type CaseListItem = {
  kind: CaseKind;
  case_id: string;
  display_name: string;
  status: CaseStatus;
  updated_at: string | null;
  problem_type: string | null;
};

export type ColumnObservation = {
  name: string;
  dtype: string;
  n_missing: number;
  n_unique: number;
  sample_values: string[];
};

export type FileObservation = {
  filename: string;
  original_filename: string;
  encoding: string;
  delimiter: string;
  n_rows: number;
  n_cols: number;
  columns: ColumnObservation[];
  parse_warnings: string[];
  kind: "csv" | "document" | "unsupported";
};

export type InspectReport = {
  files: FileObservation[];
  runnable: boolean;
  unsupported_reason: string | null;
};

export type CaseDetail = CaseListItem & {
  problem_statement: string;
  target_column: string;
  id_column: string;
  evaluation_metric: string;
  inspect: InspectReport;
  file_roles: Record<string, string>;
  unsupported_reason?: string;
};

async function apiForm<T>(path: string, form: FormData, method = "POST"): Promise<T> {
  const headers = new Headers();
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(path, { method, headers, body: form });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) {
        message = Array.isArray(body.detail)
          ? body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join(", ")
          : String(body.detail);
      }
    } catch {
      // Keep the HTTP status fallback when the server did not return JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function listCases(): Promise<CaseListItem[]> {
  return apiFetch<CaseListItem[]>("/api/cases");
}

export function getCase(caseId: string): Promise<CaseDetail> {
  return apiFetch<CaseDetail>(`/api/cases/${caseId}`);
}

export function createCase(form: FormData): Promise<CaseDetail> {
  return apiForm<CaseDetail>("/api/cases", form);
}

export function updateCase(
  caseId: string,
  body: {
    display_name?: string;
    problem_statement?: string;
    problem_type?: string;
    target_column?: string;
    id_column?: string;
    evaluation_metric?: string;
    file_roles?: Record<string, string>;
    mark_ready?: boolean;
  },
): Promise<CaseDetail> {
  return apiFetch<CaseDetail>(`/api/cases/${caseId}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export function inspectCase(caseId: string, form?: FormData): Promise<{ case_id: string; inspect: InspectReport }> {
  if (form) {
    return apiForm<{ case_id: string; inspect: InspectReport }>(`/api/cases/${caseId}/inspect`, form);
  }
  return apiFetch<{ case_id: string; inspect: InspectReport }>(`/api/cases/${caseId}/inspect`, { method: "POST" });
}

export function deleteCase(caseId: string): Promise<void> {
  return apiFetch<void>(`/api/cases/${caseId}`, { method: "DELETE" });
}
