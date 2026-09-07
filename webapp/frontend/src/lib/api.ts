const TOKEN_STORAGE_KEY = "maads.accessToken";

export type ModelCatalog = Record<string, Array<{ id: string; label: string }>>;

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

export function register(email: string, password: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function login(email: string, password: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
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

export function fetchModelCatalog(): Promise<ModelCatalog> {
  return apiFetch<ModelCatalog>("/api/models");
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
