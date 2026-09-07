import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { decryptApiKey } from "../lib/crypto";
import {
  fetchLiveModels,
  fetchTaskSpend,
  launchTask,
  listStoredKeys,
  listTasks,
  LiveModel,
  StoredKey,
  TaskSummary,
  TokenSpendEvent,
  upsertStoredKey,
} from "../lib/api";

const HOSTED_PROVIDERS = ["openai", "ollama_cloud"] as const;

function providerLabel(provider: string): string {
  return provider === "ollama_cloud" ? "Ollama Cloud" : "OpenAI";
}

function keyBlob(stored: StoredKey, selectedModel: string) {
  return {
    provider: stored.provider,
    selected_model: selectedModel,
    ciphertext_b64: stored.ciphertext_b64,
    iv_b64: stored.iv_b64,
    kdf_salt_b64: stored.kdf_salt_b64,
    kdf_params_json: stored.kdf_params_json,
  };
}

export function TasksPage() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [storedKeys, setStoredKeys] = useState<StoredKey[]>([]);
  const [provider, setProvider] = useState("openai");
  const [caseName, setCaseName] = useState("titanic");
  const [passphrase, setPassphrase] = useState("");
  const [decryptedApiKey, setDecryptedApiKey] = useState<string | null>(null);
  const [models, setModels] = useState<LiveModel[]>([]);
  const [modelId, setModelId] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [spendByTask, setSpendByTask] = useState<Record<number, TokenSpendEvent[]>>({});

  const hostedKeys = useMemo(
    () => storedKeys.filter((k) => (HOSTED_PROVIDERS as readonly string[]).includes(k.provider)),
    [storedKeys],
  );
  const activeKey = hostedKeys.find((k) => k.provider === provider) ?? hostedKeys[0];

  function refreshTasks() {
    listTasks().then(setTasks);
  }

  useEffect(() => {
    refreshTasks();
    listStoredKeys().then((keys) => {
      setStoredKeys(keys);
      const hosted = keys.filter((k) => (HOSTED_PROVIDERS as readonly string[]).includes(k.provider));
      if (hosted.length === 1) {
        setProvider(hosted[0].provider);
      }
    });
  }, []);

  function onProviderChange(next: string) {
    setProvider(next);
    setDecryptedApiKey(null);
    setModels([]);
    setModelId("");
    setStatus(null);
  }

  async function persistSelectedModel(stored: StoredKey, nextModel: string) {
    if (!nextModel || stored.selected_model === nextModel) {
      return;
    }
    const saved = await upsertStoredKey(keyBlob(stored, nextModel));
    setStoredKeys((prev) => [...prev.filter((k) => k.provider !== stored.provider), saved]);
  }

  async function onUnlock(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    if (!activeKey) {
      setStatus("No stored API key — set one up in your profile first.");
      return;
    }
    let decrypted: string;
    try {
      decrypted = await decryptApiKey(activeKey, passphrase);
    } catch {
      setStatus("Couldn't decrypt your key — check your passphrase.");
      return;
    }
    try {
      const live = await fetchLiveModels(activeKey.provider, decrypted);
      if (live.length === 0) {
        setStatus("This API key has no chat models available.");
        return;
      }
      const pref = live.some((m) => m.id === activeKey.selected_model)
        ? activeKey.selected_model
        : live[0].id;
      setDecryptedApiKey(decrypted);
      setModels(live);
      setModelId(pref);
      setPassphrase("");
      await persistSelectedModel(activeKey, pref);
    } catch (err) {
      setStatus((err as Error).message);
    }
  }

  async function onLaunch(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    if (!activeKey || !decryptedApiKey || !modelId) {
      setStatus("Unlock your key and pick a model first.");
      return;
    }
    try {
      await launchTask({
        case_name: caseName,
        provider: activeKey.provider,
        model_id: modelId,
        decrypted_api_key: decryptedApiKey,
      });
      await persistSelectedModel(activeKey, modelId);
      setStatus(`Launched ${caseName} on ${modelId}.`);
      refreshTasks();
    } catch (err) {
      setStatus((err as Error).message);
    }
  }

  async function loadSpend(taskId: number) {
    const events = await fetchTaskSpend(taskId);
    setSpendByTask((prev) => ({ ...prev, [taskId]: events }));
  }

  const providerPicker =
    hostedKeys.length > 1 ? (
      <label className="block text-sm">
        Provider
        <select
          className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
          value={provider}
          onChange={(e) => onProviderChange(e.target.value)}
        >
          {hostedKeys.map((k) => (
            <option key={k.provider} value={k.provider}>
              {providerLabel(k.provider)}
            </option>
          ))}
        </select>
      </label>
    ) : null;

  return (
    <div className="mx-auto mt-16 max-w-2xl space-y-8">
      <div>
        <h1 className="mb-3 text-xl font-semibold">Launch a task</h1>
        {hostedKeys.length === 0 ? (
          <p className="text-sm text-slate-400">
            No stored API key — set one up in{" "}
            <Link className="text-sky-400" to="/profile">
              your profile
            </Link>{" "}
            first.
          </p>
        ) : decryptedApiKey === null ? (
          <form onSubmit={onUnlock} className="space-y-3">
            {providerPicker}
            <p className="text-sm text-slate-400">
              Unlock your stored {providerLabel(activeKey?.provider ?? provider)} key to load the models that
              key can use.
            </p>
            <label className="block text-sm">
              Passphrase (to unlock your key for this session)
              <input
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
                type="password"
                value={passphrase}
                onChange={(e) => setPassphrase(e.target.value)}
                required
              />
            </label>
            {status && <p className="text-sm text-slate-300">{status}</p>}
            <button className="w-full rounded bg-sky-600 p-2 font-medium hover:bg-sky-500" type="submit">
              Unlock
            </button>
          </form>
        ) : (
          <form onSubmit={onLaunch} className="space-y-3">
            {providerPicker}
            <label className="block text-sm">
              Case
              <select
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
                value={caseName}
                onChange={(e) => setCaseName(e.target.value)}
              >
                <option value="titanic">titanic</option>
                <option value="house_prices">house_prices</option>
                <option value="disaster_tweets">disaster_tweets</option>
              </select>
            </label>
            <label className="block text-sm">
              Model
              <select
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                required
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            {status && <p className="text-sm text-slate-300">{status}</p>}
            <button className="w-full rounded bg-sky-600 p-2 font-medium hover:bg-sky-500" type="submit">
              Launch
            </button>
          </form>
        )}
      </div>

      <div>
        <h2 className="mb-3 text-lg font-semibold">Your tasks</h2>
        <div className="space-y-3">
          {tasks.map((t) => (
            <div key={t.id} className="rounded border border-slate-700 p-3">
              <div className="flex items-center justify-between text-sm">
                <span>
                  #{t.id} — {t.case_name} ({t.model_id}) — <span className="uppercase">{t.status}</span>
                </span>
                {t.status === "completed" && (
                  <button className="text-sky-400" onClick={() => loadSpend(t.id)}>
                    Show spend
                  </button>
                )}
              </div>
              {spendByTask[t.id] && (
                <div className="mt-3 h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={spendByTask[t.id]}>
                      <XAxis dataKey="agent" tick={{ fontSize: 11 }} />
                      <YAxis />
                      <Tooltip
                        formatter={(value: number, name: string) =>
                          name === "cost_usd" ? [`$${value.toFixed(4)}`, "cost"] : [value, name]
                        }
                      />
                      <Bar dataKey="input_tokens" stackId="tokens" fill="#38bdf8" />
                      <Bar dataKey="output_tokens" stackId="tokens" fill="#0ea5e9" />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="mt-1 text-right text-xs text-slate-400">
                    Total: $
                    {spendByTask[t.id].reduce((sum, e) => sum + e.cost_usd, 0).toFixed(4)}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
