import { FormEvent, useEffect, useState } from "react";
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

const HOSTED_PROVIDER = "openai";

function openaiBlob(stored: StoredKey, selectedModel: string) {
  return {
    provider: HOSTED_PROVIDER,
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
  const [caseName, setCaseName] = useState("titanic");
  const [passphrase, setPassphrase] = useState("");
  const [decryptedApiKey, setDecryptedApiKey] = useState<string | null>(null);
  const [models, setModels] = useState<LiveModel[]>([]);
  const [modelId, setModelId] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [spendByTask, setSpendByTask] = useState<Record<number, TokenSpendEvent[]>>({});

  const openaiKey = storedKeys.find((k) => k.provider === HOSTED_PROVIDER);

  function refreshTasks() {
    listTasks().then(setTasks);
  }

  useEffect(() => {
    refreshTasks();
    listStoredKeys().then(setStoredKeys);
  }, []);

  async function persistSelectedModel(stored: StoredKey, nextModel: string) {
    if (!nextModel || stored.selected_model === nextModel) {
      return;
    }
    const saved = await upsertStoredKey(openaiBlob(stored, nextModel));
    setStoredKeys((prev) => [...prev.filter((k) => k.provider !== HOSTED_PROVIDER), saved]);
  }

  async function onUnlock(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    if (!openaiKey) {
      setStatus("No stored OpenAI key — set one up in your profile first.");
      return;
    }
    let decrypted: string;
    try {
      decrypted = await decryptApiKey(openaiKey, passphrase);
    } catch {
      setStatus("Couldn't decrypt your key — check your passphrase.");
      return;
    }
    try {
      const live = await fetchLiveModels(decrypted);
      if (live.length === 0) {
        setStatus("This API key has no chat models available.");
        return;
      }
      const pref = live.some((m) => m.id === openaiKey.selected_model)
        ? openaiKey.selected_model
        : live[0].id;
      setDecryptedApiKey(decrypted);
      setModels(live);
      setModelId(pref);
      setPassphrase("");
      await persistSelectedModel(openaiKey, pref);
    } catch (err) {
      setStatus((err as Error).message);
    }
  }

  async function onLaunch(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    if (!openaiKey || !decryptedApiKey || !modelId) {
      setStatus("Unlock your key and pick a model first.");
      return;
    }
    try {
      await launchTask({
        case_name: caseName,
        provider: HOSTED_PROVIDER,
        model_id: modelId,
        decrypted_api_key: decryptedApiKey,
      });
      await persistSelectedModel(openaiKey, modelId);
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

  return (
    <div className="mx-auto mt-16 max-w-2xl space-y-8">
      <div>
        <h1 className="mb-3 text-xl font-semibold">Launch a task</h1>
        {decryptedApiKey === null ? (
          <form onSubmit={onUnlock} className="space-y-3">
            <p className="text-sm text-slate-400">
              Unlock your stored OpenAI key to load the models that key can use.
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
