import { FormEvent, useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { decryptApiKey } from "../lib/crypto";
import {
  fetchTaskSpend,
  launchTask,
  listStoredKeys,
  listTasks,
  StoredKey,
  TaskSummary,
  TokenSpendEvent,
} from "../lib/api";

export function TasksPage() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [storedKeys, setStoredKeys] = useState<StoredKey[]>([]);
  const [provider, setProvider] = useState("");
  const [caseName, setCaseName] = useState("titanic");
  const [passphrase, setPassphrase] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [spendByTask, setSpendByTask] = useState<Record<number, TokenSpendEvent[]>>({});

  function refreshTasks() {
    listTasks().then(setTasks);
  }

  useEffect(() => {
    refreshTasks();
    listStoredKeys().then((keys) => {
      setStoredKeys(keys);
      if (keys[0]) setProvider(keys[0].provider);
    });
  }, []);

  async function onLaunch(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    const stored = storedKeys.find((k) => k.provider === provider);
    if (!stored) {
      setStatus("No stored key for this provider — set one up in your profile first.");
      return;
    }
    try {
      const decryptedApiKey = await decryptApiKey(stored, passphrase);
      await launchTask({
        case_name: caseName,
        provider,
        model_id: stored.selected_model,
        decrypted_api_key: decryptedApiKey,
      });
      setPassphrase("");
      setStatus(`Launched ${caseName} on ${stored.selected_model}.`);
      refreshTasks();
    } catch {
      setStatus("Couldn't decrypt your key — check your passphrase.");
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
            Provider (using your stored key)
            <select
              className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              {storedKeys.map((k) => (
                <option key={k.provider} value={k.provider}>
                  {k.provider} ({k.selected_model})
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            Passphrase (to unlock your key for this run only)
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
            Launch
          </button>
        </form>
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
