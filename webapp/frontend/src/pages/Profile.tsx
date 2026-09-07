import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { encryptApiKey } from "../lib/crypto";
import { fetchModelCatalog, listStoredKeys, ModelCatalog, StoredKey, upsertStoredKey } from "../lib/api";
import { HelpStepper, OPENAI_KEY_HELP, PASSPHRASE_HELP } from "../components/HelpStepper";

export function ProfilePage() {
  const [catalog, setCatalog] = useState<ModelCatalog>({});
  const [storedKeys, setStoredKeys] = useState<StoredKey[]>([]);
  const [provider, setProvider] = useState("openai");
  const [modelId, setModelId] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchModelCatalog().then((c) => {
      setCatalog(c);
      const firstProvider = Object.keys(c)[0];
      if (firstProvider) {
        setProvider(firstProvider);
        setModelId(c[firstProvider][0]?.id ?? "");
      }
    });
    listStoredKeys().then(setStoredKeys);
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    try {
      const blob = await encryptApiKey(apiKey, passphrase);
      const saved = await upsertStoredKey({ provider, selected_model: modelId, ...blob });
      setStoredKeys((prev) => [...prev.filter((k) => k.provider !== provider), saved]);
      setApiKey("");
      setPassphrase("");
      setStatus(`Saved ${provider} key for ${modelId}. Taking you to your tasks…`);
      setTimeout(() => navigate("/tasks"), 800);
    } catch (err) {
      setStatus(`Error: ${(err as Error).message}`);
    }
  }

  return (
    <div className="mx-auto mt-16 max-w-xl space-y-6">
      <h1 className="text-xl font-semibold">Provider &amp; model</h1>

      <div className="space-y-2">
        <h2 className="text-sm font-medium text-slate-300">Stored keys</h2>
        {storedKeys.length === 0 && <p className="text-sm text-slate-500">No provider keys stored yet.</p>}
        {storedKeys.map((k) => (
          <div key={k.provider} className="rounded border border-slate-700 p-2 text-sm">
            {k.provider} — {k.selected_model}{" "}
            <span className="text-slate-500">(updated {new Date(k.updated_at).toLocaleString()})</span>
          </div>
        ))}
      </div>

      <form onSubmit={onSubmit} className="space-y-3">
        <label className="block text-sm">
          Provider
          <select
            className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
            value={provider}
            onChange={(e) => {
              setProvider(e.target.value);
              setModelId(catalog[e.target.value]?.[0]?.id ?? "");
            }}
          >
            {Object.keys(catalog).map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>

        <label className="block text-sm">
          Model
          <select
            className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          >
            {(catalog[provider] ?? []).map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>

        <label className="block text-sm">
          API key
          <input
            className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
            type="password"
            placeholder={`Your ${provider} API key`}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            required
          />
        </label>
        <HelpStepper title="Where do I get an OpenAI API key?" steps={OPENAI_KEY_HELP} />

        <label className="block text-sm">
          Encryption passphrase
          <input
            className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
            type="password"
            placeholder="Only you know this — we never see it"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            minLength={8}
            required
          />
        </label>
        <HelpStepper title="Why does this ask for a passphrase?" steps={PASSPHRASE_HELP} />

        {status && <p className="text-sm text-slate-300">{status}</p>}
        <button className="w-full rounded bg-sky-600 p-2 font-medium hover:bg-sky-500" type="submit">
          Save key
        </button>
      </form>
    </div>
  );
}
