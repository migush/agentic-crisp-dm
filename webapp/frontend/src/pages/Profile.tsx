import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { encryptApiKey } from "../lib/crypto";
import { listStoredKeys, StoredKey, upsertStoredKey } from "../lib/api";
import { HelpStepper, OPENAI_KEY_HELP, PASSPHRASE_HELP } from "../components/HelpStepper";

const HOSTED_PROVIDER = "openai";

export function ProfilePage() {
  const [storedKeys, setStoredKeys] = useState<StoredKey[]>([]);
  const [apiKey, setApiKey] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const navigate = useNavigate();

  const openaiKey = storedKeys.find((k) => k.provider === HOSTED_PROVIDER);

  useEffect(() => {
    listStoredKeys().then(setStoredKeys);
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    try {
      const blob = await encryptApiKey(apiKey, passphrase);
      const saved = await upsertStoredKey({
        provider: HOSTED_PROVIDER,
        selected_model: openaiKey?.selected_model ?? "",
        ...blob,
      });
      setStoredKeys((prev) => [...prev.filter((k) => k.provider !== HOSTED_PROVIDER), saved]);
      setApiKey("");
      setPassphrase("");
      setStatus("Saved OpenAI key. Taking you to your tasks…");
      setTimeout(() => navigate("/tasks"), 800);
    } catch (err) {
      setStatus(`Error: ${(err as Error).message}`);
    }
  }

  return (
    <div className="mx-auto mt-16 max-w-xl space-y-6">
      <h1 className="text-xl font-semibold">OpenAI API key</h1>

      <div className="space-y-2">
        <h2 className="text-sm font-medium text-slate-300">Stored key</h2>
        {!openaiKey && <p className="text-sm text-slate-500">No OpenAI key stored yet.</p>}
        {openaiKey && (
          <div className="rounded border border-slate-700 p-2 text-sm">
            openai
            {openaiKey.selected_model ? ` — ${openaiKey.selected_model}` : " — no model chosen yet"}{" "}
            <span className="text-slate-500">(updated {new Date(openaiKey.updated_at).toLocaleString()})</span>
          </div>
        )}
      </div>

      <form onSubmit={onSubmit} className="space-y-3">
        <label className="block text-sm">
          API key
          <input
            className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
            type="password"
            placeholder="Your OpenAI API key"
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
