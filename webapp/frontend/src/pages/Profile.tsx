import { FormEvent, useEffect, useState } from "react";
import { encryptApiKey } from "../lib/crypto";
import { listStoredKeys, StoredKey, upsertStoredKey } from "../lib/api";
import { HelpStepper, OPENAI_KEY_HELP, OLLAMA_KEY_HELP, PASSPHRASE_HELP } from "../components/HelpStepper";

type HostedProvider = "openai" | "ollama_cloud";

const PROVIDERS: {
  id: HostedProvider;
  title: string;
  placeholder: string;
  helpTitle: string;
  help: typeof OPENAI_KEY_HELP;
}[] = [
  {
    id: "openai",
    title: "OpenAI",
    placeholder: "Your OpenAI API key",
    helpTitle: "Where do I get an OpenAI API key?",
    help: OPENAI_KEY_HELP,
  },
  {
    id: "ollama_cloud",
    title: "Ollama Cloud",
    placeholder: "Your Ollama API key",
    helpTitle: "Where do I get an Ollama API key?",
    help: OLLAMA_KEY_HELP,
  },
];

function ProviderKeyForm({
  provider,
  title,
  placeholder,
  helpTitle,
  help,
  stored,
  onSaved,
}: {
  provider: HostedProvider;
  title: string;
  placeholder: string;
  helpTitle: string;
  help: typeof OPENAI_KEY_HELP;
  stored: StoredKey | undefined;
  onSaved: (saved: StoredKey) => void;
}) {
  const [apiKey, setApiKey] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setStatus(null);
    try {
      const blob = await encryptApiKey(apiKey, passphrase);
      const saved = await upsertStoredKey({
        provider,
        selected_model: stored?.selected_model ?? "",
        ...blob,
      });
      setApiKey("");
      setPassphrase("");
      onSaved(saved);
      setStatus(`Saved ${title} key.`);
    } catch (err) {
      setStatus(`Error: ${(err as Error).message}`);
    }
  }

  return (
    <section className="space-y-3 rounded border border-slate-800 p-4">
      <h2 className="text-lg font-medium">{title}</h2>
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-slate-300">Stored key</h3>
        {!stored && <p className="text-sm text-slate-500">No {title} key stored yet.</p>}
        {stored && (
          <div className="rounded border border-slate-700 p-2 text-sm">
            {provider}
            {stored.selected_model ? ` — ${stored.selected_model}` : " — no model chosen yet"}{" "}
            <span className="text-slate-500">(updated {new Date(stored.updated_at).toLocaleString()})</span>
          </div>
        )}
      </div>
      <form onSubmit={onSubmit} className="space-y-3">
        <label className="block text-sm">
          API key
          <input
            className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
            type="password"
            placeholder={placeholder}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            required
          />
        </label>
        <HelpStepper title={helpTitle} steps={help} />
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
          Save {title} key
        </button>
      </form>
    </section>
  );
}

export function ProfilePage() {
  const [storedKeys, setStoredKeys] = useState<StoredKey[]>([]);

  useEffect(() => {
    listStoredKeys().then(setStoredKeys);
  }, []);

  function onSaved(saved: StoredKey) {
    setStoredKeys((prev) => [...prev.filter((k) => k.provider !== saved.provider), saved]);
  }

  return (
    <div className="mx-auto mt-16 max-w-xl space-y-6">
      <h1 className="text-xl font-semibold">API keys</h1>
      <p className="text-sm text-slate-400">
        Store an OpenAI key, an Ollama Cloud key, or both. Each is encrypted in your browser with its own
        passphrase.
      </p>
      {PROVIDERS.map((spec) => (
        <ProviderKeyForm
          key={spec.id}
          {...spec}
          provider={spec.id}
          stored={storedKeys.find((k) => k.provider === spec.id)}
          onSaved={onSaved}
        />
      ))}
    </div>
  );
}
