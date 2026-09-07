interface Step {
  title: string;
  body: string;
}

/** Static, no-backend step-by-step help for a specific task (e.g. "get an OpenAI API key"). */
export function HelpStepper({ title, steps }: { title: string; steps: Step[] }) {
  return (
    <details className="rounded-lg border border-slate-700 bg-slate-900 p-3 text-sm">
      <summary className="cursor-pointer font-medium text-slate-200">{title}</summary>
      <ol className="mt-3 list-decimal space-y-2 pl-5 text-slate-300">
        {steps.map((step, i) => (
          <li key={i}>
            <span className="font-medium text-slate-100">{step.title}</span> — {step.body}
          </li>
        ))}
      </ol>
    </details>
  );
}

export const OPENAI_KEY_HELP: Step[] = [
  { title: "Create an OpenAI account", body: "Sign up at platform.openai.com if you don't have one yet." },
  { title: "Open the API keys page", body: "Go to platform.openai.com → API keys → \"Create new secret key\"." },
  { title: "Copy the key once", body: "OpenAI shows it only once — copy it now, you can't view it again later." },
  {
    title: "Paste it below and set a passphrase",
    body:
      "Your key is encrypted in your browser before it's sent anywhere. Choose a passphrase you'll remember — " +
      "if you forget it, the key can't be recovered and you'll need to delete and re-enter a new one.",
  },
];

export const OLLAMA_KEY_HELP: Step[] = [
  { title: "Create an Ollama account", body: "Sign in at ollama.com if you don't have an account yet." },
  { title: "Open API keys", body: "Go to ollama.com → Settings → API keys and create a new key." },
  { title: "Copy the key", body: "Paste it below. Cloud model names come from that key at launch time, not a static list." },
  {
    title: "Set a passphrase",
    body:
      "Your key is encrypted in your browser before it's sent anywhere. Choose a passphrase you'll remember — " +
      "if you forget it, the key can't be recovered and you'll need to delete and re-enter a new one.",
  },
];

export const PASSPHRASE_HELP: Step[] = [
  {
    title: "Why a separate passphrase?",
    body: "It encrypts your API key in your browser, so the server only ever stores unreadable ciphertext.",
  },
  {
    title: "There is no \"forgot passphrase\" recovery",
    body: "Nobody — including us — can recover a lost passphrase. Write it down somewhere safe, or reuse a password manager entry.",
  },
];
