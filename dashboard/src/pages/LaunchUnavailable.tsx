/**
 * Shown instead of the Launch page in the hosted deployment.
 *
 * Starting a run needs the user's LLM provider API key, and that key is only
 * ever decrypted in the browser on the account app's Tasks page (the backend
 * stores ciphertext only). So the dashboard can't launch here — the matching
 * backend route returns 403.
 */
export function LaunchUnavailable() {
  return (
    <div className="max-w-xl">
      <h2 className="text-xl font-semibold text-slate-200">Start a run from Tasks</h2>
      <p className="mt-3 text-sm text-slate-400">
        Runs use your own provider API key, which is decrypted in your browser with
        your passphrase — so they're launched from the Tasks page, not from here.
        Once a run finishes it appears in this dashboard automatically.
      </p>
      <a
        href="/tasks"
        className="mt-5 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-fuchsia-500 to-pink-500 px-4 py-2.5 text-sm font-semibold text-white shadow-md transition-all hover:scale-[1.03] hover:shadow-lg"
      >
        Go to Tasks
      </a>
    </div>
  );
}
