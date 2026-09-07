import { FormEvent, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { listStoredKeys, login, setToken } from "../lib/api";

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const { access_token } = await login(username);
      setToken(access_token);
      // Users with a provider key already set up land on their task
      // dashboard; first-timers go set one up in Profile first.
      const storedKeys = await listStoredKeys().catch(() => []);
      navigate(storedKeys.length > 0 ? "/tasks" : "/profile");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="mx-auto mt-24 max-w-sm space-y-4">
      <h1 className="text-xl font-semibold">Log in</h1>
      <form onSubmit={onSubmit} className="space-y-3">
        <input
          className="w-full rounded border border-slate-700 bg-slate-900 p-2"
          type="text"
          autoComplete="username"
          placeholder="Username"
          minLength={3}
          maxLength={254}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button className="w-full rounded bg-sky-600 p-2 font-medium hover:bg-sky-500" type="submit">
          Log in
        </button>
      </form>
      <p className="text-sm text-slate-400">
        No account? <Link className="text-sky-400" to="/register">Create one</Link>
      </p>
    </div>
  );
}
