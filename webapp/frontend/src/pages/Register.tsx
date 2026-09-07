import { FormEvent, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { register, setToken } from "../lib/api";

const USERNAME_PATTERN = "[A-Za-z0-9._-]{3,32}";

export function RegisterPage() {
  const [username, setUsername] = useState("");
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const { access_token } = await register(username);
      setToken(access_token);
      navigate("/profile");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="mx-auto mt-24 max-w-sm space-y-4">
      <h1 className="text-xl font-semibold">Create an account</h1>
      <form onSubmit={onSubmit} className="space-y-3">
        <input
          className="w-full rounded border border-slate-700 bg-slate-900 p-2"
          type="text"
          autoComplete="username"
          placeholder="Username"
          minLength={3}
          maxLength={32}
          pattern={USERNAME_PATTERN}
          title="3–32 characters: letters, digits, dot, underscore, or hyphen"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <p className="text-xs text-slate-500">3–32 characters: letters, digits, dot, underscore, or hyphen.</p>
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button className="w-full rounded bg-sky-600 p-2 font-medium hover:bg-sky-500" type="submit">
          Create account
        </button>
      </form>
      <p className="text-sm text-slate-400">
        Already have an account? <Link className="text-sky-400" to="/login">Log in</Link>
      </p>
    </div>
  );
}
