import { FormEvent, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { register, setToken } from "../lib/api";

export function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const { access_token } = await register(email, password);
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
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          className="w-full rounded border border-slate-700 bg-slate-900 p-2"
          type="password"
          placeholder="Password (min 8 characters)"
          minLength={8}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
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
