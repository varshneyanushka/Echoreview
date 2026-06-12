import { useState } from "react";
import { login } from "../api";

export default function Login({ onSuccess }) {
  const [email,    setEmail]    = useState("admin@echoreview.ai");
  const [password, setPassword] = useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await login(email, password);
      onSuccess(data.user);
    } catch (err) {
      setError(err.response?.data?.message || "Invalid credentials");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center px-4">
      <div className="w-full max-w-sm space-y-6">

        {/* Logo */}
        <div className="text-center space-y-2">
          <div className="mx-auto h-14 w-14 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600
                          flex items-center justify-center shadow-xl shadow-violet-500/30">
            <svg className="h-7 w-7 text-white" fill="none" viewBox="0 0 24 24"
                 stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3
                       m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.347.347A3.75
                       3.75 0 0114.25 21h-4.5a3.75 3.75 0 01-2.646-1.097l-.347-.347z" />
            </svg>
          </div>
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.3em] text-violet-400">
              EchoReview AI
            </p>
            <h1 className="text-2xl font-black text-white mt-0.5">
              Reputation Command Center
            </h1>
            <p className="text-sm text-slate-500 mt-1">Sign in to your workspace</p>
          </div>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-7 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">
                Email address
              </label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                placeholder="admin@echoreview.ai"
                className="w-full rounded-xl border border-slate-700 bg-slate-800 px-3.5 py-2.5
                           text-sm text-slate-200 placeholder-slate-600 focus:outline-none
                           focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50
                           transition-all"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-400 mb-1.5">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                placeholder="••••••••"
                className="w-full rounded-xl border border-slate-700 bg-slate-800 px-3.5 py-2.5
                           text-sm text-slate-200 placeholder-slate-600 focus:outline-none
                           focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50
                           transition-all"
              />
            </div>

            {error && (
              <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-3.5 py-2.5
                              text-xs text-rose-400">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-60
                         disabled:cursor-not-allowed py-2.5 text-sm font-bold text-white
                         transition-colors shadow-lg shadow-violet-600/20"
            >
              {loading
                ? <span className="flex items-center justify-center gap-2">
                    <span className="h-4 w-4 rounded-full border-2 border-white/30
                                     border-t-white animate-spin" />
                    Signing in…
                  </span>
                : "Sign in"}
            </button>
          </form>
        </div>

        <p className="text-center text-[11px] text-slate-600">
          Default: admin@echoreview.ai / Admin@123
        </p>
      </div>
    </div>
  );
}