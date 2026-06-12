import { useCallback, useEffect, useState } from "react";
import { fetchReviews, checkAIHealth, logout, getMe, isLoggedIn } from "./api";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";
import ReviewList from "./components/ReviewList";

export default function App() {
  const [user,     setUser]     = useState(null);
  const [authed,   setAuthed]   = useState(isLoggedIn());
  const [reviews,  setReviews]  = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState("");
  const [aiStatus, setAiStatus] = useState(null);

  useEffect(() => {
    if (!authed) { setLoading(false); return; }
    getMe()
      .then((u) => { setUser(u); loadReviews(); checkAI(); })
      .catch(() => { setAuthed(false); setLoading(false); });
  }, []);

  const loadReviews = useCallback(async () => {
    try {
      setError("");
      setReviews(await fetchReviews() || []);
    } catch {
      setError("Failed to load reviews — is the backend running?");
    } finally {
      setLoading(false);
    }
  }, []);

  const checkAI = useCallback(async () => {
    try { setAiStatus(await checkAIHealth()); } catch { setAiStatus(null); }
  }, []);

  useEffect(() => {
    if (!authed || !user) return;
    const t = setInterval(checkAI, 30_000);
    return () => clearInterval(t);
  }, [authed, user, checkAI]);

  const updateReview = useCallback((updated) => {
    setReviews(prev => prev.map(r => r._id === updated._id ? updated : r));
  }, []);

  function handleLoginSuccess(u) {
    setUser(u);
    setAuthed(true);
    loadReviews();
    checkAI();
  }

  // AI status indicator
  function AiDot() {
    const re = aiStatus?.replyEngine || "";
    if (!aiStatus) return <span className="text-xs text-slate-600 flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-slate-700"/>AI offline</span>;
    if (re.includes("flan"))
      return <span className="text-xs text-violet-400 flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-pulse"/>Local AI</span>;
    if (re.includes("anthropic") || re.includes("claude"))
      return <span className="text-xs text-indigo-400 flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-indigo-400"/>Anthropic</span>;
    return <span className="text-xs text-amber-400 flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-amber-400"/>Template mode</span>;
  }

  if (!authed) return <Login onSuccess={handleLoginSuccess}/>;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">

      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto max-w-6xl px-5 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 shadow-lg shadow-violet-500/20"/>
            <div>
              <p className="text-[9px] font-bold uppercase tracking-[0.25em] text-violet-400">EchoReview</p>
              <p className="text-xs font-bold text-white -mt-0.5">Reputation Command Center</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <AiDot/>
            {user && (
              <div className="flex items-center gap-3">
                <div className="hidden sm:block text-right">
                  <p className="text-xs font-semibold text-white">{user.name}</p>
                  <p className="text-[10px] text-slate-500 capitalize">{user.role}</p>
                </div>
                <button onClick={logout}
                  className="text-xs text-slate-500 hover:text-slate-300 border border-slate-700 rounded-lg px-3 py-1.5 hover:bg-slate-800 transition-colors">
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-6xl px-5 py-6 space-y-5">
        {error && (
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 text-rose-400 px-4 py-3 text-sm">
            ⚠ {error}
          </div>
        )}

        {loading ? (
          <div className="py-20 text-center">
            <div className="inline-block h-6 w-6 rounded-full border-2 border-violet-500/30 border-t-violet-500 animate-spin mb-3"/>
            <p className="text-sm text-slate-500">Loading…</p>
          </div>
        ) : (
          <>
            <Dashboard reviews={reviews}/>
            <ReviewList
              reviews={reviews}
              onRefresh={loadReviews}
              onReviewUpdate={updateReview}
            />
          </>
        )}
      </main>
    </div>
  );
}