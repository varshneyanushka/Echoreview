/**
 * ReviewList.jsx  v5.0
 * Clean card UI. Two reply modes:
 *   🤖 AI generated — Groq → safe template fallback
 *   ✏️ Manual — blank textarea
 */

import { useState, useMemo, useRef, useEffect } from "react";
import { saveReply, updateStatus, generateAIReply } from "../api";

// ── Colour maps ───────────────────────────────────────────────────────────────
const PLATFORM_COLOR = {
  Google:      "bg-blue-500/15 text-blue-400",
  Yelp:        "bg-red-500/15 text-red-400",
  Trustpilot:  "bg-emerald-500/15 text-emerald-400",
  "App Store": "bg-sky-500/15 text-sky-400",
  G2:          "bg-orange-500/15 text-orange-400",
  Website:     "bg-violet-500/15 text-violet-400",
  Facebook:    "bg-indigo-500/15 text-indigo-400",
  Other:       "bg-slate-500/15 text-slate-400",
};
const SENTIMENT_COLOR = {
  Positive: { pill: "bg-emerald-500/15 text-emerald-400", bar: "#10b981" },
  Neutral:  { pill: "bg-slate-500/15 text-slate-400",     bar: "#6366f1" },
  Negative: { pill: "bg-rose-500/15 text-rose-400",       bar: "#f43f5e" },
};
const ISSUE_COLOR = {
  delivery: "bg-amber-500/15 text-amber-400",
  billing:  "bg-rose-500/15 text-rose-400",
  support:  "bg-indigo-500/15 text-indigo-400",
  product:  "bg-sky-500/15 text-sky-400",
  refund:   "bg-purple-500/15 text-purple-400",
  general:  "bg-slate-500/15 text-slate-400",
};

// ── Source config ─────────────────────────────────────────────────────────────
const SOURCE_META = {
  "groq":       { icon: "⚡", label: "Generated with Groq AI", color: "text-violet-400", bg: "bg-violet-500/10 border-violet-500/20" },
  "template":   { icon: "📝", label: "Template fallback used", color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" },
  "manual":     { icon: "✏️", label: "Manual",       color: "text-slate-400",  bg: "bg-slate-500/10 border-slate-700" },
};

function getSourceMeta(source = "") {
  const s = source.toLowerCase();
  if (s.includes("groq")) return SOURCE_META.groq;
  if (s === "template")        return SOURCE_META["template"];
  if (s === "manual")          return SOURCE_META["manual"];
  return SOURCE_META["template"];
}

// ── Tiny helpers ──────────────────────────────────────────────────────────────
function Stars({ rating }) {
  return (
    <span>
      {[1,2,3,4,5].map(i => (
        <span key={i} className={i <= rating ? "text-amber-400" : "text-slate-700"}>★</span>
      ))}
    </span>
  );
}

function Pill({ children, className }) {
  return (
    <span className={`rounded-md px-2 py-0.5 text-[10px] font-semibold ${className}`}>
      {children}
    </span>
  );
}

function Avatar({ name }) {
  const initials = name.split(" ").filter(Boolean).slice(0,2).map(w=>w[0].toUpperCase()).join("");
  const hue = [...name].reduce((a,c) => a + c.charCodeAt(0), 0) % 360;
  return (
    <div className="h-8 w-8 flex-shrink-0 rounded-full flex items-center justify-center text-xs font-bold text-white"
      style={{ background: `hsl(${hue},60%,42%)` }}>
      {initials}
    </div>
  );
}

// ── Priority stripe color ─────────────────────────────────────────────────────
function priorityStripe(score) {
  if (score >= 70) return "#f43f5e";
  if (score >= 40) return "#f59e0b";
  return "#334155";
}

// ── Reply mode tab ────────────────────────────────────────────────────────────
function ModeTab({ active, onClick, icon, label, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all
        ${active
          ? "bg-slate-700 text-white"
          : "text-slate-500 hover:text-slate-300 hover:bg-slate-800"
        } ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}
    >
      <span>{icon}</span>
      {label}
    </button>
  );
}

// ── Review card ───────────────────────────────────────────────────────────────
function ReviewCard({ review, onUpdate }) {
  const [open, setOpen]           = useState(false);
  const [mode, setMode]           = useState("ai");   // "ai" | "manual"
  const [draft, setDraft]         = useState("");
  const [genSource, setGenSource] = useState("");
  const [loading, setLoading]     = useState(false);
  const [saving, setSaving]       = useState(false);
  const [err, setErr]             = useState("");
  const [showFull, setShowFull]   = useState(false);
  const ta = useRef(null);

  // Auto-size textarea
  useEffect(() => {
    if (ta.current) {
      ta.current.style.height = "auto";
      ta.current.style.height = ta.current.scrollHeight + "px";
    }
  }, [draft]);

  const ps         = review.priorityScore || 0;
  const resolved   = review.status === "resolved";
  const hasReply   = !!review.replyText;
  const sentiment  = review.sentimentLabel || "Neutral";
  const sentMeta   = SENTIMENT_COLOR[sentiment] || SENTIMENT_COLOR.Neutral;
  const reviewText = review.text || "";
  const preview    = reviewText.length > 160 ? reviewText.slice(0, 160) + "…" : reviewText;

  // ── Generate reply ──────────────────────────────────────────────────────────
  async function generate(newMode) {
    setMode(newMode);
    setErr("");
    if (newMode === "manual") {
      setDraft("");
      setGenSource("manual");
      setOpen(true);
      return;
    }
    setLoading(true);
    setGenSource("generating-groq");
    setOpen(true);
    try {
      const result = await generateAIReply(review._id);
      setDraft(result.reply || "");
      setGenSource(result.source || "template");
      setOpen(true);
    } catch {
      setErr("The reply service did not respond. Please try again or write a manual reply.");
      setDraft("");
      setGenSource("manual");
      setOpen(true);
    } finally {
      setLoading(false);
    }
  }

  // ── Save ────────────────────────────────────────────────────────────────────
  async function handleSave() {
    if (!draft.trim()) { setErr("Reply cannot be empty."); return; }
    setSaving(true); setErr("");
    try {
      const updated = await saveReply(review._id, draft.trim(), review.issueCategory, genSource || "manual");
      onUpdate(updated);
      setOpen(false);
    } catch {
      setErr("Failed to save. Please try again.");
    } finally {
      setSaving(false);
    }
  }

  async function handleResolve() {
    try { onUpdate(await updateStatus(review._id, "resolved")); } catch { setErr("Status update failed."); }
  }
  async function handleReopen() {
    try { onUpdate(await updateStatus(review._id, "new")); } catch { setErr("Reopen failed."); }
  }

  const srcMeta = getSourceMeta(genSource);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 overflow-hidden">
      {/* Priority stripe top */}
      <div className="h-0.5 w-full" style={{ background: priorityStripe(ps) }} />

      <div className="p-4">
        {/* ── Header row ─────────────────────────────────────────────────── */}
        <div className="flex gap-3 items-start">
          <Avatar name={review.customerName} />

          <div className="flex-1 min-w-0">
            {/* Name + stars + pills */}
            <div className="flex flex-wrap items-center gap-1.5 mb-1">
              <span className="text-sm font-semibold text-white">{review.customerName}</span>
              <Stars rating={review.rating} />
              <Pill className={PLATFORM_COLOR[review.platform] || PLATFORM_COLOR.Other}>
                {review.platform || "Other"}
              </Pill>
              <Pill className={sentMeta.pill}>{sentiment}</Pill>
              <Pill className={ISSUE_COLOR[review.issueCategory] || ISSUE_COLOR.general}>
                {review.issueCategory || "general"}
              </Pill>
            </div>

            {/* Review text */}
            <p className="text-sm text-slate-400 leading-relaxed">
              {showFull ? reviewText : preview}
              {reviewText.length > 160 && !showFull && (
                <button onClick={() => setShowFull(true)}
                  className="ml-1 text-violet-400 hover:text-violet-300 text-xs">
                  more
                </button>
              )}
            </p>

            {/* Existing reply chip */}
            {hasReply && !open && (
              <div className="mt-2 flex items-start gap-2 rounded-lg bg-emerald-500/5 border border-emerald-500/15 px-3 py-2">
                <span className="text-emerald-500 text-xs mt-0.5">↩</span>
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-slate-300 line-clamp-2">{review.replyText}</p>
                  {review.replySource && (
                    <span className={`mt-1 inline-block text-[10px] font-semibold ${getSourceMeta(review.replySource).color}`}>
                      {getSourceMeta(review.replySource).icon} {getSourceMeta(review.replySource).label}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Priority badge + date */}
          <div className="flex flex-col items-end gap-1 flex-shrink-0">
            {ps >= 40 && (
              <span className={`text-[10px] font-bold rounded px-1.5 py-0.5
                ${ps >= 70 ? "bg-rose-500/20 text-rose-400" : "bg-amber-500/20 text-amber-400"}`}>
                P{ps}
              </span>
            )}
            <span className="text-[10px] text-slate-600">
              {new Date(review.date || review.createdAt).toLocaleDateString("en-GB",
                { day:"numeric", month:"short" })}
            </span>
          </div>
        </div>

        {/* ── Action row ─────────────────────────────────────────────────── */}
        {!resolved && !open && (
          <div className="mt-3 flex flex-wrap gap-1.5 items-center">
            <button onClick={() => generate("ai")} disabled={loading}
              className="flex items-center gap-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-xs font-semibold text-white transition-colors disabled:opacity-50">
              {loading && mode === "ai"
                ? <><span className="h-3 w-3 border border-white/30 border-t-white rounded-full animate-spin"/>Generating with Groq…</>
                : <>🤖 {hasReply ? "Re-generate AI" : "AI generated"}</>}
            </button>

            <button onClick={() => generate("manual")}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 hover:border-slate-600 bg-slate-800 hover:bg-slate-700 px-3 py-1.5 text-xs font-medium text-slate-400 transition-colors">
              ✏️ Manual
            </button>

            {hasReply && !open && (
              <button onClick={handleResolve}
                className="ml-auto rounded-lg border border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 px-3 py-1.5 text-xs font-semibold text-emerald-400 transition-colors">
                ✓ Resolve
              </button>
            )}
          </div>
        )}

        {resolved && (
          <div className="mt-3 flex gap-2 items-center">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"/>Resolved
            </span>
            <button onClick={handleReopen}
              className="text-xs text-slate-500 hover:text-slate-300 border border-slate-700 rounded px-2 py-1 hover:bg-slate-800 transition-colors">
              Reopen
            </button>
          </div>
        )}

        {/* ── Reply editor ──────────────────────────────────────────────── */}
        {open && (
          <div className="mt-4 space-y-3">
            {/* Mode switcher */}
            <div className="flex gap-1 bg-slate-800/50 rounded-lg p-1 w-fit">
              <ModeTab active={mode==="ai"}       onClick={() => generate("ai")}       icon="🤖" label="AI generated" disabled={loading} />
              <ModeTab active={mode==="manual"}   onClick={() => generate("manual")}   icon="✏️" label="Manual"   disabled={loading} />
            </div>

            {/* Source info strip */}
            {loading && mode === "ai" && (
              <div className="flex items-center gap-2 rounded-lg border border-violet-500/20 bg-violet-500/10 px-3 py-2">
                <span className="h-3 w-3 border border-violet-300/30 border-t-violet-300 rounded-full animate-spin" />
                <span className="text-xs font-semibold text-violet-300">Generating with Groq AI…</span>
                <span className="text-xs text-slate-500">— template fallback will be used if it times out</span>
              </div>
            )}

            {genSource && !loading && mode !== "manual" && (
              <div className={`flex items-center gap-2 rounded-lg border px-3 py-2 ${srcMeta.bg}`}>
                <span className="text-sm">{srcMeta.icon}</span>
                <span className={`text-xs font-semibold ${srcMeta.color}`}>{srcMeta.label}</span>
                <span className="text-xs text-slate-500">— review and edit before saving</span>
              </div>
            )}

            {/* Textarea */}
            <textarea ref={ta} value={draft}
              onChange={e => setDraft(e.target.value)}
              placeholder={mode === "manual" ? "Write your reply…" : "Loading reply…"}
              disabled={loading}
              className="w-full resize-none rounded-lg border border-slate-700 bg-slate-800 p-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-violet-500/60 focus:border-violet-500/40 min-h-[80px] disabled:opacity-50"
            />

            {err && <p className="text-xs text-rose-400">{err}</p>}

            {/* Footer actions */}
            <div className="flex gap-2 items-center">
              <button onClick={handleSave} disabled={saving || !draft.trim() || loading}
                className="flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 px-4 py-2 text-xs font-semibold text-white transition-colors">
                {saving
                  ? <><span className="h-3 w-3 border border-white/30 border-t-white rounded-full animate-spin"/>Saving…</>
                  : "Save reply"}
              </button>
              <button onClick={() => setOpen(false)}
                className="text-xs text-slate-500 hover:text-slate-300 px-3 py-2 rounded-lg hover:bg-slate-800 transition-colors">
                Cancel
              </button>
              <span className="ml-auto text-[10px] text-slate-600">{draft.length} chars</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
const TABS = [
  { key:"new",         label:"Needs Reply", f:(r)=>r.status==="new",         dot:"bg-rose-500" },
  { key:"in_progress", label:"In Progress", f:(r)=>r.status==="in_progress", dot:"bg-amber-500" },
  { key:"resolved",    label:"Resolved",    f:(r)=>r.status==="resolved",    dot:"bg-emerald-500" },
  { key:"all",         label:"All",         f:()=>true,                       dot:"bg-violet-500" },
];

const SORTS = [
  { v:"priority",    l:"Priority" },
  { v:"date_desc",   l:"Newest" },
  { v:"date_asc",    l:"Oldest" },
  { v:"rating_asc",  l:"Rating ↑" },
  { v:"rating_desc", l:"Rating ↓" },
];

export default function ReviewList({ reviews, onRefresh, onReviewUpdate }) {
  const [tab,    setTab]    = useState("new");
  const [sort,   setSort]   = useState("priority");
  const [search, setSearch] = useState("");

  const counts = useMemo(() => {
    const c = {};
    TABS.forEach(({ key, f }) => { c[key] = reviews.filter(f).length; });
    return c;
  }, [reviews]);

  const visible = useMemo(() => {
    const t = TABS.find(x => x.key === tab);
    let list = reviews.filter(t.f);
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(r =>
        (r.customerName||"").toLowerCase().includes(q) ||
        (r.text||"").toLowerCase().includes(q) ||
        (r.platform||"").toLowerCase().includes(q) ||
        (r.issueCategory||"").toLowerCase().includes(q));
    }
    return [...list].sort((a,b) => {
      if (sort==="priority")    return (b.priorityScore||0)-(a.priorityScore||0);
      if (sort==="rating_asc")  return (a.rating||0)-(b.rating||0);
      if (sort==="rating_desc") return (b.rating||0)-(a.rating||0);
      if (sort==="date_asc")    return new Date(a.date||a.createdAt)-new Date(b.date||b.createdAt);
      return new Date(b.date||b.createdAt)-new Date(a.date||a.createdAt);
    });
  }, [reviews, tab, sort, search]);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 overflow-hidden">
      {/* Header */}
      <div className="border-b border-slate-800 px-5 pt-5 pb-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-white">Review Queue</h2>
          <button onClick={onRefresh}
            className="text-xs text-slate-500 hover:text-slate-300 border border-slate-700 rounded-lg px-3 py-1.5 hover:bg-slate-800 transition-colors">
            ↻ Refresh
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-0.5">
          {TABS.map(({ key, label, dot }) => {
            const active = tab === key;
            return (
              <button key={key} onClick={() => setTab(key)}
                className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all
                  ${active ? "bg-slate-800 text-white" : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"}`}>
                {active && <span className={`h-1.5 w-1.5 rounded-full ${dot}`}/>}
                {label}
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold tabular-nums
                  ${active ? "bg-slate-700 text-white" : "bg-slate-800 text-slate-500"}`}>
                  {counts[key]}
                </span>
              </button>
            );
          })}
        </div>

        {/* Search + sort */}
        <div className="flex gap-2">
          <input value={search} onChange={e=>setSearch(e.target.value)}
            placeholder="Search reviews…"
            className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-violet-500/40"/>
          <select value={sort} onChange={e=>setSort(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-violet-500/40">
            {SORTS.map(o=><option key={o.v} value={o.v}>{o.l}</option>)}
          </select>
        </div>
      </div>

      {/* Cards */}
      <div className="p-4 space-y-2.5">
        {visible.length === 0 ? (
          <div className="py-14 text-center">
            <p className="text-3xl mb-3">{tab==="resolved"?"🎉":tab==="new"?"🎊":"📭"}</p>
            <p className="text-sm text-slate-500">
              {search ? "No reviews match your search."
               : tab==="new" ? "All caught up — no reviews need a reply!"
               : tab==="in_progress" ? "Nothing in progress right now."
               : tab==="resolved" ? "No resolved reviews yet."
               : "No reviews found."}
            </p>
          </div>
        ) : (
          visible.map(r => <ReviewCard key={r._id} review={r} onUpdate={onReviewUpdate}/>)
        )}
      </div>

      {visible.length > 0 && (
        <div className="border-t border-slate-800 px-5 py-3">
          <p className="text-[11px] text-slate-600">
            {visible.length} of {counts[tab]} reviews{search && " (filtered)"}
          </p>
        </div>
      )}
    </div>
  );
}
