/**
 * Dashboard.jsx  v5.1
 * ─────────────────────────────────────────────────────────────────────────────
 * Three tabs: Overview | Charts | AI Insights
 * Insights now show Claude NLP output: churn risk, urgent alerts,
 * complaint themes, positive signals, recommendation cards with timeframes.
 */

import { useEffect, useState, useCallback, useRef } from "react";
import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer,
  AreaChart, Area,
  Tooltip as RTooltip,
} from "recharts";
import { fetchAnalytics, openAnalyticsStream, fetchInsights } from "../api";

// ── Palettes ──────────────────────────────────────────────────────────────────
const SENT_COLOR  = { Positive: "#10b981", Neutral: "#6366f1", Negative: "#f43f5e" };
const PLAT_COLORS = ["#6366f1","#10b981","#f59e0b","#f43f5e","#0ea5e9","#a855f7","#14b8a6","#fb923c"];
const ISSUE_META  = {
  delivery: { color:"#f59e0b", icon:"🚚", label:"Delivery" },
  billing:  { color:"#f43f5e", icon:"💳", label:"Billing" },
  support:  { color:"#6366f1", icon:"🎧", label:"Support" },
  product:  { color:"#0ea5e9", icon:"📦", label:"Product" },
  refund:   { color:"#a855f7", icon:"↩️", label:"Refund" },
  general:  { color:"#10b981", icon:"💬", label:"General" },
};
const SEVERITY = {
  critical: { bg:"bg-rose-500/10",    border:"border-rose-500/25",    text:"text-rose-400",    dot:"bg-rose-500",    badge:"bg-rose-500/20 text-rose-400" },
  high:     { bg:"bg-amber-500/10",   border:"border-amber-500/25",   text:"text-amber-400",   dot:"bg-amber-500",   badge:"bg-amber-500/20 text-amber-400" },
  medium:   { bg:"bg-sky-500/10",     border:"border-sky-500/25",     text:"text-sky-400",     dot:"bg-sky-500",     badge:"bg-sky-500/20 text-sky-400" },
  low:      { bg:"bg-emerald-500/10", border:"border-emerald-500/25", text:"text-emerald-400", dot:"bg-emerald-500", badge:"bg-emerald-500/20 text-emerald-400" },
};
const TIMEFRAME_COLOR = {
  immediate:   "bg-rose-500/15 text-rose-400",
  "this week": "bg-amber-500/15 text-amber-400",
  "this month":"bg-sky-500/15 text-sky-400",
};
const INSIGHT_TYPE_ICON = {
  theme:       "🔍",
  churn_risk:  "⚠️",
  urgent:      "🚨",
  positive:    "✅",
  operational: "⚙️",
};

// ── Recharts tooltip ──────────────────────────────────────────────────────────
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/95 px-3 py-2 text-xs shadow-xl">
      {label && <p className="text-slate-400 mb-1 font-medium">{label}</p>}
      {payload.map((p) => (
        <p key={p.name} className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: p.fill || p.color }}/>
          <span className="text-slate-400">{p.name}:</span>
          <span className="font-bold text-white">{p.value}</span>
        </p>
      ))}
    </div>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, icon, accent = "#6366f1", trend, trendLabel }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 relative overflow-hidden">
      <div className="absolute top-0 inset-x-0 h-0.5" style={{ background: accent }}/>
      <div className="flex items-start justify-between mb-2">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{label}</p>
        <span className="text-lg leading-none">{icon}</span>
      </div>
      <p className="text-3xl font-black text-white tabular-nums">{value}</p>
      {sub && <p className="mt-0.5 text-[11px] text-slate-500">{sub}</p>}
      {trend != null && (
        <p className={`mt-1 text-[11px] font-semibold ${trend >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
          {trend >= 0 ? "↑" : "↓"} {Math.abs(trend)} {trendLabel || "this week"}
        </p>
      )}
    </div>
  );
}

// ── Issue bar ─────────────────────────────────────────────────────────────────
function IssueBar({ issue, data, total }) {
  const meta = ISSUE_META[issue] || ISSUE_META.general;
  const sev  = SEVERITY[data.severity] || SEVERITY.low;
  const pct  = total ? Math.round(data.count / total * 100) : 0;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span>{meta.icon}</span>
          <span className="font-semibold text-slate-300 capitalize">{issue}</span>
          {data.trending && (
            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400">↑ RISING</span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${sev.badge}`}>
            {data.severity}
          </span>
          <span className="text-slate-500 tabular-nums">{data.count} ({pct}%)</span>
        </div>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: meta.color }}/>
      </div>
      <div className="flex gap-3 text-[10px] text-slate-600">
        <span>Avg {data.avgRating?.toFixed(1)}★</span>
        <span>Sentiment {data.avgSentiment}%</span>
        {data.recentCount > 0 && <span className="text-amber-500">{data.recentCount} last 7d</span>}
      </div>
    </div>
  );
}

// ── Insight card ──────────────────────────────────────────────────────────────
function InsightCard({ insight }) {
  const sev    = SEVERITY[insight.severity] || SEVERITY.low;
  const typeIcon = INSIGHT_TYPE_ICON[insight.type] || "💡";

  return (
    <div className={`rounded-xl border p-4 ${sev.bg} ${sev.border}`}>
      <div className="flex items-start gap-3">
        <span className="text-lg leading-none flex-shrink-0 mt-0.5">{typeIcon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <p className={`text-xs font-bold ${sev.text} leading-snug`}>{insight.title}</p>
            <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${sev.badge}`}>
              {insight.severity}
            </span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">{insight.detail}</p>
          {insight.affectedCount > 0 && (
            <p className={`mt-1.5 text-[10px] font-semibold ${sev.text}`}>
              Affects {insight.affectedCount} review{insight.affectedCount !== 1 ? "s" : ""}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Recommendation card ───────────────────────────────────────────────────────
function RecommendationCard({ rec, index }) {
  const tfColor = TIMEFRAME_COLOR[rec.timeframe] || TIMEFRAME_COLOR["this month"];
  const priSev  = SEVERITY[rec.priority] || SEVERITY.low;

  return (
    <div className="flex gap-3 rounded-xl border border-slate-800 bg-slate-900 p-4">
      <div className={`flex-shrink-0 h-6 w-6 rounded-full flex items-center justify-center text-xs font-black ${priSev.badge}`}>
        {index + 1}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-white leading-snug">{rec.action}</p>
        {rec.impact && (
          <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">{rec.impact}</p>
        )}
        <div className="flex gap-2 mt-2">
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${tfColor}`}>
            {rec.timeframe}
          </span>
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${priSev.badge}`}>
            {rec.priority} priority
          </span>
        </div>
      </div>
    </div>
  );
}

// ── KPI pill ──────────────────────────────────────────────────────────────────
function KpiPill({ icon, label, value, color }) {
  return (
    <div className={`flex items-center gap-2 rounded-xl border p-3 ${color}`}>
      <span className="text-xl leading-none">{icon}</span>
      <div>
        <p className="text-lg font-black text-white tabular-nums">{value}</p>
        <p className="text-[10px] text-slate-500">{label}</p>
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard({ reviews = [] }) {
  const [snap,         setSnap]         = useState(null);
  const [live,         setLive]         = useState(false);
  const [age,          setAge]          = useState("");
  const [insights,     setInsights]     = useState(null);
  const [insightsLoad, setInsightsLoad] = useState(false);
  const [insightsErr,  setInsightsErr]  = useState("");
  const [activeTab,    setActiveTab]    = useState("overview");
  const fetchedRef = useRef(false);

  const load = useCallback(async () => {
    try { setSnap(await fetchAnalytics()); } catch {}
  }, []);

  useEffect(() => {
    load();
    const es = openAnalyticsStream();
    es.onopen    = () => setLive(true);
    es.onmessage = (e) => { try { setSnap(JSON.parse(e.data)); setLive(true); } catch {} };
    es.onerror   = () => setLive(false);
    return () => es.close();
  }, [load]);

  useEffect(() => {
    if (!snap?.computedAt) return;
    const tick = () => {
      const s = Math.round((Date.now() - new Date(snap.computedAt)) / 1000);
      setAge(s < 60 ? `${s}s ago` : `${Math.round(s/60)}m ago`);
    };
    tick();
    const t = setInterval(tick, 10_000);
    return () => clearInterval(t);
  }, [snap?.computedAt]);

  async function loadInsights() {
    if (fetchedRef.current || reviews.length === 0) return;
    fetchedRef.current = true;
    setInsightsLoad(true);
    setInsightsErr("");
    try {
      setInsights(await fetchInsights(reviews));
    } catch {
      setInsightsErr("Could not load insights. Is the AI service running on port 8000?");
    } finally {
      setInsightsLoad(false);
    }
  }

  async function refreshInsights() {
    fetchedRef.current = false;
    setInsights(null);
    await loadInsights();
  }

  function handleTab(t) {
    setActiveTab(t);
    if (t === "insights") loadInsights();
  }

  if (!snap) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900 p-12 text-center">
        <div className="inline-block h-6 w-6 rounded-full border-2 border-violet-500/30 border-t-violet-500 animate-spin mb-3"/>
        <p className="text-sm text-slate-500">Loading analytics…</p>
      </div>
    );
  }

  const ov = snap.overview || {};
  const sentPie   = Object.entries(snap.sentimentBreakdown  || {}).map(([name, value]) => ({ name, value }));
  const platBar   = Object.entries(snap.platformBreakdown   || {}).map(([name, count]) => ({ name, count }));
  const ratingBar = [1,2,3,4,5].map(r => ({ name:`${r}★`, count: snap.ratingDistribution?.[String(r)] || 0 }));
  const dailyLine = (snap.dailyVolume || []).map(d => ({
    date:     (d._id || d.date || "").slice(5),
    reviews:  d.count || 0,
    negative: d.negative || 0,
    positive: d.positive || 0,
  }));

  const issueBreakdown = insights?.stats?.issueBreakdown || {};
  const issueTotal     = insights?.stats?.totalReviews || reviews.length;

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 bg-slate-800/60 rounded-lg p-1">
          {[
            { id:"overview",  label:"Overview" },
            { id:"charts",    label:"Charts" },
            { id:"insights",  label:"AI Insights", special: true },
          ].map(({ id, label, special }) => (
            <button key={id} onClick={() => handleTab(id)}
              className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all
                ${activeTab === id
                  ? "bg-slate-700 text-white shadow"
                  : "text-slate-500 hover:text-slate-300"}`}>
              {special ? `✨ ${label}` : label}
            </button>
          ))}
        </div>
        <span className="text-[10px] text-slate-600 flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 rounded-full ${live ? "bg-emerald-500 animate-pulse" : "bg-slate-700"}`}/>
          {live ? "Live" : "Offline"} · {age}
        </span>
      </div>

      {/* ── OVERVIEW ──────────────────────────────────────────────────────── */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard label="Total Reviews"  value={ov.total || 0}                        icon="📊" accent="#6366f1" />
            <StatCard label="Avg Rating"     value={`${(ov.avgRating||0).toFixed(1)}★`}   icon="⭐" accent="#f59e0b" />
            <StatCard label="Avg Sentiment"  value={`${Math.round(ov.avgSentiment||0)}%`}  icon="🧠" accent="#10b981" />
            <StatCard label="Needs Reply"    value={ov.pendingReplies || 0}                icon="📭" accent="#f43f5e"
              sub={ov.inProgress ? `${ov.inProgress} in progress` : undefined}
              trend={snap.trend?.volumeDelta} trendLabel="reviews vs last week" />
          </div>

          <div className="grid grid-cols-3 gap-3">
            {[
              { label:"New",         val:ov.pendingReplies||0, accent:"#f43f5e", bg:"border-rose-500/20 bg-rose-500/5" },
              { label:"In Progress", val:ov.inProgress||0,     accent:"#f59e0b", bg:"border-amber-500/20 bg-amber-500/5" },
              { label:"Resolved",    val:ov.resolved||0,       accent:"#10b981", bg:"border-emerald-500/20 bg-emerald-500/5" },
            ].map(({ label, val, accent, bg }) => (
              <div key={label} className={`rounded-xl border p-4 ${bg}`}>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-2">{label}</p>
                <p className="text-3xl font-black text-white tabular-nums">{val}</p>
                <div className="mt-2 h-1 rounded-full bg-slate-800 overflow-hidden">
                  <div className="h-full rounded-full" style={{
                    width: `${ov.total ? Math.round(val/ov.total*100) : 0}%`,
                    background: accent,
                  }}/>
                </div>
                <p className="text-[10px] text-slate-600 mt-1">
                  {ov.total ? `${Math.round(val/(ov.total||1)*100)}% of total` : "—"}
                </p>
              </div>
            ))}
          </div>

          {snap.trend && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-xs font-semibold text-slate-400 mb-3">Week-over-week comparison</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 mb-1">Volume this week</p>
                  <p className="text-xl font-black text-white">{snap.trend.thisWeek?.count || 0}</p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 mb-1">Volume change</p>
                  <p className={`text-xl font-black ${(snap.trend.volumeDelta||0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {snap.trend.volumeDelta >= 0 ? "+" : ""}{snap.trend.volumeDelta}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 mb-1">Sentiment change</p>
                  <p className={`text-xl font-black ${(snap.trend.sentimentDelta||0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {snap.trend.sentimentDelta >= 0 ? "+" : ""}{snap.trend.sentimentDelta}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 mb-1">Negative this week</p>
                  <p className="text-xl font-black text-rose-400">{snap.trend.thisWeek?.negativeCount || 0}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── CHARTS ────────────────────────────────────────────────────────── */}
      {activeTab === "charts" && (
        <div className="grid lg:grid-cols-2 gap-4">
          {/* Sentiment donut */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs font-semibold text-slate-400 mb-3">Sentiment breakdown</p>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={sentPie} dataKey="value" outerRadius={75} innerRadius={40}>
                  {sentPie.map((e) => <Cell key={e.name} fill={SENT_COLOR[e.name] || "#6366f1"}/>)}
                </Pie>
                <RTooltip content={<ChartTip/>}/>
              </PieChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-4">
              {sentPie.map(e => (
                <span key={e.name} className="flex items-center gap-1.5 text-[10px] text-slate-400">
                  <span className="h-2 w-2 rounded-full" style={{ background: SENT_COLOR[e.name] || "#6366f1" }}/>
                  {e.name} · {e.value}
                </span>
              ))}
            </div>
          </div>

          {/* Platform bar */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs font-semibold text-slate-400 mb-3">Reviews by platform</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={platBar} margin={{ left:-20, right:4, top:0, bottom:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                <XAxis dataKey="name" tick={{ fill:"#64748b", fontSize:10 }} axisLine={false} tickLine={false}/>
                <YAxis tick={{ fill:"#64748b", fontSize:10 }} axisLine={false} tickLine={false}/>
                <RTooltip content={<ChartTip/>}/>
                <Bar dataKey="count" name="Reviews" radius={[4,4,0,0]}>
                  {platBar.map((_,i) => <Cell key={i} fill={PLAT_COLORS[i % PLAT_COLORS.length]}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Rating distribution */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs font-semibold text-slate-400 mb-3">Rating distribution</p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={ratingBar} margin={{ left:-20, right:4, top:0, bottom:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                <XAxis dataKey="name" tick={{ fill:"#64748b", fontSize:10 }} axisLine={false} tickLine={false}/>
                <YAxis tick={{ fill:"#64748b", fontSize:10 }} axisLine={false} tickLine={false}/>
                <RTooltip content={<ChartTip/>}/>
                <Bar dataKey="count" name="Reviews" radius={[4,4,0,0]}>
                  {ratingBar.map((e,i) => (
                    <Cell key={i} fill={i < 2 ? "#f43f5e" : i === 2 ? "#f59e0b" : "#10b981"}/>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Daily volume area chart */}
          {dailyLine.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-xs font-semibold text-slate-400 mb-3">14-day volume trend</p>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={dailyLine} margin={{ left:-20, right:4, top:4, bottom:0 }}>
                  <defs>
                    <linearGradient id="gTotal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="gNeg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#f43f5e" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                  <XAxis dataKey="date" tick={{ fill:"#64748b", fontSize:9 }} axisLine={false} tickLine={false}/>
                  <YAxis tick={{ fill:"#64748b", fontSize:9 }} axisLine={false} tickLine={false}/>
                  <RTooltip content={<ChartTip/>}/>
                  <Area type="monotone" dataKey="reviews"  name="Total"    stroke="#6366f1" fill="url(#gTotal)" strokeWidth={2}/>
                  <Area type="monotone" dataKey="negative" name="Negative" stroke="#f43f5e" fill="url(#gNeg)"   strokeWidth={1.5}/>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* ── AI INSIGHTS ───────────────────────────────────────────────────── */}
      {activeTab === "insights" && (
        <div className="space-y-4">
          {insightsLoad && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-12 text-center">
              <div className="inline-block h-8 w-8 rounded-full border-2 border-violet-500/30 border-t-violet-500 animate-spin mb-4"/>
              <p className="text-sm text-slate-300">Analysing reviews with AI…</p>
              <p className="text-xs text-slate-600 mt-1">
                Reading actual review language to find patterns keyword counting misses
              </p>
            </div>
          )}

          {insightsErr && !insightsLoad && (
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 text-sm text-rose-400">
              ⚠ {insightsErr}
            </div>
          )}

          {!insightsLoad && !insightsErr && !insights && reviews.length === 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-12 text-center">
              <p className="text-2xl mb-3">📊</p>
              <p className="text-sm text-slate-400">Load reviews first to generate AI insights.</p>
            </div>
          )}

          {insights && !insightsLoad && (
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-bold text-white">AI Insights</span>
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full
                    ${insights.generatedBy?.includes("anthropic")
                      ? "bg-indigo-500/15 text-indigo-400"
                      : "bg-slate-700 text-slate-400"}`}>
                    {insights.generatedBy?.includes("anthropic")
                      ? "✨ Claude NLP — reads actual review text"
                      : "📊 Keyword analysis — add ANTHROPIC_API_KEY for deeper insights"}
                  </span>
                </div>
                <button onClick={refreshInsights}
                  className="text-xs text-slate-500 hover:text-slate-300 border border-slate-700 rounded-lg px-3 py-1.5 hover:bg-slate-800 transition-colors">
                  ↻ Refresh
                </button>
              </div>

              {/* Top-line KPIs */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <KpiPill icon="⚠️" label="Churn risk signals"
                  value={insights.churnRiskCount ?? "—"}
                  color="border-rose-500/20 bg-rose-500/5"/>
                <KpiPill icon="🚨" label="Urgent (need reply)"
                  value={insights.urgentReplyCount ?? "—"}
                  color="border-amber-500/20 bg-amber-500/5"/>
                <KpiPill icon="📌" label="Top complaint theme"
                  value={insights.topComplaintTheme ? "↓" : "—"}
                  color="border-sky-500/20 bg-sky-500/5"/>
                <KpiPill icon="📝" label="Total analysed"
                  value={insights.stats?.totalReviews || reviews.length}
                  color="border-slate-700 bg-slate-900"/>
              </div>

              {insights.topComplaintTheme && (
                <div className="rounded-xl border border-sky-500/20 bg-sky-500/5 px-4 py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-sky-400 mb-1">Top Complaint Theme</p>
                  <p className="text-sm font-semibold text-white">{insights.topComplaintTheme}</p>
                </div>
              )}

              {/* Executive summary */}
              {insights.executiveSummary && (
                <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-violet-400 mb-2">Executive Summary</p>
                  <p className="text-sm text-slate-300 leading-relaxed">{insights.executiveSummary}</p>
                </div>
              )}

              {/* Issue estimator */}
              {Object.keys(issueBreakdown).length > 0 && (
                <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-xs font-semibold text-slate-400">Issue breakdown</p>
                    <p className="text-[10px] text-slate-600">{issueTotal} reviews analysed</p>
                  </div>
                  <div className="space-y-4">
                    {Object.entries(issueBreakdown)
                      .sort((a,b) => b[1].count - a[1].count)
                      .map(([issue, data]) => (
                        <IssueBar key={issue} issue={issue} data={data} total={issueTotal}/>
                      ))}
                  </div>
                </div>
              )}

              {/* Insight cards grouped by type */}
              {insights.insights?.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-slate-400">Key findings</p>
                  {/* Urgent / critical first */}
                  {["critical", "high", "medium", "low"].map(sev =>
                    insights.insights
                      .filter(i => i.severity === sev)
                      .map((ins, idx) => <InsightCard key={`${sev}-${idx}`} insight={ins}/>)
                  )}
                </div>
              )}

              {/* Recommendation cards */}
              {insights.recommendations?.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-slate-400">Recommendations</p>
                  {/* Sort: immediate first, then this week, then this month */}
                  {[...insights.recommendations]
                    .sort((a,b) => {
                      const order = { immediate:0, "this week":1, "this month":2 };
                      return (order[a.timeframe]||3) - (order[b.timeframe]||3);
                    })
                    .map((rec, i) => {
                      // Handle both string and object formats
                      const r = typeof rec === "string"
                        ? { action: rec, timeframe: "this week", priority: "medium", impact: "" }
                        : rec;
                      return <RecommendationCard key={i} rec={r} index={i}/>;
                    })}
                </div>
              )}

              <p className="text-[10px] text-slate-600 text-right">
                Generated {new Date(insights.generatedAt).toLocaleTimeString()}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}