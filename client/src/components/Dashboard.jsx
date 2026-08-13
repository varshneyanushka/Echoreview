/**
 * Dashboard.jsx  v6.0
 * ─────────────────────────────────────────────────────────────────────────────
 * Three tabs: Overview | Charts | AI Insights
 *
 * AI Insights improvements:
 *  • 2D PCA scatter plot (issue clusters) computed client-side via SVD
 *  • Anomaly detection timeline sparkline
 *  • Sentiment velocity gauge
 *  • Key findings cards — removed trailing "stable" word
 *  • Churn risk score card
 */

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer,
  AreaChart, Area,
  ScatterChart, Scatter, ZAxis,
  Tooltip as RTooltip,
} from "recharts";
import { fetchAnalytics, openAnalyticsStream, fetchInsights } from "../api";

// ── Palettes ──────────────────────────────────────────────────────────────────
const SENT_COLOR   = { Positive: "#10b981", Neutral: "#6366f1", Negative: "#f43f5e" };
const PLAT_COLORS  = ["#6366f1","#10b981","#f59e0b","#f43f5e","#0ea5e9","#a855f7","#14b8a6","#fb923c"];
const ISSUE_META   = {
  delivery: { color:"#f59e0b", icon:"🚚", label:"Delivery" },
  billing:  { color:"#f43f5e", icon:"💳", label:"Billing" },
  support:  { color:"#6366f1", icon:"🎧", label:"Support" },
  product:  { color:"#0ea5e9", icon:"📦", label:"Product" },
  refund:   { color:"#a855f7", icon:"↩️",  label:"Refund" },
  general:  { color:"#10b981", icon:"💬", label:"General" },
};
const CLUSTER_PALETTE = [
  "#6366f1","#f43f5e","#10b981","#f59e0b","#0ea5e9",
  "#a855f7","#14b8a6","#fb923c","#ec4899","#84cc16",
];
const SEVERITY = {
  critical: { bg:"bg-rose-500/10",    border:"border-rose-500/25",    text:"text-rose-400",    dot:"bg-rose-500",    badge:"bg-rose-500/20 text-rose-400" },
  high:     { bg:"bg-amber-500/10",   border:"border-amber-500/25",   text:"text-amber-400",   dot:"bg-amber-500",   badge:"bg-amber-500/20 text-amber-400" },
  medium:   { bg:"bg-sky-500/10",     border:"border-sky-500/25",     text:"text-sky-400",     dot:"bg-sky-500",     badge:"bg-sky-500/20 text-sky-400" },
  low:      { bg:"bg-emerald-500/10", border:"border-emerald-500/25", text:"text-emerald-400", dot:"bg-emerald-500", badge:"bg-emerald-500/20 text-emerald-400" },
};
const TIMEFRAME_COLOR = {
  immediate:    "bg-rose-500/15 text-rose-400",
  "this week":  "bg-amber-500/15 text-amber-400",
  "this month": "bg-sky-500/15 text-sky-400",
};
const INSIGHT_TYPE_ICON = {
  theme:       "🔍",
  churn_risk:  "⚠️",
  urgent:      "🚨",
  positive:    "✅",
  operational: "⚙️",
  issue:       "📌",
  risk:        "🔴",
};

// ──────────────────────────────────────────────────────────────────────────────
// PCA UTILITIES  (pure JS — no scipy / no extra deps)
// We compute a minimal PCA by centering + 2-iteration power-method SVD.
// Good enough for 2D projection of 5–10 dimensional feature vectors.
// ──────────────────────────────────────────────────────────────────────────────

/** Build a numeric feature vector for one review */
function reviewToVector(r) {
  const issueMap = { delivery:0, billing:1, support:2, product:3, refund:4, general:5 };
  const platMap  = { Google:0, Yelp:1, Trustpilot:2, "App Store":3, G2:4, Website:5, Facebook:6, Other:7 };
  const sentMap  = { Negative:0, Neutral:0.5, Positive:1 };

  const issueIdx = issueMap[r.issueCategory] ?? issueMap.general;
  const platIdx  = platMap[r.platform]       ?? platMap.Other;

  // 8-dimensional feature vector
  return [
    (r.sentimentScore ?? 50) / 100,          // [0] sentiment 0-1
    (r.rating         ??  3) / 5,            // [1] rating 0-1
    sentMap[r.sentimentLabel] ?? 0.5,        // [2] sentiment label ordinal
    issueIdx / 5,                            // [3] issue category (normalised)
    platIdx  / 7,                            // [4] platform (normalised)
    (r.priorityScore  ?? 50) / 150,          // [5] priority score
    r.replyText && r.replyText.length > 0 ? 1 : 0, // [6] has reply
    r.status === "resolved" ? 1 : r.status === "in_progress" ? 0.5 : 0, // [7] status
  ];
}

function mean(arr) { return arr.reduce((s,v)=>s+v, 0) / arr.length; }

function matMul(A, B) {
  // A: [m x n], B: [n x p] → [m x p]
  const m = A.length, n = A[0].length, p = B[0].length;
  return Array.from({length:m}, (_,i) =>
    Array.from({length:p}, (_,j) =>
      Array.from({length:n}, (_,k) => A[i][k] * B[k][j]).reduce((s,v)=>s+v,0)
    )
  );
}

function norm(v) { return Math.sqrt(v.reduce((s,x)=>s+x*x,0)); }
function normalize(v) { const n=norm(v); return n===0 ? v : v.map(x=>x/n); }

/** Compact 2-component PCA via power iteration */
function pca2D(matrix) {
  const m = matrix.length;
  if (m < 2) return matrix.map(()=>[0,0]);

  const dim = matrix[0].length;
  // 1. Center
  const means = Array.from({length:dim}, (_,j) => mean(matrix.map(r=>r[j])));
  const X = matrix.map(r => r.map((v,j) => v - means[j]));

  // 2. Covariance  C = Xᵀ X / (m-1)
  const C = Array.from({length:dim}, (_,i) =>
    Array.from({length:dim}, (_,j) =>
      X.reduce((s,r) => s + r[i]*r[j], 0) / (m-1)
    )
  );

  // 3. Power iteration for first 2 eigenvectors
  function powerIter(M, steps=50) {
    let v = Array.from({length:dim}, ()=>Math.random());
    v = normalize(v);
    for (let i=0;i<steps;i++) {
      v = normalize(M.map(row => row.reduce((s,c,k)=>s+c*v[k],0)));
    }
    return v;
  }

  const v1 = powerIter(C);
  // Deflate: C2 = C - λ1 v1 v1ᵀ
  const lam1 = v1.reduce((s,vi,i)=>s + vi * C[i].reduce((ss,c,j)=>ss+c*v1[j],0),0);
  const C2 = C.map((row,i) => row.map((c,j) => c - lam1*v1[i]*v1[j]));
  const v2 = powerIter(C2);

  // 4. Project
  return X.map(r => [
    r.reduce((s,v,k)=>s+v*v1[k],0),
    r.reduce((s,v,k)=>s+v*v2[k],0),
  ]);
}

/** Assign cluster ids via k-means on raw feature vectors (k=5 max) */
function kMeans(vectors, k=5, iters=20) {
  k = Math.min(k, vectors.length);
  // Init: pick k random centroids
  let centroids = vectors.slice(0,k).map(v=>[...v]);

  function dist(a,b){ return a.reduce((s,v,i)=>s+(v-b[i])**2,0); }

  let labels = new Array(vectors.length).fill(0);
  for (let it=0;it<iters;it++) {
    // Assign
    labels = vectors.map(v =>
      centroids.reduce((best,c,ci)=>dist(v,c)<dist(v,centroids[best])?ci:best, 0)
    );
    // Update
    centroids = Array.from({length:k}, (_,ci)=>{
      const pts = vectors.filter((_,i)=>labels[i]===ci);
      if (!pts.length) return centroids[ci];
      const dim = pts[0].length;
      return Array.from({length:dim}, (_,j)=>mean(pts.map(p=>p[j])));
    });
  }
  return labels;
}

/** Main: build PCA scatter data from reviews array */
function buildPCAData(reviews) {
  if (!reviews || reviews.length < 3) return [];
  const vecs   = reviews.map(reviewToVector);
  const coords = pca2D(vecs);
  const labels = kMeans(vecs, Math.min(6, Math.ceil(Math.sqrt(reviews.length))));

  return reviews.map((r, i) => ({
    x:       parseFloat(coords[i][0].toFixed(3)),
    y:       parseFloat(coords[i][1].toFixed(3)),
    cluster: labels[i],
    issue:   r.issueCategory || "general",
    sent:    r.sentimentLabel || "Neutral",
    rating:  r.rating || 3,
    name:    r.customerName || "—",
    text:    (r.text || "").slice(0, 80),
  }));
}

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

function PCADot({ cx, cy, payload }) {
  const meta  = ISSUE_META[payload?.issue] || ISSUE_META.general;
  const color = CLUSTER_PALETTE[payload?.cluster % CLUSTER_PALETTE.length] || "#6366f1";
  return (
    <circle
      cx={cx} cy={cy} r={5}
      fill={color} fillOpacity={0.85}
      stroke={SENT_COLOR[payload?.sent] || "#6366f1"}
      strokeWidth={1.5}
    />
  );
}

function PCATooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d   = payload[0].payload;
  const sev = SEVERITY[d.sent === "Negative" ? "critical" : d.sent === "Neutral" ? "medium" : "low"];
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/98 p-3 text-xs shadow-2xl max-w-xs">
      <p className="font-bold text-white mb-1">{d.name}</p>
      <p className="text-slate-400 mb-2 leading-relaxed">{d.text}…</p>
      <div className="flex gap-2 flex-wrap">
        <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${sev.badge}`}>{d.sent}</span>
        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-slate-700 text-slate-300 capitalize">{d.issue}</span>
        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-slate-700 text-slate-300">{d.rating}★</span>
        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-violet-500/20 text-violet-300">Cluster {d.cluster}</span>
      </div>
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
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${sev.badge}`}>{data.severity}</span>
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

// ── Insight card — strips trailing "stable" / trending noise from detail ──────
function InsightCard({ insight }) {
  const sev      = SEVERITY[insight.severity] || SEVERITY.low;
  const typeIcon = INSIGHT_TYPE_ICON[insight.type] || "💡";

  // Clean detail: remove trailing ", stable" or ", trending upward"
  const cleanDetail = (insight.detail || "")
    .replace(/,\s*(stable|trending upward|trending)\.?$/i, "")
    .trim();

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
          <p className="text-xs text-slate-400 leading-relaxed">{cleanDetail}</p>
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
        {rec.impact && <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">{rec.impact}</p>}
        <div className="flex gap-2 mt-2">
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${tfColor}`}>{rec.timeframe}</span>
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${priSev.badge}`}>{rec.priority} priority</span>
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

// ── Churn Risk Gauge (SVG arc) ────────────────────────────────────────────────
function ChurnGauge({ score = 0 }) {
  // score 0–100
  const r   = 48;
  const cx  = 64;
  const cy  = 64;
  const arc = (Math.min(score, 100) / 100) * Math.PI; // half-circle
  const x   = cx + r * Math.cos(Math.PI - arc);
  const y   = cy - r * Math.sin(Math.PI - arc);
  const color = score > 65 ? "#f43f5e" : score > 40 ? "#f59e0b" : "#10b981";
  const label = score > 65 ? "High" : score > 40 ? "Medium" : "Low";

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="128" height="72" viewBox="0 0 128 72">
        {/* Track */}
        <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`}
          fill="none" stroke="#1e293b" strokeWidth="10" strokeLinecap="round"/>
        {/* Fill */}
        {score > 0 && (
          <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 ${arc>Math.PI/2?1:0} 1 ${x} ${y}`}
            fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"/>
        )}
        {/* Needle dot */}
        <circle cx={x} cy={y} r={5} fill={color}/>
        {/* Value */}
        <text x={cx} y={cy+6} textAnchor="middle" fill="white" fontSize="16" fontWeight="900">{score}</text>
      </svg>
      <p className="text-[10px] font-bold" style={{color}}>{label} Churn Risk</p>
    </div>
  );
}

// ── Anomaly Sparkline ─────────────────────────────────────────────────────────
function AnomalySparkline({ data }) {
  if (!data?.length) return null;
  const w = 280, h = 48;
  const max = Math.max(...data.map(d=>d.anomalyScore), 1);
  const pts = data.map((d,i) => {
    const x = (i / (data.length-1)) * w;
    const y = h - (d.anomalyScore / max) * h;
    return `${x},${y}`;
  }).join(" ");
  const threshold = h - (0.6) * h; // 60% threshold line

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h+8}`} preserveAspectRatio="none" className="overflow-visible">
      {/* Threshold line */}
      <line x1={0} y1={threshold} x2={w} y2={threshold}
        stroke="#f43f5e" strokeWidth={1} strokeDasharray="4 3" opacity={0.5}/>
      {/* Sparkline */}
      <polyline points={pts} fill="none" stroke="#6366f1" strokeWidth={1.5}/>
      {/* Anomaly dots */}
      {data.map((d,i) => d.anomalyScore > 0.6 ? (
        <circle key={i}
          cx={(i/(data.length-1))*w}
          cy={h-(d.anomalyScore/max)*h}
          r={3.5} fill="#f43f5e" opacity={0.9}/>
      ) : null)}
    </svg>
  );
}

// ── PCA Scatter Chart ─────────────────────────────────────────────────────────
function PCAClusterChart({ reviews }) {
  const data   = useMemo(() => buildPCAData(reviews), [reviews]);
  const [hovered, setHovered] = useState(null);

  if (!data.length) return (
    <div className="flex items-center justify-center h-48 text-slate-600 text-xs">
      Not enough data for clustering
    </div>
  );

  // Group by cluster for legend
  const clusterIds = [...new Set(data.map(d=>d.cluster))].sort((a,b)=>a-b);
  // Group by cluster for Recharts (each cluster = separate <Scatter>)
  const byCluster = clusterIds.map(ci => data.filter(d=>d.cluster===ci));

  return (
    <div className="space-y-3">
      {/* Legend row */}
      <div className="flex flex-wrap gap-3">
        {clusterIds.map(ci => (
          <span key={ci} className="flex items-center gap-1.5 text-[10px] text-slate-400">
            <span className="h-2.5 w-2.5 rounded-full"
              style={{background: CLUSTER_PALETTE[ci % CLUSTER_PALETTE.length]}}/>
            Cluster {ci}
          </span>
        ))}
      </div>

      {/* Dot guide */}
      <div className="flex gap-4 text-[9px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full border-2 border-rose-500 bg-transparent inline-block"/> Negative sentiment
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full border-2 border-emerald-500 bg-transparent inline-block"/> Positive sentiment
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full border-2 border-indigo-500 bg-transparent inline-block"/> Neutral sentiment
        </span>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{left:0, right:4, top:8, bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
          <XAxis
            dataKey="x" type="number" name="PC1"
            tick={{fill:"#64748b", fontSize:9}} axisLine={false} tickLine={false}
            label={{value:"PC 1", position:"insideBottomRight", offset:-4, fill:"#475569", fontSize:9}}
          />
          <YAxis
            dataKey="y" type="number" name="PC2"
            tick={{fill:"#64748b", fontSize:9}} axisLine={false} tickLine={false}
            label={{value:"PC 2", angle:-90, position:"insideLeft", offset:12, fill:"#475569", fontSize:9}}
          />
          <ZAxis range={[40, 40]}/>
          <RTooltip content={<PCATooltip/>}/>
          {byCluster.map((pts, ci) => (
            <Scatter
              key={ci} data={pts}
              fill={CLUSTER_PALETTE[ci % CLUSTER_PALETTE.length]}
              shape={<PCADot/>}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>

      {/* Cluster summary table */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1">
        {clusterIds.map(ci => {
          const pts  = byCluster[ci] || [];
          const negPct = pts.length
            ? Math.round(pts.filter(p=>p.sent==="Negative").length / pts.length * 100)
            : 0;
          const topIssue = pts.length
            ? pts.reduce((acc,p)=>{ acc[p.issue]=(acc[p.issue]||0)+1; return acc; }, {})
            : {};
          const dominant = Object.entries(topIssue).sort((a,b)=>b[1]-a[1])[0]?.[0] || "—";
          const meta = ISSUE_META[dominant] || ISSUE_META.general;
          return (
            <div key={ci}
              className="rounded-lg border border-slate-800 bg-slate-900/60 p-2.5 text-[10px]">
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="h-2 w-2 rounded-full flex-shrink-0"
                  style={{background: CLUSTER_PALETTE[ci % CLUSTER_PALETTE.length]}}/>
                <span className="font-bold text-slate-300">Cluster {ci}</span>
                <span className="text-slate-600 ml-auto">{pts.length} reviews</span>
              </div>
              <div className="flex items-center gap-1 text-slate-500 mb-0.5">
                <span>{meta.icon}</span>
                <span className="capitalize">{dominant}</span>
              </div>
              <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
                <div className="h-full rounded-full bg-rose-500/70"
                  style={{width:`${negPct}%`}}/>
              </div>
              <p className="text-slate-600 mt-0.5">{negPct}% negative</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Sentiment Velocity ─────────────────────────────────────────────────────────
function SentimentVelocity({ reviews }) {
  const velocity = useMemo(() => {
    if (!reviews?.length) return null;
    const sorted = [...reviews].sort((a,b)=>new Date(a.date||0)-new Date(b.date||0));
    const half   = Math.floor(sorted.length / 2);
    if (half < 2) return null;
    const older  = sorted.slice(0, half);
    const newer  = sorted.slice(half);
    const avgOld = older.reduce((s,r)=>s+(r.sentimentScore||50),0)/older.length;
    const avgNew = newer.reduce((s,r)=>s+(r.sentimentScore||50),0)/newer.length;
    return { delta: Math.round(avgNew - avgOld), avgNew: Math.round(avgNew), avgOld: Math.round(avgOld) };
  }, [reviews]);

  if (!velocity) return null;
  const pos = velocity.delta >= 0;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-3">
        Sentiment Velocity
      </p>
      <div className="flex items-end gap-4">
        <div>
          <p className={`text-3xl font-black ${pos?"text-emerald-400":"text-rose-400"}`}>
            {pos?"+":""}{velocity.delta}
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">pts vs earlier half</p>
        </div>
        <div className="flex-1 space-y-1.5">
          <div className="flex justify-between text-[10px] text-slate-600">
            <span>Earlier avg: {velocity.avgOld}</span>
            <span>Recent avg: {velocity.avgNew}</span>
          </div>
          <div className="relative h-2 rounded-full bg-slate-800 overflow-hidden">
            <div className="absolute inset-y-0 left-0 rounded-full bg-slate-700"
              style={{width:`${velocity.avgOld}%`}}/>
            <div className={`absolute inset-y-0 left-0 rounded-full opacity-80 transition-all duration-700
              ${pos?"bg-emerald-500":"bg-rose-500"}`}
              style={{width:`${velocity.avgNew}%`}}/>
          </div>
        </div>
        <span className="text-xs font-bold text-slate-500">/ 100</span>
      </div>
      <p className="text-[10px] text-slate-500 mt-2">
        {pos
          ? "📈 Sentiment improving — keep current response strategy"
          : "📉 Sentiment declining — escalate issue resolution urgency"}
      </p>
    </div>
  );
}

// ── Churn risk computation ─────────────────────────────────────────────────────
function computeChurnRisk(reviews) {
  if (!reviews?.length) return 0;
  const negPct    = reviews.filter(r=>r.sentimentLabel==="Negative").length / reviews.length;
  const lowRating = reviews.filter(r=>(r.rating||3)<=2).length / reviews.length;
  const noReply   = reviews.filter(r=>!r.replyText && r.sentimentLabel==="Negative").length / reviews.length;
  const score = Math.round((negPct * 40) + (lowRating * 35) + (noReply * 25));
  return Math.min(100, score);
}

// ── Anomaly detection (simple z-score per day) ────────────────────────────────
function buildAnomalyData(reviews) {
  if (!reviews?.length) return [];
  const byDay = {};
  reviews.forEach(r => {
    const day = (r.date || r.createdAt || "").slice(0,10);
    if (!day) return;
    if (!byDay[day]) byDay[day] = [];
    byDay[day].push(r.sentimentScore || 50);
  });
  const days = Object.keys(byDay).sort();
  if (days.length < 3) return [];
  const dailyAvg = days.map(d => byDay[d].reduce((s,v)=>s+v,0)/byDay[d].length);
  const globalMean = mean(dailyAvg);
  const globalStd  = Math.sqrt(dailyAvg.reduce((s,v)=>s+(v-globalMean)**2,0)/dailyAvg.length) || 1;
  return days.map((d,i) => ({
    date:         d.slice(5),
    avgSentiment: Math.round(dailyAvg[i]),
    anomalyScore: Math.min(1, Math.abs(dailyAvg[i]-globalMean) / (globalStd*2)),
  }));
}

// ── Main Dashboard ─────────────────────────────────────────────────────────────
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

  const churnRisk   = useMemo(() => computeChurnRisk(reviews), [reviews]);
  const anomalyData = useMemo(() => buildAnomalyData(reviews), [reviews]);

  if (!snap) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900 p-12 text-center">
        <div className="inline-block h-6 w-6 rounded-full border-2 border-violet-500/30 border-t-violet-500 animate-spin mb-3"/>
        <p className="text-sm text-slate-500">Loading analytics…</p>
      </div>
    );
  }

  const ov = snap.overview || {};
  const sentPie   = Object.entries(snap.sentimentBreakdown || {}).map(([name,value])=>({name,value}));
  const platBar   = Object.entries(snap.platformBreakdown  || {}).map(([name,count])=>({name,count}));
  const ratingBar = [1,2,3,4,5].map(r=>({name:`${r}★`, count: snap.ratingDistribution?.[String(r)]||0}));
  const dailyLine = (snap.dailyVolume || []).map(d=>({
    date:     (d._id||d.date||"").slice(5),
    reviews:  d.count   || 0,
    negative: d.negative|| 0,
    positive: d.positive|| 0,
  }));

  const issueBreakdown = insights?.stats?.issueBreakdown || {};
  const issueTotal     = insights?.stats?.totalReviews   || reviews.length;

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
          <span className={`h-1.5 w-1.5 rounded-full ${live?"bg-emerald-500 animate-pulse":"bg-slate-700"}`}/>
          {live?"Live":"Offline"} · {age}
        </span>
      </div>

      {/* ── OVERVIEW ──────────────────────────────────────────────────────── */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard label="Total Reviews"  value={ov.total||0}                        icon="📊" accent="#6366f1"/>
            <StatCard label="Avg Rating"     value={`${(ov.avgRating||0).toFixed(1)}★`}  icon="⭐" accent="#f59e0b"/>
            <StatCard label="Avg Sentiment"  value={`${Math.round(ov.avgSentiment||0)}%`} icon="🧠" accent="#10b981"/>
            <StatCard label="Needs Reply"    value={ov.pendingReplies||0}                icon="📭" accent="#f43f5e"
              sub={ov.inProgress ? `${ov.inProgress} in progress` : undefined}
              trend={snap.trend?.volumeDelta} trendLabel="reviews vs last week"/>
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
                  <div className="h-full rounded-full"
                    style={{ width:`${ov.total?Math.round(val/(ov.total||1)*100):0}%`, background: accent }}/>
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
                  <p className="text-xl font-black text-white">{snap.trend.thisWeek?.count||0}</p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 mb-1">Volume change</p>
                  <p className={`text-xl font-black ${(snap.trend.volumeDelta||0)>=0?"text-emerald-400":"text-rose-400"}`}>
                    {snap.trend.volumeDelta>=0?"+":""}{snap.trend.volumeDelta}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 mb-1">Sentiment change</p>
                  <p className={`text-xl font-black ${(snap.trend.sentimentDelta||0)>=0?"text-emerald-400":"text-rose-400"}`}>
                    {snap.trend.sentimentDelta>=0?"+":""}{snap.trend.sentimentDelta}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 mb-1">Negative this week</p>
                  <p className="text-xl font-black text-rose-400">{snap.trend.thisWeek?.negativeCount||0}</p>
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
                  {sentPie.map((e)=><Cell key={e.name} fill={SENT_COLOR[e.name]||"#6366f1"}/>)}
                </Pie>
                <RTooltip content={<ChartTip/>}/>
              </PieChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-4">
              {sentPie.map(e=>(
                <span key={e.name} className="flex items-center gap-1.5 text-[10px] text-slate-400">
                  <span className="h-2 w-2 rounded-full" style={{background:SENT_COLOR[e.name]||"#6366f1"}}/>
                  {e.name} · {e.value}
                </span>
              ))}
            </div>
          </div>

          {/* Platform bar */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs font-semibold text-slate-400 mb-3">Reviews by platform</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={platBar} margin={{left:-20, right:4, top:0, bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                <XAxis dataKey="name" tick={{fill:"#64748b", fontSize:10}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:"#64748b", fontSize:10}} axisLine={false} tickLine={false}/>
                <RTooltip content={<ChartTip/>}/>
                <Bar dataKey="count" name="Reviews" radius={[4,4,0,0]}>
                  {platBar.map((_,i)=><Cell key={i} fill={PLAT_COLORS[i%PLAT_COLORS.length]}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Rating distribution */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs font-semibold text-slate-400 mb-3">Rating distribution</p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={ratingBar} margin={{left:-20, right:4, top:0, bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                <XAxis dataKey="name" tick={{fill:"#64748b", fontSize:10}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:"#64748b", fontSize:10}} axisLine={false} tickLine={false}/>
                <RTooltip content={<ChartTip/>}/>
                <Bar dataKey="count" name="Reviews" radius={[4,4,0,0]}>
                  {ratingBar.map((e,i)=>(
                    <Cell key={i} fill={i<2?"#f43f5e":i===2?"#f59e0b":"#10b981"}/>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Daily volume area */}
          {dailyLine.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-xs font-semibold text-slate-400 mb-3">14-day volume trend</p>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={dailyLine} margin={{left:-20, right:4, top:4, bottom:0}}>
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
                  <XAxis dataKey="date" tick={{fill:"#64748b", fontSize:9}} axisLine={false} tickLine={false}/>
                  <YAxis tick={{fill:"#64748b", fontSize:9}} axisLine={false} tickLine={false}/>
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
              <p className="text-xs text-slate-600 mt-1">Reading actual review language to find patterns keyword counting misses</p>
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

          {/* ─── ML Signals (always visible once reviews are loaded) ─────── */}
          {!insightsLoad && reviews.length > 0 && (
            <div className="space-y-4">

              {/* Row 1: Churn gauge + Sentiment velocity */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Churn risk */}
                <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 flex flex-col items-center">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-3 self-start">
                    Churn Risk Score
                  </p>
                  <ChurnGauge score={churnRisk}/>
                  <p className="text-[10px] text-slate-500 mt-2 text-center">
                    Weighted: negative rate × low ratings × unreplied negatives
                  </p>
                </div>

                {/* Sentiment velocity */}
                <SentimentVelocity reviews={reviews}/>
              </div>

              

              {/* Row 3: PCA Cluster Map */}
              <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-xs font-semibold text-slate-400">Issue Cluster Map (PCA)</p>
                  <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400">
                    k-means · 8-dim features
                  </span>
                </div>
                <p className="text-[10px] text-slate-600 mb-4">
                  Each dot = one review. Color = ML cluster. Border = sentiment. Axes = principal components of sentiment, rating, issue category &amp; platform.
                </p>
                <PCAClusterChart reviews={reviews}/>
              </div>
            </div>
          )}

          {/* ─── Sentiment, fault and clustering insights ──────────────── */}
          {insights && !insightsLoad && (
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-bold text-white">AI Insights</span>
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-400">
                    🧠 Weighted sentiment · fault detection · pattern clustering
                  </span>
                </div>
                <button onClick={refreshInsights}
                  className="text-xs text-slate-500 hover:text-slate-300 border border-slate-700 rounded-lg px-3 py-1.5 hover:bg-slate-800 transition-colors">
                  ↻ Refresh
                </button>
              </div>

              {insights.topComplaintTheme && (
                <div className="rounded-xl border border-sky-500/20 bg-sky-500/5 px-4 py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-sky-400 mb-1">Top Complaint Theme</p>
                  <p className="text-sm font-semibold text-white">{insights.topComplaintTheme}</p>
                </div>
              )}

              {insights.executiveSummary && (
                <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-violet-400 mb-2">Executive Summary</p>
                  <p className="text-sm text-slate-300 leading-relaxed">{insights.executiveSummary}</p>
                </div>
              )}

              {insights.faultPatterns?.length > 0 && (
                <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-rose-400 mb-2">Fault signals requiring attention</p>
                  <div className="space-y-1.5">
                    {insights.faultPatterns.slice(0, 4).map((fault) => (
                      <p key={fault.fault} className="text-xs text-slate-300">
                        <span className="font-semibold text-rose-300">{fault.title}</span>
                        {` — ${fault.detail}`}
                      </p>
                    ))}
                  </div>
                </div>
              )}

              {insights.clusters?.length > 0 && (
                <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">Review patterns</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {insights.clusters.slice(0, 6).map((cluster) => (
                      <div key={cluster.clusterId} className="rounded-lg bg-slate-800/70 px-3 py-2">
                        <p className="text-xs font-semibold text-slate-200">{cluster.clusterName}</p>
                        <p className="text-[10px] text-slate-500">{cluster.clusterSummary}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Key findings */}
              {insights.insights?.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-slate-400">Key findings</p>
                  {["critical","high","medium","low"].map(sev =>
                    insights.insights
                      .filter(i => i.severity === sev)
                      .map((ins, idx) => <InsightCard key={`${sev}-${idx}`} insight={ins}/>)
                  )}
                </div>
              )}

              {/* Recommendations */}
              {insights.recommendations?.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-slate-400">Recommendations</p>
                  {[...insights.recommendations]
                    .sort((a,b)=>{
                      const order={immediate:0,"this week":1,"this month":2};
                      return (order[a.timeframe]||3)-(order[b.timeframe]||3);
                    })
                    .map((rec,i) => {
                      const r = typeof rec==="string"
                        ? {action:rec, timeframe:"this week", priority:"medium", impact:""}
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
