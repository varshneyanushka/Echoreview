import axios from "axios";

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  "https://echoreview-api.onrender.com/api";

export const AI_BASE =
  import.meta.env.VITE_AI_SERVICE_URL ||
  "https://echoreview-ai.onrender.com";

const TOKEN_KEY = "echoreview_token";

// ─────────────────────────────────────────────────────────────
// Auth helpers
// ─────────────────────────────────────────────────────────────
export const getToken   = () => localStorage.getItem(TOKEN_KEY);
export const setToken   = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);
export const isLoggedIn = () => !!getToken();

// ─────────────────────────────────────────────────────────────
// Axios instance (backend API)
// ─────────────────────────────────────────────────────────────
const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
  timeout: 20000,
});

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      clearToken();
      window.location.href = "/";
    }
    return Promise.reject(err);
  }
);

export default api;

// ─────────────────────────────────────────────────────────────
// AUTH
// ─────────────────────────────────────────────────────────────
export async function login(email, password) {
  const { data } = await api.post("/auth/login", { email, password });
  setToken(data.token);
  return data;
}

export function logout() {
  clearToken();
  window.location.href = "/";
}

export const getMe = () =>
  api.get("/auth/me").then(r => r.data.user);

// ─────────────────────────────────────────────────────────────
// REVIEWS
// ─────────────────────────────────────────────────────────────
export const fetchReviews = (params) =>
  api.get("/reviews", { params }).then(r => r.data);

export const createReview = (body) =>
  api.post("/reviews", body).then(r => r.data);

export const saveReply = (id, replyText, issueCategory, replySource) =>
  api.patch(`/reviews/${id}/reply`, { replyText, issueCategory, replySource }).then(r => r.data);

export const updateStatus = (id, status) =>
  api.patch(`/reviews/${id}/status`, { status }).then(r => r.data);

export const deleteReview = (id) =>
  api.delete(`/reviews/${id}`).then(r => r.data);

// ─────────────────────────────────────────────────────────────
// ANALYTICS (core dashboard)
// ─────────────────────────────────────────────────────────────
export const fetchAnalytics = () =>
  api.get("/analytics/summary").then(r => r.data);

// live stream (SSE)
export function openAnalyticsStream() {
  return new EventSource(
    `${API_BASE}/analytics/stream?token=${getToken()}`
  );
}

// ─────────────────────────────────────────────────────────────
// AI REPLY ENGINE
// ─────────────────────────────────────────────────────────────

// The gateway supplies the review payload and keeps the AI service private.
// Its pipeline is Groq → template, and returns the source actually used.
export const generateAIReply = (reviewId) =>
  api.post(`/reviews/${reviewId}/generate-reply`).then(r => r.data);

// ─────────────────────────────────────────────────────────────
// AI INSIGHTS (Claude / ML pipeline)
// ─────────────────────────────────────────────────────────────
export async function fetchInsights(reviews) {
  const res = await fetch(`${AI_BASE}/insights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews }),
  });

  if (!res.ok) throw new Error(`Insights service ${res.status}`);
  return res.json();
}

// ─────────────────────────────────────────────────────────────
// 🧠 ML: SEMANTIC ISSUE MAP (PCA from embeddings)
// ─────────────────────────────────────────────────────────────
export async function fetchIssueMap(reviews) {
  const res = await fetch(`${AI_BASE}/issues/map`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews }),
  });

  if (!res.ok) {
    throw new Error(`Issue map service ${res.status}`);
  }

  return res.json();
}

// ─────────────────────────────────────────────────────────────
// 🧠 ML: SENTIMENT ANALYSIS (optional fine-grained endpoint)
// ─────────────────────────────────────────────────────────────
export async function analyzeSentiment(text) {
  const res = await fetch(`${AI_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) throw new Error(`Sentiment service ${res.status}`);
  return res.json();
}

// ─────────────────────────────────────────────────────────────
// FAST ISSUE SUMMARY (keyword fallback / non-ML baseline)
// ─────────────────────────────────────────────────────────────
export async function fetchIssuesSummary(reviews) {
  const res = await fetch(`${AI_BASE}/issues/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews }),
  });

  if (!res.ok) throw new Error(`Issues service ${res.status}`);
  return res.json();
}

// ─────────────────────────────────────────────────────────────
// HEALTH CHECKS
// ─────────────────────────────────────────────────────────────
export const checkAIHealth = () =>
  fetch(`${AI_BASE}/health`)
    .then(r => r.json())
    .catch(() => null);

export const checkClusterHealth = () =>
  fetch(`${AI_BASE}/cluster/health`)
    .then(r => r.json())
    .catch(() => null);
