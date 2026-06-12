import axios from "axios";

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000/api";
export const AI_BASE  = import.meta.env.VITE_AI_SERVICE_URL || "http://localhost:8000";

const TOKEN_KEY  = "echoreview_token";
export const getToken   = () => localStorage.getItem(TOKEN_KEY);
export const setToken   = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);
export const isLoggedIn = () => !!getToken();

const api = axios.create({ baseURL: API_BASE, headers: { "Content-Type": "application/json" }, timeout: 20_000 });

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) { clearToken(); window.location.href = "/"; }
    return Promise.reject(err);
  }
);

export default api;

// ── Auth ──────────────────────────────────────────────────────────────────────
export async function login(email, password) {
  const { data } = await api.post("/auth/login", { email, password });
  setToken(data.token);
  return data;
}
export function logout() { clearToken(); window.location.href = "/"; }
export const getMe = () => api.get("/auth/me").then(r => r.data.user);

// ── Reviews ───────────────────────────────────────────────────────────────────
export const fetchReviews  = (params) => api.get("/reviews", { params }).then(r => r.data);
export const createReview  = (body) => api.post("/reviews", body).then(r => r.data);
export const saveReply     = (id, replyText, issueCategory) =>
  api.patch(`/reviews/${id}/reply`, { replyText, issueCategory }).then(r => r.data);
export const updateStatus  = (id, status) =>
  api.patch(`/reviews/${id}/status`, { status }).then(r => r.data);
export const deleteReview  = (id) => api.delete(`/reviews/${id}`).then(r => r.data);

// ── Analytics ─────────────────────────────────────────────────────────────────
export const fetchAnalytics = () => api.get("/analytics/summary").then(r => r.data);
export function openAnalyticsStream() {
  return new EventSource(`${API_BASE}/analytics/stream?token=${getToken()}`);
}

// ── AI Reply — auto mode (FLAN → template → Anthropic) ───────────────────────
export async function generateAIReply(payload) {
  const res = await fetch(`${AI_BASE}/generate-reply`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`AI service ${res.status}`);
  return res.json();
}

// ── AI Reply — always template ────────────────────────────────────────────────
export async function generateTemplateReply(payload) {
  const res = await fetch(`${AI_BASE}/generate-reply/template`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Template service ${res.status}`);
  return res.json();
}

// ── AI Insights (batch analysis of reviews) ───────────────────────────────────
export async function fetchInsights(reviews) {
  const res = await fetch(`${AI_BASE}/insights`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews }),
  });
  if (!res.ok) throw new Error(`Insights service ${res.status}`);
  return res.json();
}

// ── Issue summary (fast keyword breakdown) ─────────────────────────────────────
export async function fetchIssuesSummary(reviews) {
  const res = await fetch(`${AI_BASE}/issues/summary`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews }),
  });
  if (!res.ok) throw new Error(`Issues service ${res.status}`);
  return res.json();
}

export async function analyzeSentiment(text) {
  const res = await fetch(`${AI_BASE}/analyze`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`AI service ${res.status}`);
  return res.json();
}

export const checkAIHealth = () =>
  fetch(`${AI_BASE}/health`).then(r => r.json()).catch(() => null);