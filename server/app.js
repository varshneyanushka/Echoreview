/**
 * EchoReview AI  –  Node API Gateway
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture
 *   React Frontend  →  Node API Gateway (this file)
 *                          ├── MongoDB  (via Mongoose)
 *                          └── Python AI Layer  (proxied via axios in routes)
 *
 * Routes
 *   POST   /api/auth/login          public
 *   POST   /api/auth/register       admin-only
 *   GET    /api/auth/me             protected
 *   GET    /api/reviews             protected
 *   POST   /api/reviews             protected
 *   PATCH  /api/reviews/:id/reply   protected
 *   PATCH  /api/reviews/:id/status  protected
 *   DELETE /api/reviews/:id         protected
 *   GET    /api/analytics/summary   protected
 *   GET    /api/analytics/stream    protected (SSE)
 *   GET    /api/analytics/export    protected (CSV)
 *   GET    /api/health              public
 *
 * env vars  (set in .env)
 *   PORT          default 5000
 *   MONGODB_URI   default mongodb://127.0.0.1:27017/echoreviewai
 *   JWT_SECRET    default dev secret — CHANGE IN PRODUCTION
 *   JWT_EXPIRES   default 7d
 *
 * npm install:
 *   express cors mongoose dotenv jsonwebtoken bcryptjs
 */

const express  = require("express");
const cors     = require("cors");
const mongoose = require("mongoose");
const path     = require("path");
const dotenv   = require("dotenv");

dotenv.config({ path: path.join(__dirname, "..", ".env") });

const authRoutes      = require("./routes/auth");
const reviewRoutes    = require("./routes/reviews");
const analyticsRoutes = require("./routes/analytics");
const worker          = require("./workers/analyticsWorker");

const app  = express();
const PORT = process.env.PORT || 5000;
const DB   = process.env.MONGODB_URI || process.env.MONGO_URI
           || "mongodb://127.0.0.1:27017/echoreviewai";
const CLIENT_ORIGINS = (process.env.CLIENT_ORIGINS || "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000")
  .split(",").map((origin) => origin.trim()).filter(Boolean);

// ── CORS ──────────────────────────────────────────────────────────────────────
app.use(cors({
  origin: CLIENT_ORIGINS,
  credentials: true,
}));

// ── Body parsing ──────────────────────────────────────────────────────────────
app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: true }));

// ── Request logging (dev) ─────────────────────────────────────────────────────
if (process.env.NODE_ENV !== "production") {
  app.use((req, _res, next) => {
    console.log(`${new Date().toISOString().slice(11,23)} ${req.method} ${req.path}`);
    next();
  });
}

// ── Routes ────────────────────────────────────────────────────────────────────
app.get("/api/health", (_req, res) => res.json({
  ok: true, service: "EchoReview AI Gateway", ts: new Date().toISOString(),
}));

app.use("/api/auth",      authRoutes);
app.use("/api/reviews",   reviewRoutes);
app.use("/api/analytics", analyticsRoutes);

// ── 404 ───────────────────────────────────────────────────────────────────────
app.use((_req, res) => res.status(404).json({ message: "Route not found" }));

// ── Global error handler ──────────────────────────────────────────────────────
// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  console.error("[Unhandled error]", err);
  res.status(500).json({ message: "Internal server error" });
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function bootstrap() {
  mongoose.set("strictQuery", true);
  await mongoose.connect(DB);
  console.log("✓ MongoDB connected");

  // Seed default admin if no users exist
  const User = require("./models/User");
  const count = await User.countDocuments();
  if (count === 0) {
    await User.create({
      name:     "Admin",
      email:    "admin@echoreview.ai",
      password: "Admin@123",
      role:     "admin",
    });
    console.log("✓ Default admin created  →  admin@echoreview.ai / Admin@123");
  }

  // Start analytics background worker
  worker.start();

  app.listen(PORT, () => {
    console.log(`✓ Server running on http://localhost:${PORT}`);
    console.log(`  Auth      → POST /api/auth/login`);
    console.log(`  Reviews   → GET  /api/reviews`);
    console.log(`  Analytics → GET  /api/analytics/summary`);
    console.log(`  SSE feed  → GET  /api/analytics/stream?token=<jwt>`);
  });
}

if (require.main === module) {
  bootstrap().catch((err) => { console.error("Bootstrap failed:", err); process.exit(1); });
}

module.exports = app;
