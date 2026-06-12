const express = require("express");
const { protect } = require("../middleware/auth");
const worker  = require("../workers/analyticsWorker");

const router = express.Router();
router.use(protect);

// GET /api/analytics/summary
router.get("/summary", (req, res) => {
  const snap = worker.getSnapshot();
  if (!snap) return res.status(503).json({ message: "Analytics not ready yet" });
  res.json(snap);
});

// GET /api/analytics/stream  (Server-Sent Events)
router.get("/stream", (req, res) => {
  res.setHeader("Content-Type",  "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection",    "keep-alive");
  res.flushHeaders();

  // Send current snapshot immediately
  const snap = worker.getSnapshot();
  if (snap) res.write(`data: ${JSON.stringify(snap)}\n\n`);

  // Forward future updates
  const handler = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);
  worker.events.on("update", handler);

  // Heartbeat every 30 s
  const hb = setInterval(() => res.write(": heartbeat\n\n"), 30_000);

  req.on("close", () => {
    worker.events.off("update", handler);
    clearInterval(hb);
  });
});

// GET /api/analytics/export  (CSV)
router.get("/export", async (req, res) => {
  try {
    const Review = require("../models/Review");
    const reviews = await Review.find().lean();
    const header = "ID,Customer,Rating,Sentiment,Score,Platform,Issue,Status,Date\n";
    const rows   = reviews.map((r) =>
      [
        r._id, `"${r.customerName}"`, r.rating,
        r.sentimentLabel, r.sentimentScore, r.platform,
        r.issueCategory, r.status,
        new Date(r.date).toISOString().split("T")[0],
      ].join(",")
    ).join("\n");
    res.setHeader("Content-Type",        "text/csv");
    res.setHeader("Content-Disposition", "attachment; filename=echoreview-export.csv");
    res.send(header + rows);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

module.exports = router;