const express = require("express");
const Review  = require("../models/Review");
const { protect } = require("../middleware/auth");
const worker  = require("../workers/analyticsWorker");

const AI_SERVICE_URL = (process.env.AI_SERVICE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const AI_TIMEOUT_MS = Number(process.env.AI_SERVICE_TIMEOUT_MS || 9000);

function templateReply(review) {
  const name = (review.customerName || "Customer").trim().split(/\s+/)[0] || "Customer";
  const rating = Number(review.rating || 3);
  const issue = (review.issueCategory || "general").toLowerCase();

  if (rating <= 2) {
    const topic = issue !== "general" ? ` regarding ${issue}` : "";
    return `Hi ${name}, we are sorry to hear about your experience${topic}. We have shared your feedback with the relevant team and will look into this carefully. Please contact our support team directly so we can help resolve this.`;
  }
  if (rating >= 4) {
    return `Hi ${name}, thank you for your kind feedback. We are delighted to hear about your positive experience and truly appreciate your support.`;
  }
  return `Hi ${name}, thank you for sharing your feedback. We have noted your comments and will use them to improve your experience going forward.`;
}

const router = express.Router();
router.use(protect);

// GET /api/reviews
router.get("/", async (req, res) => {
  try {
    const { status, platform, sentiment, search, sort = "-date", limit = 200 } = req.query;
    const filter = {};
    if (status)    filter.status         = status;
    if (platform)  filter.platform       = platform;
    if (sentiment) filter.sentimentLabel = sentiment;
    if (search)    filter.$or = [
      { customerName: { $regex: search, $options: "i" } },
      { text:         { $regex: search, $options: "i" } },
    ];

    const reviews = await Review.find(filter)
      .sort(sort)
      .limit(Number(limit))
      .lean();
    res.json(reviews);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// POST /api/reviews
router.post("/", async (req, res) => {
  try {
    const review = await Review.create(req.body);
    worker.trigger();
    res.status(201).json(review);
  } catch (err) {
    res.status(400).json({ message: err.message });
  }
});

// Generate server-side so API keys stay only in the Python service and the
// browser never needs a public AI_SERVICE_URL/CORS exception.
router.post("/:id/generate-reply", async (req, res) => {
  let review;
  try {
    review = await Review.findById(req.params.id).lean();
    if (!review) return res.status(404).json({ message: "Review not found" });

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), AI_TIMEOUT_MS);
    let upstream;
    try {
      console.log(`[AI reply] forwarding ${review._id} to ${AI_SERVICE_URL}/generate-reply`);
      upstream = await fetch(`${AI_SERVICE_URL}/generate-reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          customerName: review.customerName,
          text: review.text,
          rating: review.rating,
          platform: review.platform,
          sentimentScore: review.sentimentScore,
          sentimentLabel: review.sentimentLabel,
        }),
      });
    } finally {
      clearTimeout(timer);
    }

    if (!upstream.ok) throw new Error(`AI service returned HTTP ${upstream.status}`);
    const result = await upstream.json();
    if (!result.reply) throw new Error("AI service returned no reply");
    res.json(result);
  } catch (err) {
    // The gateway fallback is deliberately independent of the AI service. It
    // guarantees users always receive an editable reply during cold starts,
    // provider outages, or a local AI-service configuration mistake.
    console.warn(`[AI reply] ${err.message}; using gateway template fallback`);
    if (!review) return res.status(500).json({ message: "Could not load review" });
    res.json({
      reply: templateReply(review),
      issueCategory: review.issueCategory || "general",
      source: "template",
      metadata: { fallbackReason: "ai-service-unavailable" },
    });
  }
});

// PATCH /api/reviews/:id/reply
router.patch("/:id/reply", async (req, res) => {
  try {
    const { replyText, issueCategory, replySource } = req.body;
    const review = await Review.findByIdAndUpdate(
      req.params.id,
      { replyText, issueCategory, replySource: replySource || "manual", status: "in_progress" },
      { new: true, runValidators: true }
    );
    if (!review) return res.status(404).json({ message: "Review not found" });
    worker.trigger();
    res.json(review);
  } catch (err) {
    res.status(400).json({ message: err.message });
  }
});

// PATCH /api/reviews/:id/status
router.patch("/:id/status", async (req, res) => {
  try {
    const { status } = req.body;
    const review = await Review.findByIdAndUpdate(
      req.params.id,
      { status },
      { new: true, runValidators: true }
    );
    if (!review) return res.status(404).json({ message: "Review not found" });
    worker.trigger();
    res.json(review);
  } catch (err) {
    res.status(400).json({ message: err.message });
  }
});

// DELETE /api/reviews/:id
router.delete("/:id", async (req, res) => {
  try {
    const review = await Review.findByIdAndDelete(req.params.id);
    if (!review) return res.status(404).json({ message: "Review not found" });
    worker.trigger();
    res.json({ message: "Deleted" });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

module.exports = router;
