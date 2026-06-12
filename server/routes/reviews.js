const express = require("express");
const Review  = require("../models/Review");
const { protect } = require("../middleware/auth");
const worker  = require("../workers/analyticsWorker");

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

// PATCH /api/reviews/:id/reply
router.patch("/:id/reply", async (req, res) => {
  try {
    const { replyText, issueCategory } = req.body;
    const review = await Review.findByIdAndUpdate(
      req.params.id,
      { replyText, issueCategory, status: "in_progress" },
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