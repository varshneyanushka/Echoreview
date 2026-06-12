/**
 * analyticsWorker.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Background worker that computes analytics from MongoDB every REFRESH_MS ms.
 * Results are cached in memory and broadcast to all SSE subscribers.
 *
 * Usage:
 *   const worker = require("./workers/analyticsWorker");
 *   worker.start();                           // begin background loop
 *   const snap = worker.getSnapshot();        // latest cached result
 *   worker.events.on("update", (snap) => {}); // subscribe to live updates
 *   worker.trigger();                         // force immediate recompute
 */

const EventEmitter = require("events");
const Review       = require("../models/Review");

const REFRESH_MS = Number(process.env.ANALYTICS_REFRESH_MS) || 2 * 60 * 1000; // 2 min

const events   = new EventEmitter();
events.setMaxListeners(200);   // allow many SSE connections

let _snapshot  = null;
let _timer     = null;
let _computing = false;

// ── Core computation (MongoDB aggregation) ────────────────────────────────────
async function compute() {
  if (_computing) return;
  _computing = true;

  try {
    const now    = new Date();
    const day7   = new Date(now - 7  * 24 * 3600 * 1000);
    const day14  = new Date(now - 14 * 24 * 3600 * 1000);

    // ── 1. Overview (single aggregation pass) ─────────────────────────────────
    const [overview] = await Review.aggregate([
      {
        $group: {
          _id: null,
          total:          { $sum: 1 },
          avgRating:      { $avg: "$rating" },
          avgSentiment:   { $avg: "$sentimentScore" },
          pendingReplies: { $sum: { $cond: [{ $eq: ["$status", "new"] },         1, 0] } },
          inProgress:     { $sum: { $cond: [{ $eq: ["$status", "in_progress"] }, 1, 0] } },
          resolved:       { $sum: { $cond: [{ $eq: ["$status", "resolved"] },    1, 0] } },
          critical: {
            $sum: {
              $cond: [{ $and: [{ $gte: ["$priorityScore", 70] }, { $ne: ["$status", "resolved"] }] }, 1, 0],
            },
          },
        },
      },
    ]);

    // ── 2. Sentiment breakdown ────────────────────────────────────────────────
    const sentimentRaw = await Review.aggregate([
      { $group: { _id: "$sentimentLabel", count: { $sum: 1 } } },
    ]);
    const sentimentBreakdown = Object.fromEntries(
      sentimentRaw.map(({ _id, count }) => [_id || "Unknown", count])
    );

    // ── 3. Platform breakdown ─────────────────────────────────────────────────
    const platformRaw = await Review.aggregate([
      { $group: { _id: "$platform", count: { $sum: 1 } } },
      { $sort:  { count: -1 } },
    ]);
    const platformBreakdown = Object.fromEntries(
      platformRaw.map(({ _id, count }) => [_id || "Other", count])
    );

    // ── 4. Issue breakdown ────────────────────────────────────────────────────
    const issueRaw = await Review.aggregate([
      { $group: { _id: "$issueCategory", count: { $sum: 1 } } },
      { $sort:  { count: -1 } },
    ]);
    const issueBreakdown = Object.fromEntries(
      issueRaw.map(({ _id, count }) => [_id || "general", count])
    );

    // ── 5. Rating distribution ────────────────────────────────────────────────
    const ratingRaw = await Review.aggregate([
      { $group: { _id: "$rating", count: { $sum: 1 } } },
      { $sort:  { _id: 1 } },
    ]);
    const ratingDistribution = Object.fromEntries(
      ratingRaw.map(({ _id, count }) => [String(_id), count])
    );

    // ── 6. 7-day trend ────────────────────────────────────────────────────────
    const [thisWeek] = await Review.aggregate([
      { $match: { date: { $gte: day7 } } },
      {
        $group: {
          _id: null,
          count:        { $sum: 1 },
          avgSentiment: { $avg: "$sentimentScore" },
          avgRating:    { $avg: "$rating" },
          negative:     { $sum: { $cond: [{ $eq: ["$sentimentLabel", "Negative"] }, 1, 0] } },
        },
      },
    ]);
    const [lastWeek] = await Review.aggregate([
      { $match: { date: { $gte: day14, $lt: day7 } } },
      {
        $group: {
          _id: null,
          count:        { $sum: 1 },
          avgSentiment: { $avg: "$sentimentScore" },
        },
      },
    ]);

    // ── 7. Priority queue (top 10 unresolved) ─────────────────────────────────
    const priorityQueue = await Review.find({ status: { $ne: "resolved" } })
      .sort({ priorityScore: -1 })
      .limit(10)
      .select("customerName text rating platform sentimentLabel issueCategory priorityScore status date")
      .lean();

    // ── 8. Recent activity (last 8 for timeline) ──────────────────────────────
    const recentActivity = await Review.find()
      .sort({ createdAt: -1 })
      .limit(8)
      .select("customerName sentimentLabel rating platform issueCategory status createdAt")
      .lean();

    // ── 9. Daily volume (last 14 days) ────────────────────────────────────────
    const dailyVolume = await Review.aggregate([
      { $match: { date: { $gte: day14 } } },
      {
        $group: {
          _id: { $dateToString: { format: "%Y-%m-%d", date: "$date" } },
          count:    { $sum: 1 },
          negative: { $sum: { $cond: [{ $eq: ["$sentimentLabel", "Negative"] }, 1, 0] } },
          positive: { $sum: { $cond: [{ $eq: ["$sentimentLabel", "Positive"] }, 1, 0] } },
        },
      },
      { $sort: { _id: 1 } },
    ]);

    _snapshot = {
      computedAt: now.toISOString(),
      overview: {
        total:          overview?.total          ?? 0,
        avgRating:      +(overview?.avgRating    ?? 0).toFixed(2),
        avgSentiment:   +(overview?.avgSentiment ?? 0).toFixed(1),
        pendingReplies: overview?.pendingReplies ?? 0,
        inProgress:     overview?.inProgress     ?? 0,
        resolved:       overview?.resolved       ?? 0,
        critical:       overview?.critical       ?? 0,
      },
      sentimentBreakdown,
      platformBreakdown,
      issueBreakdown,
      ratingDistribution,
      trend: {
        thisWeek: {
          count:        thisWeek?.count        ?? 0,
          avgSentiment: +(thisWeek?.avgSentiment ?? 0).toFixed(1),
          avgRating:    +(thisWeek?.avgRating    ?? 0).toFixed(2),
          negativeCount: thisWeek?.negative      ?? 0,
        },
        lastWeek: {
          count:        lastWeek?.count        ?? 0,
          avgSentiment: +(lastWeek?.avgSentiment ?? 0).toFixed(1),
        },
        sentimentDelta: +(
          (thisWeek?.avgSentiment ?? 0) - (lastWeek?.avgSentiment ?? 0)
        ).toFixed(1),
        volumeDelta: (thisWeek?.count ?? 0) - (lastWeek?.count ?? 0),
      },
      priorityQueue,
      recentActivity,
      dailyVolume,
    };

    events.emit("update", _snapshot);

  } catch (err) {
    console.error("[Analytics Worker] Computation failed:", err.message);
  } finally {
    _computing = false;
  }
}

// ── Public API ────────────────────────────────────────────────────────────────
function start() {
  if (_timer) return;
  compute();                              // immediate first run
  _timer = setInterval(compute, REFRESH_MS);
  console.log(`[Analytics Worker] Started — refreshing every ${REFRESH_MS / 1000}s`);
}

function stop() {
  if (_timer) { clearInterval(_timer); _timer = null; }
}

function trigger() {
  // Call this when a review is created/updated to push a fresh snapshot
  setImmediate(compute);
}

function getSnapshot() {
  return _snapshot;
}

module.exports = { start, stop, trigger, getSnapshot, events };