const mongoose = require("mongoose");

const ReviewSchema = new mongoose.Schema(
  {
    customerName:   { type: String, required: true, trim: true },
    text:           { type: String, required: true, trim: true },
    rating:         { type: Number, required: true, min: 1, max: 5 },
    sentimentScore: { type: Number, default: 0 },
    sentimentLabel: { type: String, enum: ["Positive", "Neutral", "Negative"], default: "Neutral" },
    platform:       { type: String, default: "Other", trim: true },
    replyText:      { type: String, default: "", trim: true },
    issueCategory:  { type: String, default: "general", trim: true },
    priorityScore:  { type: Number, default: 0 },
    status:         { type: String, enum: ["new", "in_progress", "resolved"], default: "new" },
    date:           { type: Date, default: Date.now },
  },
  { timestamps: true }
);

// Auto-compute priority score before save
ReviewSchema.pre("save", function (next) {
  const sentimentWeight = 100 - (this.sentimentScore || 50);
  const ratingWeight    = (6 - (this.rating || 3)) * 10;
  this.priorityScore    = Math.min(150, Math.round(sentimentWeight + ratingWeight));
  next();
});

module.exports = mongoose.model("Review", ReviewSchema);