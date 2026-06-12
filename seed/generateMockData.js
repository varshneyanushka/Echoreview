/**
 * seed/generateMockData.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Seeds MongoDB with 40 realistic, varied customer reviews.
 *
 * Usage:
 *   node seed/generateMockData.js
 *
 * Requires MONGODB_URI in .env (or uses the local default).
 */

const mongoose = require("mongoose");
const path     = require("path");
const dotenv   = require("dotenv");

dotenv.config({ path: path.join(__dirname, "..", ".env") });

const MONGODB_URI =
  process.env.MONGODB_URI ||
  process.env.MONGO_URI   ||
  "mongodb://127.0.0.1:27017/echoreviewai";

// ── Inline Review schema (avoids relative-path issues) ──────────────────────
const ReviewSchema = new mongoose.Schema(
  {
    customerName:   { type: String, required: true },
    text:           { type: String, required: true },
    rating:         { type: Number, required: true, min: 1, max: 5 },
    sentimentScore: { type: Number, default: 0 },
    sentimentLabel: { type: String, enum: ["Positive", "Neutral", "Negative"], default: "Neutral" },
    platform:       { type: String, default: "Other" },
    replyText:      { type: String, default: "" },
    issueCategory:  { type: String, default: "general" },
    priorityScore:  { type: Number, default: 0 },
    status:         { type: String, enum: ["new", "in_progress", "resolved"], default: "new" },
    date:           { type: Date, default: Date.now },
  },
  { timestamps: true }
);
const Review = mongoose.model("Review", ReviewSchema);

// ── Mock data ─────────────────────────────────────────────────────────────────
const MOCK_REVIEWS = [
  // ── NEGATIVE ─────────────────────────────────────────────────────────────
  {
    customerName: "Priya Sharma",
    text: "My order #ORD-8821 was supposed to arrive within 3 days but it's been 3 weeks and I still haven't received anything. The tracking page says 'in transit' since day 1. Customer support hasn't replied to any of my 4 emails. Completely unacceptable.",
    rating: 1, sentimentScore: 8, sentimentLabel: "Negative",
    platform: "Google", issueCategory: "delivery", priorityScore: 132, status: "new",
  },
  {
    customerName: "Rohit Verma",
    text: "I was charged twice for the same subscription — ₹4,999 appeared twice on my bank statement dated 2nd June. I've raised a ticket (TICK-2204) and nobody has gotten back to me. I want a full refund of ₹4,999 immediately.",
    rating: 1, sentimentScore: 5, sentimentLabel: "Negative",
    platform: "Trustpilot", issueCategory: "billing", priorityScore: 135, status: "new",
  },
  {
    customerName: "Anjali Mehta",
    text: "The laptop stand I ordered arrived completely shattered. The packaging was intact so it must have been packaged already broken. Quality control is terrible. This was a gift and now I'm embarrassed. Requesting a replacement or refund immediately.",
    rating: 1, sentimentScore: 12, sentimentLabel: "Negative",
    platform: "App Store", issueCategory: "product", priorityScore: 129, status: "new",
  },
  {
    customerName: "Sanjay Gupta",
    text: "Your customer service representative was extremely rude during my call on 3rd June. I called about a billing discrepancy and the agent put me on hold for 40 minutes, then disconnected the call without resolving anything. I've been a customer for 5 years and never been treated this way.",
    rating: 2, sentimentScore: 18, sentimentLabel: "Negative",
    platform: "Google", issueCategory: "support", priorityScore: 112, status: "new",
  },
  {
    customerName: "Fatima Khan",
    text: "I requested a refund 3 weeks ago after returning the defective mixer. I have the return tracking number (RET-5503) confirming delivery to your warehouse but the refund hasn't been processed. I've emailed 3 times and called twice with no resolution.",
    rating: 1, sentimentScore: 7, sentimentLabel: "Negative",
    platform: "Yelp", issueCategory: "refund", priorityScore: 133, status: "new",
  },
  {
    customerName: "Kiran Nair",
    text: "The wireless earbuds stopped working after exactly 6 days. One earbud just went silent. These cost ₹6,499 and I expected them to last at least a year. The product description said '2-year warranty' but when I tried to claim it, I was told the issue was not covered. That's misleading advertising.",
    rating: 2, sentimentScore: 15, sentimentLabel: "Negative",
    platform: "G2", issueCategory: "product", priorityScore: 115, status: "new",
  },
  {
    customerName: "Deepak Joshi",
    text: "Three days after placing my order I got a notification saying my item was cancelled — no explanation, no alternative offered, no apology email. I had to find out by logging in. This is terrible communication and I've had to scramble to find the item elsewhere for my event tomorrow.",
    rating: 2, sentimentScore: 20, sentimentLabel: "Negative",
    platform: "Website", issueCategory: "delivery", priorityScore: 110, status: "new",
  },
  {
    customerName: "Sneha Pillai",
    text: "I've been trying to cancel my subscription for 2 months. The cancel button on the website doesn't work — it just redirects to the homepage. I emailed support and was told to call. I called and was told to email. I'm still being charged ₹899/month for a service I don't use.",
    rating: 1, sentimentScore: 9, sentimentLabel: "Negative",
    platform: "Trustpilot", issueCategory: "billing", priorityScore: 131, status: "new",
  },

  // ── MIXED / NEUTRAL ───────────────────────────────────────────────────────
  {
    customerName: "Amit Patel",
    text: "The product itself is decent for the price. However delivery took 12 days instead of the promised 5. I was also overcharged by ₹150 compared to the checkout price — small amount but it's the principle. Would appreciate if these operational issues were fixed.",
    rating: 3, sentimentScore: 45, sentimentLabel: "Neutral",
    platform: "Google", issueCategory: "delivery", priorityScore: 75, status: "new",
  },
  {
    customerName: "Ritu Singh",
    text: "Support eventually resolved my issue but it took 9 days and 6 follow-up messages to get there. The solution itself was good but the process of getting help was exhausting. Hope you improve response times.",
    rating: 3, sentimentScore: 50, sentimentLabel: "Neutral",
    platform: "App Store", issueCategory: "support", priorityScore: 70, status: "new",
  },
  {
    customerName: "Vikram Bose",
    text: "App works fine on iOS but crashes consistently on Android when I try to open the reports section. I've reinstalled it twice. Please fix the Android version — the iOS experience is great though.",
    rating: 3, sentimentScore: 48, sentimentLabel: "Neutral",
    platform: "App Store", issueCategory: "product", priorityScore: 72, status: "new",
  },
  {
    customerName: "Meera Iyer",
    text: "The quality of the material is okay but not what I expected from the photos. The colour looks quite different in real life. Delivery was fast though which I appreciated. Probably won't order again without a better photo.",
    rating: 3, sentimentScore: 42, sentimentLabel: "Neutral",
    platform: "Website", issueCategory: "product", priorityScore: 78, status: "new",
  },

  // ── POSITIVE ─────────────────────────────────────────────────────────────
  {
    customerName: "Rahul Dubey",
    text: "Absolutely brilliant service from start to finish. Placed my order on Monday, arrived Tuesday morning — couldn't believe the speed. The packaging was premium and the product exceeded expectations. Will definitely be a repeat customer.",
    rating: 5, sentimentScore: 96, sentimentLabel: "Positive",
    platform: "Google", issueCategory: "general", priorityScore: 0, status: "new",
  },
  {
    customerName: "Kavya Reddy",
    text: "I had an issue with my order and Arjun from customer service sorted it out within the hour. He was incredibly patient, proactive, and followed up to make sure everything was resolved. This is how customer service should be done. 10/10.",
    rating: 5, sentimentScore: 97, sentimentLabel: "Positive",
    platform: "Trustpilot", issueCategory: "support", priorityScore: 0, status: "resolved",
  },
  {
    customerName: "Suresh Krishnan",
    text: "Best purchase I've made this year. The build quality is exceptional, the setup took under 10 minutes, and performance has been flawless for the past 3 months. Worth every rupee. Highly recommend to anyone looking for reliability.",
    rating: 5, sentimentScore: 98, sentimentLabel: "Positive",
    platform: "G2", issueCategory: "general", priorityScore: 0, status: "new",
  },
  {
    customerName: "Pooja Agarwal",
    text: "Genuinely impressed by how smoothly the whole experience went. Easy checkout, real-time tracking, delivered a day early, and the product was exactly as described. Refreshing to have an online shopping experience that just works.",
    rating: 5, sentimentScore: 94, sentimentLabel: "Positive",
    platform: "Facebook", issueCategory: "general", priorityScore: 0, status: "new",
  },
  {
    customerName: "Manoj Tiwari",
    text: "My refund was processed in under 48 hours after I returned the item. No questions asked, no friction. That kind of hassle-free return policy builds real trust. I'll keep shopping here because I know if something's wrong, it'll be made right.",
    rating: 5, sentimentScore: 92, sentimentLabel: "Positive",
    platform: "Website", issueCategory: "refund", priorityScore: 0, status: "resolved",
  },
  {
    customerName: "Nisha Choudhary",
    text: "Outstanding product quality and super fast delivery. I've ordered 4 times now and every experience has been consistently excellent. The app is also very intuitive. This company clearly cares about the customer experience.",
    rating: 5, sentimentScore: 95, sentimentLabel: "Positive",
    platform: "App Store", issueCategory: "general", priorityScore: 0, status: "new",
  },

  // ── IN PROGRESS (already replied) ────────────────────────────────────────
  {
    customerName: "Arjun Malhotra",
    text: "The courier marked my package as delivered but it never arrived. My neighbours don't have it either. This is the second time this has happened with the same courier partner. Very disappointed.",
    rating: 2, sentimentScore: 18, sentimentLabel: "Negative",
    platform: "Google", issueCategory: "delivery", priorityScore: 112,
    status: "in_progress",
    replyText: "Hi Arjun, thank you for letting us know — a package marked delivered but not received is completely unacceptable, and we are sorry this has happened to you again. We have escalated this directly to our logistics partner and a replacement has been dispatched on priority today. Please expect delivery within 24 hours. We have flagged your account to avoid the same courier on future orders.",
  },
  {
    customerName: "Sunita Yadav",
    text: "Was charged ₹2,999 for an annual plan I never signed up for. I only wanted the monthly ₹299 plan. Please look into this urgently.",
    rating: 1, sentimentScore: 10, sentimentLabel: "Negative",
    platform: "Trustpilot", issueCategory: "billing", priorityScore: 130,
    status: "in_progress",
    replyText: "Hi Sunita, we sincerely apologise for the incorrect charge of ₹2,999 — this was clearly an error on our part and you should never have been enrolled in the annual plan. We have already downgraded your account to the monthly plan and a full refund of ₹2,700 (the difference) will appear on your statement within 3 business days. Thank you for flagging this.",
  },
  {
    customerName: "Tarun Kapoor",
    text: "The blender stopped working after 2 uses. The motor makes a grinding noise and it simply won't turn on anymore. Extremely disappointing for a product that costs ₹3,500.",
    rating: 2, sentimentScore: 15, sentimentLabel: "Negative",
    platform: "Yelp", issueCategory: "product", priorityScore: 115,
    status: "in_progress",
    replyText: "Hi Tarun, we are truly sorry that your blender failed after just 2 uses — this is completely below the quality standard we stand behind. A brand-new replacement unit has been dispatched today and should arrive within 2 days. You don't need to return the faulty unit. We have also extended your warranty to 3 years as an apology for the inconvenience.",
  },

  // ── MORE VARIETY ──────────────────────────────────────────────────────────
  {
    customerName: "Geeta Sharma",
    text: "Good product, slow shipping. I ordered express delivery (paid extra ₹199) but it arrived in standard time. The money for express delivery should be refunded.",
    rating: 3, sentimentScore: 40, sentimentLabel: "Neutral",
    platform: "Website", issueCategory: "delivery", priorityScore: 80, status: "new",
  },
  {
    customerName: "Ramesh Babu",
    text: "The app dashboard is confusing and hard to navigate. I spent 20 minutes trying to find where to download invoices. Functionality is good once you figure it out but the UX needs work.",
    rating: 3, sentimentScore: 44, sentimentLabel: "Neutral",
    platform: "App Store", issueCategory: "product", priorityScore: 76, status: "new",
  },
  {
    customerName: "Lalitha Murthy",
    text: "Ordered a blue shirt and received a green one. The label inside says blue but the colour is obviously green. Either your inventory management is off or the labels are wrong. Need an exchange.",
    rating: 2, sentimentScore: 22, sentimentLabel: "Negative",
    platform: "Google", issueCategory: "product", priorityScore: 108, status: "new",
  },
  {
    customerName: "Harish Menon",
    text: "Fantastic company. I've recommended to all my friends. The subscription plan is fair and the product updates keep improving. Keep up the great work!",
    rating: 5, sentimentScore: 93, sentimentLabel: "Positive",
    platform: "G2", issueCategory: "general", priorityScore: 0, status: "new",
  },
  {
    customerName: "Divya Saxena",
    text: "The helpdesk team resolved my issue but I had to reach out on Twitter before anyone responded. Please invest in your email support — 6 days without a reply is too long.",
    rating: 3, sentimentScore: 46, sentimentLabel: "Neutral",
    platform: "Facebook", issueCategory: "support", priorityScore: 74, status: "new",
  },
  {
    customerName: "Arun Kumar",
    text: "Placed an order on sale day. Price was ₹1,200 at checkout but my card was charged ₹1,650. The difference is ₹450 — either the sale wasn't applied correctly or there's a bug in the checkout.",
    rating: 2, sentimentScore: 25, sentimentLabel: "Negative",
    platform: "Website", issueCategory: "billing", priorityScore: 105, status: "new",
  },
  {
    customerName: "Shweta Jain",
    text: "I absolutely love this product. It's changed how I work every day. The design is elegant, the performance is rock solid, and the customer support team went above and beyond when I had a setup question.",
    rating: 5, sentimentScore: 96, sentimentLabel: "Positive",
    platform: "Trustpilot", issueCategory: "general", priorityScore: 0, status: "new",
  },
  {
    customerName: "Venkat Rao",
    text: "Been waiting 18 days for my replacement unit. First one was defective and I had to fight for 2 weeks just to get a replacement approved. Now I'm waiting again. This has been a 5-week ordeal.",
    rating: 1, sentimentScore: 6, sentimentLabel: "Negative",
    platform: "Google", issueCategory: "product", priorityScore: 134, status: "new",
  },
  {
    customerName: "Preeti Mishra",
    text: "Delivery was 2 days late but the product itself is great value for money. I'd buy again but would appreciate more accurate delivery estimates.",
    rating: 4, sentimentScore: 72, sentimentLabel: "Positive",
    platform: "Website", issueCategory: "delivery", priorityScore: 20, status: "new",
  },
  {
    customerName: "Nitin Sharma",
    text: "My account was locked without warning and I can't log in. I have active subscriptions under this account. Support says they can see my account but can't tell me why it was locked or when it'll be fixed. I need access urgently.",
    rating: 1, sentimentScore: 10, sentimentLabel: "Negative",
    platform: "App Store", issueCategory: "support", priorityScore: 130, status: "new",
  },
  {
    customerName: "Bhavna Desai",
    text: "Seamless experience. Ordered Sunday evening, delivered Monday afternoon. The packaging was eco-friendly which I appreciated. Product quality is excellent. I'm impressed.",
    rating: 5, sentimentScore: 95, sentimentLabel: "Positive",
    platform: "Google", issueCategory: "general", priorityScore: 0, status: "new",
  },
  {
    customerName: "Santosh Pillai",
    text: "The annual subscription auto-renewed without any reminder email. I had planned to cancel. ₹5,999 hit my account unexpectedly. At minimum you should send a reminder 7 days before renewal. Please process a refund.",
    rating: 2, sentimentScore: 20, sentimentLabel: "Negative",
    platform: "G2", issueCategory: "billing", priorityScore: 110, status: "new",
  },
  {
    customerName: "Rekha Bhat",
    text: "The product is good but arrived with a scratch on the front panel. It was in a protective sleeve but the scratch suggests it happened before packaging. Minor issue but disappointing for a new item at this price point.",
    rating: 3, sentimentScore: 40, sentimentLabel: "Neutral",
    platform: "Website", issueCategory: "product", priorityScore: 80, status: "new",
  },
  {
    customerName: "Chandrasekhar Rao",
    text: "Five star experience, no hesitation. The team is responsive, the product is premium, and delivery was surprisingly fast to my Tier-2 city. Pleasantly surprised. Will be back.",
    rating: 5, sentimentScore: 97, sentimentLabel: "Positive",
    platform: "Facebook", issueCategory: "general", priorityScore: 0, status: "new",
  },
  {
    customerName: "Isha Kapoor",
    text: "I contacted support about a billing question and the agent was dismissive and kept reading off a script. When I asked to speak to a manager I was told none were available and was asked to email instead. Not a helpful experience.",
    rating: 2, sentimentScore: 18, sentimentLabel: "Negative",
    platform: "Yelp", issueCategory: "support", priorityScore: 112, status: "new",
  },
  {
    customerName: "Mohan Das",
    text: "Returned my item on 20 May. Tracking shows it was received by your warehouse on 22 May. It is now 5 June and the refund of ₹3,200 has not been credited. I need this resolved.",
    rating: 1, sentimentScore: 8, sentimentLabel: "Negative",
    platform: "Trustpilot", issueCategory: "refund", priorityScore: 132, status: "new",
  },
];

// ── Seed function ─────────────────────────────────────────────────────────────
async function seed() {
  console.log("🌱  Connecting to MongoDB…");
  await mongoose.connect(MONGODB_URI);
  console.log("✓  Connected:", MONGODB_URI);

  const existing = await Review.countDocuments();
  if (existing > 0) {
    console.log(`⚠  ${existing} reviews already exist.`);
    const answer = process.argv.includes("--force") ? "yes" : "no";
    if (answer !== "yes") {
      console.log('   Pass --force to drop and re-seed. Exiting.');
      await mongoose.disconnect();
      return;
    }
    await Review.deleteMany({});
    console.log("✓  Cleared existing reviews.");
  }

  const now = new Date();
  const withDates = MOCK_REVIEWS.map((r, i) => ({
    ...r,
    date: new Date(now - (i * 1000 * 60 * 60 * 8)), // space 8 hrs apart
  }));

  await Review.insertMany(withDates);
  console.log(`✅  Seeded ${withDates.length} reviews successfully.`);
  await mongoose.disconnect();
  console.log("✓  Disconnected. Done.");
}

seed().catch((err) => {
  console.error("❌  Seed failed:", err);
  process.exit(1);
});