"""
main.py  –  EchoReview AI Service  v5.0
──────────────────────────────────────────────────────────────────────────────
Groq + Template version. Lightweight and free-tier friendly.

Reply chain: Groq → Template (always works, zero API calls).

What this file does:
  • Uses lightweight keyword sentiment analysis (no model downloads at startup).
  • Generates replies with Groq → Template (automatic fallback chain).
  • Provides lightweight keyword-based insights.

Environment variables (all free, no billing needed):
  GROQ_API_KEY    — from console.groq.com/keys         (14 400 req/day free)
  GROQ_MODEL      — default: llama-3.3-70b-versatile
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Groq — obtain a key from console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "6"))

ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv("CLIENT_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app = FastAPI(title="EchoReview AI Service", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_groq_client   = None
_reply_engine = "template"


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class GenerateReplyRequest(BaseModel):
    customerName: str = Field(default="Customer")
    text: str = Field(..., min_length=1)
    rating: Optional[float] = None
    platform: Optional[str] = None
    sentimentScore: Optional[float] = None
    sentimentLabel: Optional[str] = None


class InsightsRequest(BaseModel):
    reviews: List[Dict[str, Any]] = Field(..., min_length=1)


class IssuesSummaryRequest(BaseModel):
    reviews: List[Dict[str, Any]] = Field(..., min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "delivery": [
        "delivery", "delivered", "shipping", "shipped", "dispatch", "dispatched",
        "courier", "parcel", "package", "arrived", "arrival", "transit",
        "late", "delay", "delayed", "not arrived", "tracking", "lost in transit",
    ],
    "billing": [
        "charge", "charged", "billing", "billed", "invoice", "invoiced",
        "payment", "paid", "refund", "subscription", "price", "fee",
        "overcharged", "double charge", "duplicate", "amount", "rupee",
        "auto-renew", "auto renewal", "cancel subscription",
    ],
    "support": [
        "support", "customer service", "customer care", "agent", "representative",
        "helpdesk", "help desk", "chat", "call", "ticket", "email",
        "response", "reply", "responded", "ignored", "rude", "unhelpful",
        "no response", "never replied", "waiting", "escalate",
    ],
    "product": [
        "product", "item", "quality", "defect", "defective", "broken",
        "damaged", "malfunction", "stopped working", "not working",
        "wrong item", "not as described", "poor quality", "fell apart",
        "missing parts", "scratch", "dent",
    ],
    "refund": [
        "refund", "return", "returns", "returned", "replacement", "exchange",
        "money back", "reimbursement", "credit", "return label",
        "return portal", "return process", "warranty claim",
    ],
}

# Fault signals are intentionally explicit and auditable. They catch high-risk
# wording that a simple positive/negative score would otherwise hide.
_FAULT_SIGNALS: Dict[str, List[str]] = {
    "urgent": ["urgent", "immediately", "asap", "emergency", "fraud", "scam"],
    "repeat": ["again", "second time", "third time", "repeated", "still", "every time"],
    "service_failure": ["no response", "never replied", "cannot reach", "ignored", "unresolved"],
    "money_risk": ["charged twice", "double charge", "unauthorised", "unauthorized", "overcharged", "refund not received"],
    "delivery_failure": ["not arrived", "lost in transit", "never arrived", "marked delivered"],
    "product_failure": ["unsafe", "fire", "injured", "dangerous", "stopped working", "defective"],
}

_POSITIVE_TERMS = {
    "great": 2, "excellent": 3, "amazing": 3, "love": 2, "best": 2,
    "wonderful": 2, "fantastic": 3, "perfect": 3, "happy": 2,
    "satisfied": 2, "brilliant": 2, "impressed": 2, "fast": 1,
    "smooth": 1, "easy": 1, "helpful": 2, "recommend": 2, "outstanding": 3,
}
_NEGATIVE_TERMS = {
    "terrible": 3, "awful": 3, "worst": 3, "hate": 3, "horrible": 3,
    "disappointed": 2, "bad": 1, "poor": 2, "useless": 3, "broken": 3,
    "never": 1, "refused": 2, "ignored": 2, "defective": 3, "damaged": 2,
    "overcharged": 3, "late": 1, "waiting": 1, "rude": 2, "unacceptable": 3,
    "frustrated": 2, "angry": 2, "cancel": 1,
}


def detect_issue(text: str) -> str:
    lower = text.lower()
    scores: Dict[str, int] = {}
    for issue, keywords in _ISSUE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score:
            scores[issue] = score
    return max(scores, key=lambda k: scores[k]) if scores else "general"


def detect_issues(text: str) -> List[str]:
    """Return all relevant issue categories, ordered by keyword evidence."""
    lower = text.lower()
    ranked = sorted(
        ((issue, sum(1 for word in words if word in lower)) for issue, words in _ISSUE_KEYWORDS.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return [issue for issue, score in ranked if score] or ["general"]


def detect_faults(text: str) -> List[str]:
    lower = text.lower()
    return [fault for fault, phrases in _FAULT_SIGNALS.items() if any(phrase in lower for phrase in phrases)]


def extract_specifics(text: str) -> Dict[str, Any]:
    details: Dict[str, Any] = {}

    refs = re.findall(r"#?\b([A-Z]{1,4}[-–]?\d{3,10})\b", text)
    if refs:
        details["reference"] = refs[0]

    durations = re.findall(
        r"\b(\d+\s*(?:day|week|month|hour|minute)s?)\b",
        text,
        re.IGNORECASE,
    )
    if durations:
        details["duration"] = durations[0]

    amounts = re.findall(
        r"[₹$£€]?\s*\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:USD|GBP|EUR|INR|rupee))?",
        text,
    )
    if amounts:
        details["amount"] = amounts[0].strip()

    return details


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def derive_label(score: float) -> str:
    if score < 40:
        return "Negative"
    if score < 60:
        return "Neutral"
    return "Positive"


def _rating_num(rating: Any) -> float:
    try:
        return float(rating)
    except (TypeError, ValueError):
        return 3.0


def _first_name(full: str) -> str:
    return (full or "Customer").strip().split()[0]


def cleanup_reply(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Strip known preamble prefixes — LLaMA/Gemini often add these before the reply
    for prefix in [
        "reply:", "response:", "answer:", "write a reply:", "customer reply:",
        "generate a reply:", "here is a reply:", "here's a reply:",
        "customer service reply:", "professional reply:",
        "here is a response:", "here's a response:",
        "here is the reply:", "here's the reply:",
        "here is a customer support reply:", "here's a customer support reply:",
        "sure, here is", "sure, here's", "sure! here",
        "of course, here", "of course! here", "certainly, here",
        "certainly! here",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]

    # If there is still a preamble before the actual greeting, skip to the greeting.
    # e.g. LLaMA returns "Here's a reply:\n\nHi John, ..." — after stripping the
    # prefix above the remaining text may still have a line break before "Hi ".
    for greeting in ("Hi ", "Dear ", "Hello "):
        idx = text.find(greeting)
        if 0 < idx < 200:          # greeting found but not at position 0
            text = text[idx:]
            break

    return re.sub(r"\s+", " ", text).strip()


def _is_bad_reply(reply: str, original_text: str, label: str = "") -> bool:
    """Return True if reply is too short, echoes the review, or sounds wrong.
    Pass label="Groq" / "Gemini" etc. to get named rejection logs.
    """
    tag = f"[{label}] " if label else ""

    if not reply:
        return True

    reply = reply.strip()

    if len(reply) < 30:
        print(f"{tag}Reply rejected: too short ({len(reply)} chars)")
        return True

    lower = reply.lower()

    bad_phrases = [
        "generate a",
        "customer review",
        "sentiment:",
        "issue type:",
        "platform:",
        "star rating:",
        "instructions:",
        "task:",
        "prompt:",
        "tone:",
        "1-star",
        "2-star",
        "3-star",
        "4-star",
        "5-star",
        "business reply:",
        "customer review:",
        "issue category:",
    ]
    for p in bad_phrases:
        if p in lower:
            print(f"{tag}Reply rejected: bad phrase '{p}'")
            return True

    reviewer_phrases = [
        "i'm not a fan",
        "i am not a fan",
        "i've been here",
        "i have been here",
        "i ordered from here",
        "i bought this",
        "i would not recommend",
        "i will not be coming back",
        "definitely not coming back",
        "save your money",
        "worst purchase",
        "worst experience",
        "the food was",
        "my order was",
    ]
    for p in reviewer_phrases:
        if p in lower:
            print(f"{tag}Reply rejected: reviewer phrase '{p}'")
            return True

    negative_company_phrases = [
        "terrible product",
        "bad product",
        "awful product",
        "worst product",
        "horrible service",
        "awful service",
        "terrible service",
        "our product is",
        "our service is terrible",
        "our service is bad",
        "do not buy",
        "don't buy",
        "not worth buying",
    ]
    for p in negative_company_phrases:
        if p in lower:
            print(f"{tag}Reply rejected: negative-company phrase '{p}'")
            return True

    if not (lower.startswith("hi ") or lower.startswith("dear ") or lower.startswith("hello ")):
        print(f"{tag}Reply rejected: does not start with Hi/Dear/Hello — starts with: {reply[:60]!r}")
        return True

    orig_words  = set(w for w in re.findall(r"\w+", original_text.lower()) if len(w) > 3)
    reply_words = set(w for w in re.findall(r"\w+", lower)                  if len(w) > 3)

    if len(orig_words) > 8:
        overlap = len(orig_words & reply_words) / len(orig_words)
        if overlap > 0.75:          # raised from 0.60 — support replies naturally reuse topic words
            print(f"{tag}Reply rejected: word overlap too high ({overlap:.0%})")
            return True

    if len(reply.split()) < 12 and ("sorry" in lower or "apolog" in lower):
        print(f"{tag}Reply rejected: too short and only apologises")
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE REPLY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
_RESOLUTION_STEPS: Dict[str, str] = {
    "delivery": (
        "Our logistics team is investigating your shipment right now as a priority. "
        "Please share your order number with our support team and we will arrange "
        "an immediate replacement or full refund — whichever you prefer."
    ),
    "billing": (
        "Our finance team is reviewing your account immediately. "
        "Please contact our support team with your transaction details and we "
        "will correct any errors and process a refund within 3–5 business days."
    ),
    "support": (
        "This is not the standard we hold ourselves to and we are escalating "
        "your case directly to a senior team member, who will reach out to you "
        "within the next few hours to resolve this personally."
    ),
    "product": (
        "This falls well below our quality standards. Please contact our support "
        "team with a photo of the issue and we will arrange a free replacement or "
        "a full refund — whichever you prefer — at no cost to you."
    ),
    "refund": (
        "Your refund should be simple and stress-free. Please reach out to our "
        "support team directly and we will personally ensure your refund is "
        "processed within 3–5 business days."
    ),
    "general": (
        "We want to make this right for you. Please reach out to our support "
        "team so we can look into your case personally and find a resolution."
    ),
}

_ISSUE_PHRASE: Dict[str, str] = {
    "delivery": "the delivery issue",
    "billing": "the billing concern",
    "support": "the support experience you described",
    "product": "the product issue",
    "refund": "the refund difficulty",
    "general": "the experience you shared",
}


def _template_reply(payload: Dict[str, Any]) -> str:
    name = _first_name(payload.get("customerName", "Customer"))
    review = (payload.get("text") or "").strip()
    platform = (payload.get("platform") or "our platform").strip()
    sentiment = (payload.get("sentimentLabel") or "Neutral").strip()
    rating = _rating_num(payload.get("rating"))
    issue = detect_issue(review)
    specifics = extract_specifics(review)
    ref = f" (ref: {specifics['reference']})" if specifics.get("reference") else ""

    if sentiment == "Positive" or rating >= 4:
        return (
            f"Hi {name}, thank you so much for this wonderful review! "
            f"We're genuinely thrilled to hear you had such a positive experience with {platform}. "
            f"Feedback like yours motivates our entire team every day. "
            f"We look forward to serving you again soon!"
        )

    if sentiment == "Negative" or rating <= 2:
        phrase = _ISSUE_PHRASE.get(issue, "the issue")
        resolution = _RESOLUTION_STEPS.get(issue, _RESOLUTION_STEPS["general"])

        if issue == "delivery" and specifics.get("duration"):
            phrase = f"the {specifics['duration']} delivery delay{ref}"
        elif issue == "billing" and specifics.get("amount"):
            phrase = f"the billing issue of {specifics['amount']}{ref}"
        elif issue == "product":
            phrase = "the product issue you experienced"

        return (
            f"Hi {name}, thank you for your feedback{ref}, and we are truly sorry "
            f"about {phrase}. This is absolutely not the experience we aim to deliver. "
            f"{resolution}"
        )

    return (
        f"Hi {name}, thank you for sharing your experience with {platform}. "
        f"We have noted your comments and will use them to keep improving. "
        f"If there is anything specific we can do better, please reach out to us directly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# GROQ REPLY  (free-tier fallback — llama-3.3-70b-versatile)
# ─────────────────────────────────────────────────────────────────────────────
def _run_groq_reply(payload: Dict[str, Any]) -> Optional[str]:
    if _groq_client is None:
        return None

    name      = _first_name(payload.get("customerName", "Customer"))
    review    = (payload.get("text") or "").strip()
    platform  = (payload.get("platform") or "our platform").strip()
    issue     = detect_issue(review)
    sentiment = (payload.get("sentimentLabel") or derive_label(
        _rating_num(payload.get("rating")) * 20
    )).strip()
    specifics = extract_specifics(review)

    spec_lines = []
    if specifics.get("reference"):
        spec_lines.append(f"Reference: {specifics['reference']}")
    if specifics.get("duration"):
        spec_lines.append(f"Duration: {specifics['duration']}")
    if specifics.get("amount"):
        spec_lines.append(f"Amount: {specifics['amount']}")
    spec_block = "\n".join(spec_lines) if spec_lines else "None"

    prompt = f"""You are a professional customer support representative writing on behalf of a company.

Customer name: {name}
Platform: {platform}
Issue category: {issue}
Sentiment: {sentiment}
Extracted details: {spec_block}

Customer review:
{review}

Write a short customer support reply.

Rules:
- Start with exactly: Hi {name},
- Sound human, warm, and professional
- If negative, apologise briefly and say the team will look into it
- If positive, thank the customer warmly
- Do NOT say you already refunded, replaced, cancelled, shipped, or fixed anything unless the review explicitly says so
- Do NOT repeat the review
- Keep it to 2 to 4 sentences
- Output only the reply
"""

    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.4,
            timeout=GROQ_TIMEOUT_SECONDS,
        )

        raw = response.choices[0].message.content or ""
        print(f"[Groq RAW] {raw[:250]!r}")

        reply = cleanup_reply(raw)
        print(f"[Groq CLEAN] {reply[:250]!r}")

        return reply if reply else None

    except Exception as exc:
        print(f"[Groq] Reply error: {exc}")
        return None
    
# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT CASCADE
# ─────────────────────────────────────────────────────────────────────────────
def _keyword_fallback(text: str) -> Dict[str, Any]:
    lower = text.lower()
    words = re.findall(r"[a-z']+", lower)
    pos = sum(weight for term, weight in _POSITIVE_TERMS.items() if term in lower)
    neg = sum(weight for term, weight in _NEGATIVE_TERMS.items() if term in lower)
    faults = detect_faults(text)
    # Strong negations, caps, exclamation clusters and repeat/fault reports
    # increase severity without turning a mixed review into an arbitrary 22/75.
    neg += sum(1 for i, word in enumerate(words[:-1]) if word in {"not", "never", "no"} and words[i + 1] in _POSITIVE_TERMS)
    neg += min(2, lower.count("!"))
    neg += len(faults)
    raw = 50 + (pos - neg) * 7
    score = round(max(3, min(97, raw)), 1)
    label = derive_label(score)
    confidence = round(min(0.95, 0.45 + abs(pos - neg) * 0.08 + len(faults) * 0.05) * 100, 1)

    return {
        "sentimentScore": score,
        "sentimentLabel": label,
        "sentimentProbs": {"confidence": confidence, "positiveSignals": pos, "negativeSignals": neg},
        "aspectLabel": detect_issue(text),
        "aspectProbs": {issue: 1 for issue in detect_issues(text)},
        "faultSignals": faults,
        "modelSource": "weighted-lexicon-v2",
    }


def analyze_text(text: str) -> Dict[str, Any]:
    return _keyword_fallback(text)

# ─────────────────────────────────────────────────────────────────────────────
# REPLY GENERATOR  —  chain: Groq → Template
# ─────────────────────────────────────────────────────────────────────────────
def generate_reply(payload: Dict[str, Any], force_template: bool = False) -> Dict[str, Any]:
    issue = detect_issue(payload.get("text", ""))

    reply = None
    source = "template"

    if not force_template:
        # 1) Groq
        if not reply and _groq_client is not None:
            try:
                print("[Reply] Trying Groq...")
                r = _run_groq_reply(payload)
                if r and not _is_bad_reply(r, payload.get("text", ""), label="Groq"):
                    reply = r
                    source = f"groq/{GROQ_MODEL}"
                    print(f"[Reply] Groq accepted: {reply[:120]!r}")
                else:
                    print("[Reply] Groq rejected")
            except Exception as exc:
                print(f"[Reply] Groq failed: {exc}")

    # 2) Template
    if not reply:
        reply = _template_reply(payload)
        source = "template"
        print(f"[Reply] Template used: {reply[:120]!r}")

    return {
        "reply": cleanup_reply(reply),
        "issueCategory": issue,
        "source": source,
        "metadata": {
            "customerName": payload.get("customerName"),
            "platform": payload.get("platform"),
            "sentimentLabel": payload.get("sentimentLabel"),
            "sentimentScore": payload.get("sentimentScore"),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# ISSUE STATS
# ─────────────────────────────────────────────────────────────────────────────
def _compute_issue_stats(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(reviews)
    if not total:
        return {}

    issue_counts = Counter()
    issue_sentiment = defaultdict(list)
    issue_ratings = defaultdict(list)
    platform_counts = Counter()
    recent_issues = Counter()
    cutoff = datetime.utcnow() - timedelta(days=7)

    for r in reviews:
        text = r.get("text", "")
        issue = r.get("issueCategory") or detect_issue(text)
        sent = r.get("sentimentScore", 50)
        rating = _rating_num(r.get("rating", 3))
        plat = r.get("platform", "Other")
        date_str = r.get("date") or r.get("createdAt", "")

        issue_counts[issue] += 1
        issue_sentiment[issue].append(sent)
        issue_ratings[issue].append(rating)
        platform_counts[plat] += 1

        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.replace(tzinfo=None) >= cutoff:
                recent_issues[issue] += 1
        except Exception:
            pass

    result = {}
    for issue, count in issue_counts.items():
        avg_sent = round(sum(issue_sentiment[issue]) / len(issue_sentiment[issue]), 1)
        avg_rating = round(sum(issue_ratings[issue]) / len(issue_ratings[issue]), 2)
        result[issue] = {
            "count": count,
            "percentage": round(count / total * 100, 1),
            "avgSentiment": avg_sent,
            "avgRating": avg_rating,
            "severity": (
                "critical" if avg_sent < 25 or avg_rating <= 1.5 else
                "high" if avg_sent < 40 or avg_rating <= 2.5 else
                "medium" if avg_sent < 55 or avg_rating <= 3.5 else
                "low"
            ),
            "recentCount": recent_issues.get(issue, 0),
            "trending": recent_issues.get(issue, 0) > (count * 0.3),
        }

    return {
        "issueBreakdown": result,
        "topIssue": issue_counts.most_common(1)[0][0] if issue_counts else "general",
        "platformBreakdown": dict(platform_counts.most_common()),
        "totalReviews": total,
    }


def cluster_reviews(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cluster by multi-label issue + fault signature; no remote model required."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        text = review.get("text", "")
        issues = detect_issues(text)
        faults = detect_faults(text)
        # Primary issue produces stable group names. The first fault adds a
        # useful split such as "billing · money risk".
        key = issues[0]
        if faults:
            key = f"{key} · {faults[0].replace('_', ' ')}"
        groups[key].append(review)

    clusters = []
    for cluster_id, (name, items) in enumerate(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])), start=1):
        scores = [float(item.get("sentimentScore", analyze_text(item.get("text", ""))["sentimentScore"])) for item in items]
        fault_counts = Counter(fault for item in items for fault in detect_faults(item.get("text", "")))
        clusters.append({
            "clusterId": cluster_id,
            "clusterName": name.title(),
            "clusterSummary": f"{len(items)} reviews; average sentiment {round(sum(scores) / len(scores), 1)}.",
            "size": len(items),
            "avgSentiment": round(sum(scores) / len(scores), 1),
            "faults": dict(fault_counts.most_common(3)),
            "sampleText": items[0].get("text", ""),
            "reviews": items[:3],
        })
    return clusters


def detect_fault_patterns(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fault_reviews: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        for fault in detect_faults(review.get("text", "")):
            fault_reviews[fault].append(review)
    findings = []
    for fault, items in fault_reviews.items():
        avg_score = sum(float(item.get("sentimentScore", 50)) for item in items) / len(items)
        severity = "critical" if fault in {"money_risk", "product_failure", "urgent"} or avg_score < 25 else "high"
        findings.append({
            "fault": fault,
            "title": fault.replace("_", " ").title(),
            "count": len(items),
            "severity": severity,
            "detail": f"Detected in {len(items)} review(s), with average sentiment {round(avg_score, 1)}.",
        })
    return sorted(findings, key=lambda item: ({"critical": 0, "high": 1}.get(item["severity"], 2), -item["count"]))


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD INSIGHTS (NO ANTHROPIC IN THIS VERSION)
# ─────────────────────────────────────────────────────────────────────────────
def _keyword_insights(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = _compute_issue_stats(reviews)
    breakdown = stats.get("issueBreakdown", {})
    total = stats.get("totalReviews", 1)

    sentiment_scores = [r.get("sentimentScore", 50) for r in reviews]
    avg_sentiment = sum(sentiment_scores) / max(len(sentiment_scores), 1)

    neg_reviews = [r for r in reviews if r.get("sentimentLabel") == "Negative"]
    neg_ratio = len(neg_reviews) / max(total, 1)

    # trending detection (simple delta-based ML logic)
    trending_issues = []
    for issue, data in breakdown.items():
        if data.get("trending"):
            trending_issues.append(issue)

    # severity scoring (ML-style weighted function)
    def severity(score, count, trend):
        base = (100 - score) * 0.6 + count * 2
        if trend:
            base *= 1.3
        if base > 70:
            return "critical"
        elif base > 50:
            return "high"
        elif base > 30:
            return "medium"
        return "low"

    insights = []
    recommendations = []

    for issue, data in breakdown.items():
        sev = severity(
            data["avgSentiment"],
            data["count"],
            data.get("trending", False),
        )

        insights.append({
            "type": "issue",
            "title": f"{issue.capitalize()} issue pattern detected",
            "detail": (
                f"{data['count']} occurrences, "
                f"avg sentiment {data['avgSentiment']}, "
                f"{'trending upward' if data.get('trending') else 'stable'}"
            ),
            "severity": sev,
        })

        if sev in ["high", "critical"]:
            recommendations.append({
                "action": f"Fix {issue} pipeline urgently",
                "impact": "Reduce negative sentiment & improve rating",
                "priority": sev,
            })

    fault_patterns = detect_fault_patterns(reviews)
    for fault in fault_patterns:
        insights.append({
            "type": "fault",
            "title": f"{fault['title']} detected",
            "detail": fault["detail"],
            "severity": fault["severity"],
        })
        recommendations.append({
            "action": f"Investigate {fault['title'].lower()} cases and assign an owner",
            "impact": "Reduce escalations and customer churn",
            "priority": fault["severity"],
            "timeframe": "immediate" if fault["severity"] == "critical" else "this week",
        })

    # global risk insight
    if neg_ratio > 0.4:
        insights.append({
            "type": "risk",
            "title": "High negative sentiment cluster detected",
            "detail": f"{round(neg_ratio*100,1)}% reviews are negative",
            "severity": "critical",
        })

    return {
        "executiveSummary": (
            f"Dataset shows avg sentiment {round(avg_sentiment,1)} with "
            f"{round(neg_ratio*100,1)}% negative reviews."
        ),
        "insights": insights,
        "recommendations": recommendations,
        "stats": stats,
        "clusters": cluster_reviews(reviews),
        "faultPatterns": fault_patterns,
        "topComplaintTheme": stats.get("topIssue"),
        "generatedBy": "weighted-sentiment-fault-clustering-v2",
        "generatedAt": datetime.utcnow().isoformat(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_everything():
    global _groq_client, _reply_engine
    print("\n[EchoReview AI] v5.0 starting…")

    # Groq client
    if GROQ_API_KEY:
        try:
            from groq import Groq
            _groq_client  = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS, max_retries=0)
            _reply_engine = f"groq/{GROQ_MODEL}"
            print(f"[EchoReview AI] ✓ Groq ready ({GROQ_MODEL})")
        except Exception as exc:
            _groq_client = None
            print(f"[EchoReview AI] ✗ Groq init failed: {exc}")
    else:
        _groq_client = None
        print("[EchoReview AI] ℹ GROQ_API_KEY not set — skipping Groq")

    # Determine active reply engine label for logs
    if not _groq_client:
        _reply_engine = "template"

    print("[EchoReview AI] Sentiment : keyword-analysis")
    print(f"[EchoReview AI] Reply     : {_reply_engine}  (chain: Groq → Template)")
    print("[EchoReview AI] Insights  : keyword-analysis")
    print("[EchoReview AI] Ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "EchoReview AI Service",
        "version": "5.0.0",
        "sentimentEngine": "keyword-analysis",
        "replyEngine":      _reply_engine,
        "groqEnabled":      _groq_client   is not None,
        "anthropicEnabled": False,
        "insightsEngine":   "keyword-analysis",
    }


@app.post("/analyze")
def analyze_sentiment(req: AnalyzeRequest):
    text = req.text.strip()
    result = analyze_text(text)
    issue = result.get("aspectLabel") or detect_issue(text)
    return {
        "text": text,
        "sentimentScore": result["sentimentScore"],
        "sentimentLabel": result["sentimentLabel"],
        "sentimentProbs": result.get("sentimentProbs", {}),
        "issueCategory": issue,
        "aspectProbs": result.get("aspectProbs", {}),
        "modelSource": result.get("modelSource", "unknown"),
    }


@app.post("/generate-reply")
def gen_reply(req: GenerateReplyRequest, mode: str = "auto"):
    return generate_reply(req.model_dump(), force_template=(mode == "template"))


@app.post("/generate-reply/template")
def gen_reply_template(req: GenerateReplyRequest):
    issue = detect_issue(req.text)
    return {
        "reply": _template_reply(req.model_dump()),
        "issueCategory": issue,
        "source": "template",
        "metadata": {"customerName": req.customerName, "sentimentLabel": req.sentimentLabel},
    }


@app.post("/insights")
def get_insights(req: InsightsRequest):
    return _keyword_insights(req.reviews)



@app.post("/issues/summary")
def issues_summary(req: IssuesSummaryRequest):
    return _compute_issue_stats(req.reviews)
@app.post("/issues/clusters")
def issue_clusters(req: IssuesSummaryRequest):
    return {"success": True, "clusters": cluster_reviews(req.reviews)}


@app.post("/issues/map")
def issue_map(req: IssuesSummaryRequest):
    return {"success": True, "points": cluster_reviews(req.reviews)}


@app.get("/cluster/health")
def cluster_health():
    return {
        "status": "ok",
        "model": "weighted-lexicon + multi-signal fault clustering"
    }
