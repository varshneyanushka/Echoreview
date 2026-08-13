"""
main.py  –  EchoReview AI Service  v4.4
──────────────────────────────────────────────────────────────────────────────
Gemini + Groq + Template version.  100% free to run — no billing required.

Reply chain (in order of preference):
  1. Gemini  — Google AI Studio free tier (get key at aistudio.google.com)
  2. Groq    — Llama free tier (get key at console.groq.com)
  3. Template — always works, zero API calls

What this file does:
  • Sentiment analysis using your custom local sentiment model if available.
  • Falls back to Hugging Face sentiment model if custom model is missing.
  • Generates replies with Gemini → Groq → Template (automatic fallback chain).
  • Provides lightweight keyword-based insights.

Environment variables (all free, no billing needed):
  GEMINI_API_KEY  — from aistudio.google.com/apikey   (1 500 req/day free)
  GROQ_API_KEY    — from console.groq.com/keys         (14 400 req/day free)
  GEMINI_MODEL    — default: gemini-2.0-flash
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
import ssl
import certifi
import httpx
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# SSL FIX  (must run BEFORE `from google import genai`)
# Fixes "certificate verify failed: unable to get local issuer certificate"
# on macOS (Python.org installer) and some Linux environments.
#
# Root cause: httpx (used internally by google-genai SDK) calls
# ssl.create_default_context() which reads the system CA store — often
# incomplete on macOS Python.org installs.
# Fix: monkey-patch httpx.Client / AsyncClient so every instance defaults
# to certifi's up-to-date CA bundle, without touching HttpOptions at all.
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

_orig_client_init       = httpx.Client.__init__
_orig_async_client_init = httpx.AsyncClient.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", certifi.where())
    _orig_client_init(self, *args, **kwargs)

def _patched_async_client_init(self, *args, **kwargs):
    kwargs.setdefault("verify", certifi.where())
    _orig_async_client_init(self, *args, **kwargs)

httpx.Client.__init__      = _patched_client_init
httpx.AsyncClient.__init__ = _patched_async_client_init
# ─────────────────────────────────────────────────────────────────────────────

from google import genai

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Gemini — free tier via Google AI Studio (aistudio.google.com/apikey)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-2.0-flash")

# Groq — free tier, no billing ever (console.groq.com/keys)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")

CUSTOM_MODEL_DIR = os.getenv("CUSTOM_MODEL_DIR", "models/sentiment_model")
HF_FALLBACK_MODEL = os.getenv(
    "SENTIMENT_MODEL",
    "nlptown/bert-base-multilingual-uncased-sentiment",
)

app = FastAPI(title="EchoReview AI Service", version="4.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_custom_predictor = None
_hf_pipeline = None
_hf_model_name = ""
_gemini_client = None
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


def detect_issue(text: str) -> str:
    lower = text.lower()
    scores: Dict[str, int] = {}
    for issue, keywords in _ISSUE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score:
            scores[issue] = score
    return max(scores, key=lambda k: scores[k]) if scores else "general"


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
# GEMINI REPLY
# ─────────────────────────────────────────────────────────────────────────────
def _run_gemini_reply(payload: Dict[str, Any]) -> Optional[str]:
    if _gemini_client is None:
        return None

    name = _first_name(payload.get("customerName", "Customer"))
    review = (payload.get("text") or "").strip()
    platform = (payload.get("platform") or "our platform").strip()
    issue = detect_issue(review)
    sentiment = (payload.get("sentimentLabel") or derive_label(_rating_num(payload.get("rating")) * 20)).strip()
    specifics = extract_specifics(review)

    spec_lines = []
    if specifics.get("reference"):
        spec_lines.append(f"Reference: {specifics['reference']}")
    if specifics.get("duration"):
        spec_lines.append(f"Duration: {specifics['duration']}")
    if specifics.get("amount"):
        spec_lines.append(f"Amount: {specifics['amount']}")
    spec_block = "\n".join(spec_lines) if spec_lines else "None"

    prompt = f"""
You are writing a reply on behalf of a company.

Customer name: {name}
Platform: {platform}
Issue category: {issue}
Sentiment: {sentiment}
Extracted details: {spec_block}

Customer review:
{review}

Write a short, natural customer support reply.
Rules:
- Start with exactly: Hi {name},
- Be polite, professional, and human.
- Acknowledge the concern or praise.
- If negative, apologise briefly and mention the team will look into it.
- If positive, thank the customer warmly.
- Do NOT say you already refunded, replaced, cancelled, shipped, or fixed anything unless the review explicitly says so.
- Do NOT repeat the review.
- Keep it to 2 to 4 sentences.
- Output only the reply.
""".strip()

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        if not getattr(response, "text", None):
            return None
        return cleanup_reply(response.text)
    except Exception as exc:
        print(f"[Gemini] Reply error: {exc}")
        return None


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
            max_tokens=300,
            temperature=0.4,
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
def _run_custom_model(text: str) -> Optional[Dict[str, Any]]:
    if _custom_predictor is None:
        return None

    r = _custom_predictor.predict(text)
    return {
        "sentimentScore": r.sentiment_score,
        "sentimentLabel": r.sentiment_label,
        "sentimentProbs": r.sentiment_probs,
        "aspectLabel": r.aspect_label,
        "aspectProbs": r.aspect_probs,
        "modelSource": "custom-distilbert",
    }


def _run_nlptown(text: str) -> Optional[Dict[str, Any]]:
    if _hf_pipeline is None or "nlptown" not in _hf_model_name:
        return None

    result = _hf_pipeline(text[:512], truncation=True)[0]
    stars = int(result["label"][0])
    conf = float(result["score"])

    if stars <= 2:
        label, score = "Negative", round((stars / 2) * 40 * conf, 2)
    elif stars == 3:
        label, score = "Neutral", round(40 + 20 * conf, 2)
    else:
        label, score = "Positive", round(60 + ((stars - 3) / 2) * 40 * conf, 2)

    return {
        "sentimentScore": min(100, max(0, score)),
        "sentimentLabel": label,
        "sentimentProbs": {label: round(conf * 100, 2)},
        "aspectLabel": detect_issue(text),
        "aspectProbs": {},
        "modelSource": "nlptown-5star",
    }


def _run_distilbert_binary(text: str) -> Optional[Dict[str, Any]]:
    if _hf_pipeline is None:
        return None

    result = _hf_pipeline(text[:512], truncation=True)[0]
    raw = result["label"].upper()
    sc = float(result["score"])
    score = round(min(sc * 100, 100), 2) if raw == "POSITIVE" else round(min((1 - sc) * 100, 100), 2)

    return {
        "sentimentScore": score,
        "sentimentLabel": derive_label(score),
        "sentimentProbs": {},
        "aspectLabel": detect_issue(text),
        "aspectProbs": {},
        "modelSource": "distilbert-binary",
    }


def _keyword_fallback(text: str) -> Dict[str, Any]:
    lower = text.lower()
    pos = sum(
        1
        for w in [
            "great", "excellent", "amazing", "love", "best", "wonderful",
            "fantastic", "perfect", "happy", "satisfied", "brilliant", "impressed",
            "fast", "smooth", "easy", "helpful", "recommend", "outstanding",
        ]
        if w in lower
    )
    neg = sum(
        1
        for w in [
            "terrible", "awful", "worst", "hate", "horrible", "disappointed",
            "bad", "poor", "useless", "broken", "never", "refused", "ignored",
            "defective", "damaged", "overcharged", "late", "waiting", "rude",
        ]
        if w in lower
    )

    if pos > neg:
        score, label = 75.0, "Positive"
    elif neg > pos:
        score, label = 22.0, "Negative"
    else:
        score, label = 50.0, "Neutral"

    return {
        "sentimentScore": score,
        "sentimentLabel": label,
        "sentimentProbs": {},
        "aspectLabel": detect_issue(text),
        "aspectProbs": {},
        "modelSource": "keyword-fallback",
    }


def analyze_text(text: str) -> Dict[str, Any]:
    return (_run_custom_model(text) or _run_nlptown(text) or _run_distilbert_binary(text) or _keyword_fallback(text))


# ─────────────────────────────────────────────────────────────────────────────
# REPLY GENERATOR  —  chain: Gemini → Groq → Template
# ─────────────────────────────────────────────────────────────────────────────
def generate_reply(payload: Dict[str, Any], force_template: bool = False) -> Dict[str, Any]:
    issue = detect_issue(payload.get("text", ""))

    reply = None
    source = "template"

    if not force_template:
        # 1) Gemini
        if _gemini_client is not None:
            try:
                print("[Reply] Trying Gemini...")
                r = _run_gemini_reply(payload)
                if r and not _is_bad_reply(r, payload.get("text", ""), label="Gemini"):
                    reply = r
                    source = f"gemini/{GEMINI_MODEL}"
                    print(f"[Reply] Gemini accepted: {reply[:120]!r}")
                else:
                    print("[Reply] Gemini rejected")
            except Exception as exc:
                print(f"[Reply] Gemini failed: {exc}")

        # 2) Groq
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

    # 3) Template
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


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD INSIGHTS (NO ANTHROPIC IN THIS VERSION)
# ─────────────────────────────────────────────────────────────────────────────
def _keyword_insights(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = _compute_issue_stats(reviews)
    breakdown = stats.get("issueBreakdown", {})
    total = stats.get("totalReviews", 1)

    neg_count = sum(1 for r in reviews if r.get("sentimentLabel") == "Negative")
    unresolved = sum(1 for r in reviews if r.get("status") == "new")
    neg_pct = round(neg_count / max(total, 1) * 100, 1)

    insights = []
    recommendations = []

    if breakdown:
        top_issue, top_data = max(breakdown.items(), key=lambda x: x[1]["count"])
        insights.append({
            "type": "theme",
            "title": f"{top_issue.capitalize()} is your top complaint area",
            "detail": (
                f"{top_data['count']} reviews ({top_data['percentage']}%) mention {top_issue} issues. "
                f"Average rating for this issue is {top_data['avgRating']:.1f}★ and average sentiment is {top_data['avgSentiment']}%. "
                f"Severity is classified as {top_data['severity']}."
            ),
            "severity": top_data["severity"],
            "affectedCount": top_data["count"],
        })
        recommendations.append({
            "action": f"Prioritise fixing {top_issue} — it affects {top_data['percentage']}% of your reviews",
            "impact": "Reduce negative review rate and improve average rating",
            "timeframe": "this week",
            "priority": top_data["severity"],
        })

    trending = [k for k, v in breakdown.items() if v.get("trending")]
    if trending:
        insights.append({
            "type": "operational",
            "title": f"Rising issues: {', '.join(t.capitalize() for t in trending)}",
            "detail": (
                "These issue categories have increased in the last 7 days, suggesting a possible "
                "operational change or external event is affecting customer experience."
            ),
            "severity": "high",
            "affectedCount": sum(breakdown[t]["recentCount"] for t in trending),
        })

    if neg_pct > 40:
        insights.append({
            "type": "urgent",
            "title": f"{neg_pct:.0f}% of reviews are negative — above safe threshold",
            "detail": (
                "A negative review rate above 40% significantly impacts platform rankings and brand perception. "
                "Immediate intervention on top issues is recommended."
            ),
            "severity": "critical" if neg_pct > 60 else "high",
            "affectedCount": neg_count,
        })
        recommendations.append({
            "action": "Set up automated alerts for 1-2 star reviews and respond within 24 hours",
            "impact": "Reduce escalation and show responsiveness to potential customers reading reviews",
            "timeframe": "immediate",
            "priority": "critical",
        })

    if unresolved > 10:
        insights.append({
            "type": "urgent",
            "title": f"{unresolved} reviews still awaiting a reply",
            "detail": (
                "A large backlog of unanswered reviews signals poor responsiveness to new customers "
                "and may affect your platform ranking."
            ),
            "severity": "high" if unresolved > 20 else "medium",
            "affectedCount": unresolved,
        })
        recommendations.append({
            "action": f"Use AI Reply to clear the {unresolved}-review backlog",
            "impact": "Improve response rate metric and customer trust",
            "timeframe": "this week",
            "priority": "high",
        })

    top_issue_name = (breakdown and max(breakdown.items(), key=lambda x: x[1]["count"])[0]) or "general"

    return {
        "executiveSummary": (
            f"Analysis of {total} reviews shows {neg_pct}% negative sentiment. "
            f"The top issue is {top_issue_name} ({breakdown.get(top_issue_name, {}).get('percentage', 0)}% of reviews). "
            f"{unresolved} reviews are still awaiting a reply."
        ),
        "insights": insights,
        "recommendations": recommendations,
        "stats": stats,
        "churnRiskCount": 0,
        "urgentReplyCount": unresolved,
        "topComplaintTheme": top_issue_name,
        "generatedBy": "keyword-analysis",
        "generatedAt": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_everything():
    global _custom_predictor, _hf_pipeline, _hf_model_name, _gemini_client, _groq_client, _reply_engine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EchoReview AI] v4.4 starting on {device}…")

    # 1) Custom sentiment model
    try:
        from sentiment_model import SentimentPredictor
        _custom_predictor = SentimentPredictor.from_directory(CUSTOM_MODEL_DIR, device)
        print("[EchoReview AI] ✓ Custom sentiment model loaded")
    except FileNotFoundError:
        print("[EchoReview AI] ℹ Custom model not found — using HF fallback")
        _custom_predictor = None
    except Exception as exc:
        print(f"[EchoReview AI] ✗ Custom model error: {exc}")
        _custom_predictor = None

    # 2) Hugging Face fallback sentiment model
    if _custom_predictor is None:
        try:
            from transformers import pipeline as hf_pipeline
            _hf_pipeline = hf_pipeline(
                "text-classification",
                model=HF_FALLBACK_MODEL,
                device=0 if torch.cuda.is_available() else -1,
            )
            _hf_model_name = HF_FALLBACK_MODEL
            print(f"[EchoReview AI] ✓ HF sentiment fallback: {HF_FALLBACK_MODEL}")
        except Exception as exc:
            print(f"[EchoReview AI] ✗ HF fallback failed: {exc}")
            _hf_pipeline = None
            _hf_model_name = ""

    # 3) Gemini client — httpx already patched above to use certifi
    if GEMINI_API_KEY:
        try:
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            _reply_engine  = f"gemini/{GEMINI_MODEL}"
            print(f"[EchoReview AI] ✓ Gemini ready ({GEMINI_MODEL})")
        except Exception as exc:
            _gemini_client = None
            print(f"[EchoReview AI] ✗ Gemini init failed: {exc}")
    else:
        _gemini_client = None
        print("[EchoReview AI] ℹ GEMINI_API_KEY not set — skipping Gemini")

    # 4) Groq client — free tier, no billing needed (console.groq.com)
    if GROQ_API_KEY:
        try:
            from groq import Groq
            _groq_client  = Groq(api_key=GROQ_API_KEY)
            _reply_engine = _reply_engine if _gemini_client else f"groq/{GROQ_MODEL}"
            print(f"[EchoReview AI] ✓ Groq ready ({GROQ_MODEL})")
        except Exception as exc:
            _groq_client = None
            print(f"[EchoReview AI] ✗ Groq init failed: {exc}")
    else:
        _groq_client = None
        print("[EchoReview AI] ℹ GROQ_API_KEY not set — skipping Groq")

    # Determine active reply engine label for logs
    if not _gemini_client and not _groq_client:
        _reply_engine = "template"

    sentiment_engine = "custom-distilbert" if _custom_predictor else _hf_model_name or "keyword"
    print(f"\n[EchoReview AI] Sentiment : {sentiment_engine}")
    print(f"[EchoReview AI] Reply     : {_reply_engine}  (chain: Gemini → Groq → Template)")
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
        "version": "4.4.0",
        "sentimentEngine": (
            "custom-distilbert" if _custom_predictor
            else _hf_model_name if _hf_pipeline else "keyword-fallback"
        ),
        "replyEngine":      _reply_engine,
        "geminiEnabled":    _gemini_client is not None,
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