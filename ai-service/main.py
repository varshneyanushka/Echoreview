"""
main.py  –  EchoReview AI Service  v4.2
──────────────────────────────────────────────────────────────────────────────
Key changes from v4.1:
  • FLAN prompt uses FEW-SHOT examples (shows the model exactly what a good
    business reply looks like before asking it to write one). This fixes the
    "I'm not a fan of this place" hallucination — the model now copies the
    business voice from the examples rather than inventing a reviewer voice.
  • Supports a fine-tuned FLAN model if train_reply.py has been run.
    Set FLAN_REPLY_MODEL_DIR=models/flan_reply to use it.
    Falls back to base FLAN with few-shot if fine-tuned model not found.
  • /insights endpoint now sends actual review texts to Claude for real NLP:
    topic extraction, churn risk scoring, urgent issue detection, and
    specific recommendations. Falls back to keyword analysis if no key.
  • Stricter echo detection — catches partial echoes too.
"""

from __future__ import annotations

import os
import re
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from anthropic import Anthropic as _Anthropic
    _ANTHROPIC_PKG = True
except ImportError:
    _ANTHROPIC_PKG = False

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL          = "claude-sonnet-4-20250514"
CUSTOM_MODEL_DIR      = os.getenv("CUSTOM_MODEL_DIR",      "models/sentiment_model")
HF_FALLBACK_MODEL     = os.getenv("SENTIMENT_MODEL",       "nlptown/bert-base-multilingual-uncased-sentiment")
FLAN_BASE_MODEL       = os.getenv("FLAN_MODEL",            "google/flan-t5-base")
FLAN_REPLY_MODEL_DIR  = os.getenv("FLAN_REPLY_MODEL_DIR",  "models/flan_reply")

app = FastAPI(title="EchoReview AI Service", version="4.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_custom_predictor  = None
_hf_pipeline       = None
_hf_model_name     = ""
_flan_tokenizer    = None
_flan_model        = None
_flan_is_finetuned = False
_anthropic_client  = None
_reply_engine      = "template"


# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# One gold (prompt → reply) pair per issue × sentiment combination.
# These are prepended to every FLAN inference call so the model understands
# the exact voice, structure, and length we want BEFORE seeing the real task.
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = {
    ("delivery", "Negative"): (
        "Write an apologetic customer service reply for Priya who left a 1-star delivery review: "
        "My order has been in transit for 3 weeks with no updates and no delivery.",
        "Hi Priya, we are truly sorry for the 3-week delay — this is completely unacceptable. "
        "Our logistics team has escalated your shipment as a priority. "
        "Please share your order number and we will arrange an immediate replacement or full refund. "
        "We sincerely apologise for this experience."
    ),
    ("billing", "Negative"): (
        "Write an apologetic customer service reply for Sunita who left a 1-star billing review: "
        "I was charged twice for the same subscription. 4999 appeared twice on my statement.",
        "Hi Sunita, we are so sorry about the duplicate charge — this is clearly an error. "
        "Our finance team is reviewing your account now and the duplicate will be fully refunded within 3 business days. "
        "We apologise for the stress this has caused."
    ),
    ("support", "Negative"): (
        "Write an apologetic customer service reply for Fatima who left a 1-star support review: "
        "Customer support hasn't responded to my 4 emails sent over 2 weeks.",
        "Hi Fatima, we are truly sorry for the silence — 4 unanswered emails over two weeks is completely unacceptable. "
        "I am personally escalating your case now and a senior team member will contact you within 2 hours. "
        "We will resolve your issue as a priority and sincerely apologise."
    ),
    ("product", "Negative"): (
        "Write an apologetic customer service reply for Arjun who left a 1-star product review: "
        "The blender stopped working after 2 uses. Motor makes a grinding noise.",
        "Hi Arjun, we are very sorry your blender failed after just two uses — this falls well below our quality standards. "
        "We are shipping a brand-new replacement to you today, no return required. "
        "We apologise for the inconvenience and are investigating this as a quality issue."
    ),
    ("refund", "Negative"): (
        "Write an apologetic customer service reply for Mohan who left a 1-star refund review: "
        "Returned item 3 weeks ago. Warehouse confirmed receipt. Refund still not processed.",
        "Hi Mohan, we sincerely apologise for the 3-week delay — you should not have had to wait this long. "
        "Your refund is being processed as a priority today and will appear within 2 business days. "
        "We are sorry for the stress this has caused."
    ),
    ("general", "Negative"): (
        "Write an apologetic customer service reply for Ravi who left a 1-star general review: "
        "Worst experience ever. Nothing worked as promised and nobody helped me.",
        "Hi Ravi, we are very sorry for this experience — this is absolutely not the standard we hold ourselves to. "
        "Please reach out to our support team directly and we will personally ensure your issue is resolved as a priority. "
        "We appreciate your patience and apologise sincerely."
    ),
    ("general", "Positive"): (
        "Write a warm grateful customer service reply for Rahul who left a 5-star general review: "
        "Absolutely brilliant service. Ordered Monday, arrived Tuesday. Product exceeded expectations.",
        "Hi Rahul, thank you so much for this wonderful review — it truly made our team's day! "
        "Next-day delivery and a product that exceeds expectations is exactly what we aim for every time. "
        "We look forward to serving you again soon."
    ),
    ("general", "Neutral"): (
        "Write an empathetic constructive customer service reply for Amit who left a 3-star general review: "
        "Product is decent but delivery was 12 days late. Also overcharged by 150.",
        "Hi Amit, thank you for the honest feedback. We are sorry the delivery was late and you were overcharged — both should not have happened. "
        "We are refunding the overcharge immediately and investigating the delivery delay. "
        "We hope your next experience is much better."
    ),
}


def _get_few_shot(issue: str, sentiment: str) -> tuple:
    """Get the most relevant few-shot example for this issue+sentiment combo."""
    key = (issue, sentiment)
    if key in FEW_SHOT_EXAMPLES:
        return FEW_SHOT_EXAMPLES[key]
    # Fall back to same sentiment with general issue
    fallback = (issue, "Negative") if sentiment == "Negative" else ("general", sentiment)
    return FEW_SHOT_EXAMPLES.get(fallback, FEW_SHOT_EXAMPLES[("general", "Negative")])


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)

class GenerateReplyRequest(BaseModel):
    customerName:   str   = Field(default="Customer")
    text:           str   = Field(..., min_length=1)
    rating:         Optional[float] = None
    platform:       Optional[str]   = None
    sentimentScore: Optional[float] = None
    sentimentLabel: Optional[str]   = None

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
    durations = re.findall(r"\b(\d+\s*(?:day|week|month|hour|minute)s?)\b", text, re.IGNORECASE)
    if durations:
        details["duration"] = durations[0]
    amounts = re.findall(r"[₹$£€]?\s*\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:USD|GBP|EUR|INR|rupee))?", text)
    if amounts:
        details["amount"] = amounts[0].strip()
    return details


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def derive_label(score: float) -> str:
    if score < 40: return "Negative"
    if score < 60: return "Neutral"
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
    for prefix in [
        "reply:", "response:", "answer:", "write a reply:", "customer reply:",
        "generate a reply:", "here is a reply:", "here's a reply:",
        "customer service reply:", "professional reply:",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    return re.sub(r"\s+", " ", text).strip()

def _is_bad_reply(reply: str, original_text: str) -> bool:
    """Return True if reply is an echo, too short, looks like a reviewer comment, or copies the review."""
    if not reply or len(reply.strip()) < 30:
        return True

    # Contains instruction artifacts
    bad_phrases = [
        "write a", "generate a", "customer review", "sentiment:", "issue type:",
        "platform:", "star rating:", "instructions:", "task:", "prompt:", "tone:",
        "1-star", "2-star", "3-star", "4-star", "5-star",
    ]
    lower = reply.lower()
    if any(p in lower for p in bad_phrases):
        return True

    # Sounds like a reviewer, not a business (key tells of FLAN hallucination)
    reviewer_phrases = [
        "i'm not a fan", "i am not a fan", "i've been here", "i have been here",
        "i would not recommend", "i will not be coming back",
        "i ordered from here", "i bought this", "the food was",
        "definitely not coming back", "save your money",
    ]
    if any(p in lower for p in reviewer_phrases):
        return True

    # Too similar to original review — echo detection
    orig_words  = set(original_text.lower().split())
    reply_words = set(reply.lower().split())
    if len(orig_words) > 8:
        overlap = len(orig_words & reply_words) / len(orig_words)
        if overlap > 0.55:
            return True

    # Must start with Hi or Dear (business reply convention)
    if not (lower.startswith("hi ") or lower.startswith("dear ")):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE REPLY ENGINE  (always works, high quality, issue-specific)
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
    "billing":  "the billing concern",
    "support":  "the support experience you described",
    "product":  "the product issue",
    "refund":   "the refund difficulty",
    "general":  "the experience you shared",
}

def _template_reply(payload: Dict[str, Any]) -> str:
    name      = _first_name(payload.get("customerName", "Customer"))
    review    = (payload.get("text") or "").strip()
    platform  = (payload.get("platform") or "our platform").strip()
    sentiment = (payload.get("sentimentLabel") or "Neutral").strip()
    rating    = _rating_num(payload.get("rating"))
    issue     = detect_issue(review)
    specifics = extract_specifics(review)
    ref       = f" (ref: {specifics['reference']})" if specifics.get("reference") else ""

    if sentiment == "Positive" or rating >= 4:
        return (
            f"Hi {name}, thank you so much for this wonderful review! "
            f"We're genuinely thrilled to hear you had such a positive experience with {platform}. "
            f"Feedback like yours motivates our entire team every day. "
            f"We look forward to serving you again soon!"
        )

    if sentiment == "Negative" or rating <= 2:
        phrase     = _ISSUE_PHRASE.get(issue, "the issue")
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
# FLAN-T5 REPLY  (few-shot prompting)
# ─────────────────────────────────────────────────────────────────────────────
def _build_flan_prompt(payload: Dict[str, Any]) -> str:
    """
    Few-shot prompt: show FLAN one gold example of the expected output style,
    then give it the real task. This anchors the model to business-reply voice
    rather than reviewer voice.

    Format used by both base FLAN inference and the fine-tuned model.
    """
    name      = _first_name(payload.get("customerName", "Customer"))
    review    = (payload.get("text") or "").strip()[:300]
    rating    = _rating_num(payload.get("rating"))
    sentiment = (payload.get("sentimentLabel") or "Neutral").strip()
    issue     = detect_issue(review)

    # Tone word
    if sentiment == "Negative" or rating <= 2:
        tone = "apologetic"
    elif sentiment == "Positive" or rating >= 4:
        tone = "warm grateful"
    else:
        tone = "empathetic constructive"

    # Pick a relevant few-shot example
    ex_prompt, ex_reply = _get_few_shot(issue, sentiment)

    # Build few-shot prompt: Example + separator + real task
    full_prompt = (
        f"{ex_prompt}\n"
        f"{ex_reply}\n\n"
        f"Write a {tone} customer service reply for {name} who left a "
        f"{int(rating)}-star {issue} review: {review}"
    )
    return full_prompt


def _run_flan_local(payload: Dict[str, Any]) -> Optional[str]:
    global _flan_tokenizer, _flan_model
    if _flan_tokenizer is None or _flan_model is None:
        return None

    try:
        prompt = _build_flan_prompt(payload)
        inputs = _flan_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=350,      # longer to accommodate the few-shot example
            truncation=True,
        )
        device = next(_flan_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _flan_model.generate(
                **inputs,
                max_new_tokens=110,
                min_new_tokens=40,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=1.4,
                length_penalty=0.9,
            )

        reply = _flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = cleanup_reply(reply)

        if _is_bad_reply(reply, payload.get("text", "")):
            print(f"[FLAN] Bad reply detected → falling back to template. Was: {reply[:80]!r}")
            return None

        # Ensure it starts with Hi <name>
        name = _first_name(payload.get("customerName", "Customer"))
        if not reply.lower().startswith("hi "):
            reply = f"Hi {name}, {reply[0].lower()}{reply[1:]}"

        return reply

    except Exception as exc:
        print(f"[FLAN] Generation error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC REPLY  (tertiary, only if key set)
# ─────────────────────────────────────────────────────────────────────────────
def _build_claude_reply_prompt(payload: Dict[str, Any]) -> str:
    name      = (payload.get("customerName") or "Customer").strip()
    first     = name.split()[0]
    review    = (payload.get("text") or "").strip()
    platform  = (payload.get("platform") or "our platform").strip()
    rating    = payload.get("rating")
    sentiment = (payload.get("sentimentLabel") or "Neutral").strip()
    issue     = detect_issue(review)
    specifics = extract_specifics(review)
    rn        = _rating_num(rating)

    tone = (
        "sincere apology opening, acknowledge the specific problem named in the review, concrete next step with timeline"
        if (sentiment == "Negative" or rn <= 2) else
        "warm gratitude, reference exactly what they praised, make them feel genuinely valued"
        if (sentiment == "Positive" or rn >= 4) else
        "empathy for the mixed experience, acknowledge the specific concern, clear next step"
    )

    spec_parts = [f"{k}: {v}" for k, v in specifics.items()]
    spec_str = ", ".join(spec_parts) if spec_parts else "none"

    return (
        f"You are a customer success manager replying to a {issue} review on behalf of a business.\n\n"
        f"Customer: {name} | Platform: {platform} | Rating: {rating}/5 | Sentiment: {sentiment}\n"
        f"Extracted details: {spec_str}\n"
        f"Review: \"{review}\"\n\n"
        f"Structure: {tone}\n\n"
        f"RULES (mandatory):\n"
        f"1. Start with exactly: Hi {first},\n"
        f"2. Reference the specific problem or praise from the review — not generic phrases\n"
        f"3. 60–90 words total. Plain prose. No bullet points.\n"
        f"4. No sign-off. No mention of AI. Sound like a real human.\n\n"
        f"Write ONLY the reply:"
    )

def _run_anthropic_reply(payload: Dict[str, Any]) -> Optional[str]:
    if _anthropic_client is None:
        return None
    try:
        msg   = _anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": _build_claude_reply_prompt(payload)}],
        )
        reply = cleanup_reply(msg.content[0].text)
        return reply if reply and len(reply) > 30 else None
    except Exception as exc:
        print(f"[Anthropic] Reply error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT CASCADE
# ─────────────────────────────────────────────────────────────────────────────
def _run_custom_model(text: str) -> Optional[Dict]:
    if _custom_predictor is None:
        return None
    r = _custom_predictor.predict(text)
    return {"sentimentScore": r.sentiment_score, "sentimentLabel": r.sentiment_label,
            "sentimentProbs": r.sentiment_probs, "aspectLabel": r.aspect_label,
            "aspectProbs": r.aspect_probs, "modelSource": "custom-distilbert"}

def _run_nlptown(text: str) -> Optional[Dict]:
    if _hf_pipeline is None or "nlptown" not in _hf_model_name:
        return None
    result = _hf_pipeline(text[:512], truncation=True)[0]
    stars  = int(result["label"][0])
    conf   = float(result["score"])
    if stars <= 2:   label, score = "Negative", round((stars/2)*40*conf, 2)
    elif stars == 3: label, score = "Neutral",  round(40+20*conf, 2)
    else:            label, score = "Positive", round(60+((stars-3)/2)*40*conf, 2)
    return {"sentimentScore": min(100,max(0,score)), "sentimentLabel": label,
            "sentimentProbs": {label: round(conf*100,2)}, "aspectLabel": detect_issue(text),
            "aspectProbs": {}, "modelSource": "nlptown-5star"}

def _run_distilbert_binary(text: str) -> Optional[Dict]:
    if _hf_pipeline is None:
        return None
    result = _hf_pipeline(text[:512], truncation=True)[0]
    raw    = result["label"].upper()
    sc     = float(result["score"])
    score  = round(min(sc*100,100),2) if raw == "POSITIVE" else round(min((1-sc)*100,100),2)
    return {"sentimentScore": score, "sentimentLabel": derive_label(score),
            "sentimentProbs": {}, "aspectLabel": detect_issue(text),
            "aspectProbs": {}, "modelSource": "distilbert-binary"}

def _keyword_fallback(text: str) -> Dict:
    lower = text.lower()
    pos = sum(1 for w in ["great","excellent","amazing","love","best","wonderful",
        "fantastic","perfect","happy","satisfied","brilliant","impressed",
        "fast","smooth","easy","helpful","recommend","outstanding"] if w in lower)
    neg = sum(1 for w in ["terrible","awful","worst","hate","horrible","disappointed",
        "bad","poor","useless","broken","never","refused","ignored",
        "defective","damaged","overcharged","late","waiting","rude"] if w in lower)
    if pos > neg:   score, label = 75.0, "Positive"
    elif neg > pos: score, label = 22.0, "Negative"
    else:           score, label = 50.0, "Neutral"
    return {"sentimentScore": score, "sentimentLabel": label,
            "sentimentProbs": {}, "aspectLabel": detect_issue(text),
            "aspectProbs": {}, "modelSource": "keyword-fallback"}

def analyze_text(text: str) -> Dict:
    return (_run_custom_model(text) or _run_nlptown(text)
            or _run_distilbert_binary(text) or _keyword_fallback(text))


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED REPLY GENERATOR
# Priority: Fine-tuned FLAN / Base FLAN (few-shot) → Template → Anthropic
# ─────────────────────────────────────────────────────────────────────────────
def generate_reply(payload: Dict[str, Any], force_template: bool = False) -> Dict[str, Any]:
    issue  = detect_issue(payload.get("text", ""))
    reply  = None
    source = "template"

    if not force_template:
        flan_reply = _run_flan_local(payload)
        if flan_reply:
            model_name = "flan-t5-finetuned" if _flan_is_finetuned else FLAN_BASE_MODEL.split("/")[-1]
            reply  = flan_reply
            source = f"flan-t5/{model_name}"

    if not reply:
        reply  = _template_reply(payload)
        source = "template"

    # Anthropic upgrade only when template was used and key is available
    if source == "template" and _anthropic_client is not None and not force_template:
        ar = _run_anthropic_reply(payload)
        if ar:
            reply  = ar
            source = f"anthropic/{CLAUDE_MODEL}"

    return {
        "reply":         cleanup_reply(reply),
        "issueCategory": issue,
        "source":        source,
        "metadata": {
            "customerName":   payload.get("customerName"),
            "platform":       payload.get("platform"),
            "sentimentLabel": payload.get("sentimentLabel"),
            "sentimentScore": payload.get("sentimentScore"),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE STATS  (fast keyword-based, no AI needed)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_issue_stats(reviews: List[Dict]) -> Dict[str, Any]:
    total = len(reviews)
    if not total:
        return {}

    issue_counts    = Counter()
    issue_sentiment = defaultdict(list)
    issue_ratings   = defaultdict(list)
    platform_counts = Counter()
    recent_issues   = Counter()
    cutoff          = datetime.utcnow() - timedelta(days=7)

    for r in reviews:
        text    = r.get("text", "")
        issue   = r.get("issueCategory") or detect_issue(text)
        sent    = r.get("sentimentScore", 50)
        rating  = _rating_num(r.get("rating", 3))
        plat    = r.get("platform", "Other")
        date_str = r.get("date") or r.get("createdAt", "")

        issue_counts[issue]    += 1
        issue_sentiment[issue].append(sent)
        issue_ratings[issue].append(rating)
        platform_counts[plat]  += 1

        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.replace(tzinfo=None) >= cutoff:
                recent_issues[issue] += 1
        except Exception:
            pass

    result = {}
    for issue, count in issue_counts.items():
        avg_sent   = round(sum(issue_sentiment[issue]) / len(issue_sentiment[issue]), 1)
        avg_rating = round(sum(issue_ratings[issue]) / len(issue_ratings[issue]), 2)
        result[issue] = {
            "count":      count,
            "percentage": round(count / total * 100, 1),
            "avgSentiment": avg_sent,
            "avgRating":    avg_rating,
            "severity": (
                "critical" if avg_sent < 25 or avg_rating <= 1.5 else
                "high"     if avg_sent < 40 or avg_rating <= 2.5 else
                "medium"   if avg_sent < 55 or avg_rating <= 3.5 else
                "low"
            ),
            "recentCount": recent_issues.get(issue, 0),
            "trending":    recent_issues.get(issue, 0) > (count * 0.3),
        }

    return {
        "issueBreakdown": result,
        "topIssue":       issue_counts.most_common(1)[0][0] if issue_counts else "general",
        "platformBreakdown": dict(platform_counts.most_common()),
        "totalReviews":   total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI INSIGHTS  — Claude NLP analysis
# Sends actual review TEXTS to Claude for real language understanding:
#   • Recurring complaint themes (extracted from text, not just keywords)
#   • Churn risk signals (customers threatening to leave)
#   • Urgent unresolved issues needing immediate attention
#   • Positive signals and what to amplify
#   • Specific actionable recommendations
# Falls back to keyword analysis if no Anthropic key.
# ─────────────────────────────────────────────────────────────────────────────
def _build_insights_prompt(reviews: List[Dict], stats: Dict) -> str:
    total      = len(reviews)
    neg_count  = sum(1 for r in reviews if r.get("sentimentLabel") == "Negative")
    unresolved = sum(1 for r in reviews if r.get("status") == "new")

    # Send up to 30 review texts to Claude, prioritising negative + unresolved
    priority = sorted(
        reviews,
        key=lambda r: (
            r.get("status") == "new",
            r.get("sentimentLabel") == "Negative",
            -(r.get("priorityScore") or 0),
        ),
        reverse=True,
    )[:30]

    review_block = "\n".join(
        f'[{i+1}] {r.get("rating", "?")}★ | {r.get("platform","?")} | '
        f'{r.get("sentimentLabel","?")} | {r.get("issueCategory","general")} | '
        f'Status: {r.get("status","?")} | '
        f'"{(r.get("text",""))[:180]}"'
        for i, r in enumerate(priority)
    )

    breakdown = stats.get("issueBreakdown", {})
    issue_summary = " | ".join(
        f"{k}: {v['count']} ({v['percentage']}%, avg {v['avgRating']}★, {v['severity']})"
        for k, v in sorted(breakdown.items(), key=lambda x: -x[1]["count"])
    )

    return f"""You are a senior customer experience analyst. Analyse these {total} customer reviews and provide deep, actionable insights.

AGGREGATE STATS:
Total reviews: {total} | Negative: {neg_count} ({round(neg_count/max(total,1)*100,1)}%) | Unresolved: {unresolved}
Issue breakdown: {issue_summary or "not computed"}

REVIEW SAMPLE (prioritised by urgency — actual customer language):
{review_block}

YOUR TASK: Perform genuine NLP analysis on the actual review texts above. Look for:
1. Recurring complaint THEMES in the language (not just keyword counts — what are customers actually saying?)
2. CHURN RISK signals (customers saying they're leaving, switching, never returning)
3. URGENT issues that need a reply within 24h (high priority scores, angry tone, specific problems named)
4. POSITIVE patterns worth amplifying (what do happy customers consistently praise?)
5. OPERATIONAL insights (patterns suggesting systemic process failures, not one-off issues)

Respond ONLY with a valid JSON object. No markdown fences, no explanation outside JSON.

{{
  "executiveSummary": "3-4 sentences summarising overall review health, biggest risk, and biggest opportunity. Be specific — use numbers.",
  "insights": [
    {{
      "type": "theme|churn_risk|urgent|positive|operational",
      "title": "concise title under 10 words",
      "detail": "2-3 sentences with specific examples from the review texts above. Quote or paraphrase actual customer language where useful.",
      "severity": "critical|high|medium|low",
      "affectedCount": 0
    }}
  ],
  "recommendations": [
    {{
      "action": "one specific action to take",
      "impact": "expected outcome",
      "timeframe": "immediate|this week|this month",
      "priority": "critical|high|medium"
    }}
  ],
  "churnRiskCount": 0,
  "urgentReplyCount": 0,
  "topComplaintTheme": "the most repeated complaint theme in plain English"
}}

Provide 5-7 insights and 4-6 recommendations. Be specific, reference actual review language, and focus on insights that keyword counting alone would miss."""


def _keyword_insights(reviews: List[Dict]) -> Dict[str, Any]:
    """Fallback when Anthropic is not available."""
    stats    = _compute_issue_stats(reviews)
    breakdown = stats.get("issueBreakdown", {})
    total    = stats.get("totalReviews", 1)

    neg_count  = sum(1 for r in reviews if r.get("sentimentLabel") == "Negative")
    unresolved = sum(1 for r in reviews if r.get("status") == "new")
    neg_pct    = round(neg_count / max(total, 1) * 100, 1)

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
            "detail": f"These issue categories have increased in the last 7 days, suggesting a possible operational change or external event is affecting customer experience.",
            "severity": "high",
            "affectedCount": sum(breakdown[t]["recentCount"] for t in trending),
        })

    if neg_pct > 40:
        insights.append({
            "type": "urgent",
            "title": f"{neg_pct:.0f}% of reviews are negative — above safe threshold",
            "detail": f"A negative review rate above 40% significantly impacts platform rankings and brand perception. Immediate intervention on top issues is recommended.",
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
            "detail": f"A large backlog of unanswered reviews signals poor responsiveness to new customers and may affect your platform ranking.",
            "severity": "high" if unresolved > 20 else "medium",
            "affectedCount": unresolved,
        })
        recommendations.append({
            "action": f"Use AI Reply to clear the {unresolved}-review backlog",
            "impact": "Improve response rate metric and customer trust",
            "timeframe": "this week",
            "priority": "high",
        })

    top_issue_name = breakdown and max(breakdown.items(), key=lambda x: x[1]["count"])[0] or "general"

    return {
        "executiveSummary": (
            f"Analysis of {total} reviews shows {neg_pct}% negative sentiment. "
            f"The top issue is {top_issue_name} ({breakdown.get(top_issue_name, {}).get('percentage', 0)}% of reviews). "
            f"{unresolved} reviews are still awaiting a reply. "
            f"Note: connect an Anthropic API key for deeper AI-powered insights from actual review text."
        ),
        "insights":           insights,
        "recommendations":    recommendations,
        "stats":              stats,
        "churnRiskCount":     0,
        "urgentReplyCount":   unresolved,
        "topComplaintTheme":  top_issue_name,
        "generatedBy":        "keyword-analysis",
        "generatedAt":        datetime.utcnow().isoformat(),
    }


def _claude_insights(reviews: List[Dict]) -> Optional[Dict[str, Any]]:
    """Send real review texts to Claude for NLP-level insights."""
    if _anthropic_client is None:
        return None

    try:
        stats  = _compute_issue_stats(reviews)
        prompt = _build_insights_prompt(reviews, stats)

        msg  = _anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = msg.content[0].text.strip()
        raw  = re.sub(r"^```(?:json)?\s*", "", raw)
        raw  = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        data["stats"]       = stats
        data["generatedBy"] = f"anthropic/{CLAUDE_MODEL}"
        data["generatedAt"] = datetime.utcnow().isoformat()
        return data

    except json.JSONDecodeError as e:
        print(f"[Insights] Claude returned invalid JSON: {e}")
        return None
    except Exception as exc:
        print(f"[Insights] Claude error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_everything():
    global _custom_predictor, _hf_pipeline, _hf_model_name
    global _flan_tokenizer, _flan_model, _flan_is_finetuned
    global _anthropic_client, _reply_engine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EchoReview AI] v4.2 starting on {device}…")

    # 1. Custom sentiment model
    try:
        from sentiment_model import SentimentPredictor
        _custom_predictor = SentimentPredictor.from_directory(CUSTOM_MODEL_DIR, device)
        print(f"[EchoReview AI] ✓ Custom sentiment model loaded")
    except FileNotFoundError:
        print(f"[EchoReview AI] ℹ Custom model not found — run `python train.py` first. Using HF fallback.")
        _custom_predictor = None
    except Exception as exc:
        print(f"[EchoReview AI] ✗ Custom model error: {exc}")
        _custom_predictor = None

    # 2. HuggingFace sentiment fallback
    if _custom_predictor is None:
        try:
            from transformers import pipeline as hf_pipeline
            _hf_pipeline   = hf_pipeline("text-classification", model=HF_FALLBACK_MODEL,
                                          device=0 if torch.cuda.is_available() else -1)
            _hf_model_name = HF_FALLBACK_MODEL
            print(f"[EchoReview AI] ✓ HF sentiment fallback: {HF_FALLBACK_MODEL}")
        except Exception as exc:
            print(f"[EchoReview AI] ✗ HF fallback failed: {exc}")

    # 3. Try fine-tuned FLAN reply model first
    flan_loaded = False
    if os.path.isdir(FLAN_REPLY_MODEL_DIR):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"[EchoReview AI] Loading fine-tuned FLAN reply model from {FLAN_REPLY_MODEL_DIR}…")
            _flan_tokenizer    = AutoTokenizer.from_pretrained(FLAN_REPLY_MODEL_DIR)
            _flan_model        = AutoModelForSeq2SeqLM.from_pretrained(FLAN_REPLY_MODEL_DIR)
            _flan_model.to(device)
            _flan_model.eval()
            _flan_is_finetuned = True
            _reply_engine      = "flan-t5/flan-t5-finetuned"
            flan_loaded        = True
            print(f"[EchoReview AI] ✓ Fine-tuned FLAN reply model loaded (best quality)")
        except Exception as exc:
            print(f"[EchoReview AI] ✗ Fine-tuned model load failed: {exc} — falling back to base FLAN")

    # 4. Fall back to base FLAN with few-shot prompting
    if not flan_loaded:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"[EchoReview AI] Loading base FLAN: {FLAN_BASE_MODEL} (with few-shot prompting)…")
            _flan_tokenizer    = AutoTokenizer.from_pretrained(FLAN_BASE_MODEL)
            _flan_model        = AutoModelForSeq2SeqLM.from_pretrained(FLAN_BASE_MODEL)
            _flan_model.to(device)
            _flan_model.eval()
            _flan_is_finetuned = False
            _reply_engine      = f"flan-t5/{FLAN_BASE_MODEL.split('/')[-1]}-fewshot"
            print(f"[EchoReview AI] ✓ Base FLAN loaded with few-shot prompting")
        except Exception as exc:
            print(f"[EchoReview AI] ✗ FLAN load failed: {exc} — using template only")
            _flan_tokenizer = None
            _flan_model     = None
            _reply_engine   = "template"

    # 5. Anthropic client (optional)
    if _ANTHROPIC_PKG and ANTHROPIC_API_KEY:
        try:
            _anthropic_client = _Anthropic(api_key=ANTHROPIC_API_KEY)
            print(f"[EchoReview AI] ✓ Anthropic client ready (reply upgrade + AI insights)")
        except Exception as exc:
            print(f"[EchoReview AI] ✗ Anthropic init failed: {exc}")
    else:
        print("[EchoReview AI] ℹ Anthropic not configured — set ANTHROPIC_API_KEY to enable AI insights")

    print(f"\n[EchoReview AI] Sentiment : {'custom-distilbert' if _custom_predictor else _hf_model_name or 'keyword'}")
    print(f"[EchoReview AI] Reply     : {_reply_engine} {'(fine-tuned ✓)' if _flan_is_finetuned else '(few-shot)'}")
    print(f"[EchoReview AI] Insights  : {'Claude NLP ✓' if _anthropic_client else 'keyword-analysis'}")
    print(f"[EchoReview AI] Ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True, "service": "EchoReview AI Service", "version": "4.2.0",
        "sentimentEngine": (
            "custom-distilbert" if _custom_predictor
            else _hf_model_name if _hf_pipeline else "keyword-fallback"
        ),
        "replyEngine":      _reply_engine,
        "flanLoaded":       _flan_model is not None,
        "flanFinetuned":    _flan_is_finetuned,
        "anthropicEnabled": _anthropic_client is not None,
        "insightsEngine":   "anthropic-nlp" if _anthropic_client else "keyword-analysis",
    }

@app.post("/analyze")
def analyze_sentiment(req: AnalyzeRequest):
    text   = req.text.strip()
    result = analyze_text(text)
    issue  = result.get("aspectLabel") or detect_issue(text)
    return {
        "text": text,
        "sentimentScore": result["sentimentScore"],
        "sentimentLabel": result["sentimentLabel"],
        "sentimentProbs": result.get("sentimentProbs", {}),
        "issueCategory":  issue,
        "aspectProbs":    result.get("aspectProbs", {}),
        "modelSource":    result.get("modelSource", "unknown"),
    }

@app.post("/generate-reply")
def gen_reply(req: GenerateReplyRequest, mode: str = "auto"):
    return generate_reply(req.model_dump(), force_template=(mode == "template"))

@app.post("/generate-reply/template")
def gen_reply_template(req: GenerateReplyRequest):
    issue = detect_issue(req.text)
    return {
        "reply":         _template_reply(req.model_dump()),
        "issueCategory": issue,
        "source":        "template",
        "metadata": {"customerName": req.customerName, "sentimentLabel": req.sentimentLabel},
    }

@app.post("/insights")
def get_insights(req: InsightsRequest):
    """
    Real AI insights: sends actual review texts to Claude for NLP analysis.
    Falls back to keyword analysis if no Anthropic key is set.
    """
    result = _claude_insights(req.reviews) or _keyword_insights(req.reviews)
    return result

@app.post("/issues/summary")
def issues_summary(req: IssuesSummaryRequest):
    return _compute_issue_stats(req.reviews)