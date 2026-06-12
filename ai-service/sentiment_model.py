"""
sentiment_model.py
──────────────────────────────────────────────────────────────────────────────
Custom DistilBERT model trained specifically for customer review sentiment.

Architecture
────────────
  DistilBERT (distilbert-base-uncased)
      └── [CLS] token representation
            ├── Shared projection (768 → 256, GELU, Dropout)
            ├── Sentiment head  → 3 classes  (Negative / Neutral / Positive)
            └── Aspect head     → 6 classes  (delivery / billing / support /
                                              product / refund / general)

Classes
────────
  CustomerReviewSentimentModel  – nn.Module, forward() returns two logit tensors
  SentimentPredictor            – high-level inference wrapper with scoring utils
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast


# ── Label maps ────────────────────────────────────────────────────────────────
SENTIMENT_LABELS: List[str] = ["Negative", "Neutral", "Positive"]
ASPECT_LABELS:    List[str] = ["delivery", "billing", "support",
                                "product",  "refund",  "general"]

SENTIMENT_IDX:   Dict[str, int] = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
ASPECT_IDX:      Dict[str, int] = {l: i for i, l in enumerate(ASPECT_LABELS)}


# ── Model ─────────────────────────────────────────────────────────────────────
class CustomerReviewSentimentModel(nn.Module):
    """
    Multi-task DistilBERT model.

    Inputs
    ------
    input_ids      (B, seq_len)  – tokenised text
    attention_mask (B, seq_len)  – 1 for real tokens, 0 for padding

    Returns
    -------
    sentiment_logits  (B, 3)   – un-normalised scores for 3 sentiment classes
    aspect_logits     (B, 6)   – un-normalised scores for 6 aspect classes
    """

    BASE_MODEL = "distilbert-base-uncased"

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(self.BASE_MODEL)
        hidden = self.bert.config.hidden_size  # 768

        # Shared representation layer
        self.shared = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Task-specific heads ───────────────────────────────────────────────
        # Sentiment: 3-class (Negative=0, Neutral=1, Positive=2)
        self.sentiment_head = nn.Linear(256, len(SENTIMENT_LABELS))

        # Aspect: multi-class single-label  (which ONE aspect is primary?)
        self.aspect_head    = nn.Linear(256, len(ASPECT_LABELS))

        self._init_heads()

    def _init_heads(self):
        for head in (self.sentiment_head, self.aspect_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr  = out.last_hidden_state[:, 0]   # [CLS] token → (B, 768)
        shared    = self.shared(cls_repr)           # (B, 256)
        return self.sentiment_head(shared), self.aspect_head(shared)

    # ── Serialisation helpers ─────────────────────────────────────────────────
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(directory, "model_weights.pt"))
        meta = {
            "base_model":      self.BASE_MODEL,
            "sentiment_labels": SENTIMENT_LABELS,
            "aspect_labels":   ASPECT_LABELS,
            "dropout":         0.3,
        }
        with open(os.path.join(directory, "model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Model] Saved to {directory}")

    @classmethod
    def load(cls, directory: str, device: torch.device) -> "CustomerReviewSentimentModel":
        meta_path = os.path.join(directory, "model_meta.json")
        weights_path = os.path.join(directory, "model_weights.pt")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No model_weights.pt found in {directory}")

        with open(meta_path) as f:
            meta = json.load(f)

        model = cls(dropout=meta.get("dropout", 0.3))
        model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        model.to(device)
        model.eval()
        return model


# ── Prediction result ─────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    text:               str
    sentiment_label:    str            # "Positive" | "Neutral" | "Negative"
    sentiment_score:    float          # 0–100  (probability × 100)
    sentiment_probs:    Dict[str, float] = field(default_factory=dict)  # all 3 probs
    aspect_label:       str  = "general"
    aspect_probs:       Dict[str, float] = field(default_factory=dict)
    raw_label:          str  = ""
    raw_score:          float = 0.0


# ── High-level predictor ──────────────────────────────────────────────────────
class SentimentPredictor:
    """
    Wraps the trained model for easy inference.

    Usage
    -----
    predictor = SentimentPredictor.from_directory("models/sentiment_model")
    result    = predictor.predict("Delivery took 3 weeks, terrible experience!")
    print(result.sentiment_label, result.sentiment_score)
    """

    MAX_LEN = 256

    def __init__(
        self,
        model:     CustomerReviewSentimentModel,
        tokenizer: DistilBertTokenizerFast,
        device:    torch.device,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

    # ── Factory methods ───────────────────────────────────────────────────────
    @classmethod
    def from_directory(
        cls,
        directory: str,
        device: Optional[torch.device] = None,
    ) -> "SentimentPredictor":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = CustomerReviewSentimentModel.load(directory, device)
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            CustomerReviewSentimentModel.BASE_MODEL
        )
        return cls(model=model, tokenizer=tokenizer, device=device)

    # ── Core inference ────────────────────────────────────────────────────────
    def predict(self, text: str) -> SentimentResult:
        text = text.strip()[:2000]     # cap length
        encoding = self.tokenizer(
            text,
            max_length=self.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            sentiment_logits, aspect_logits = self.model(input_ids, attention_mask)

        # ── Sentiment ─────────────────────────────────────────────────────────
        sent_probs  = torch.softmax(sentiment_logits[0], dim=-1).cpu().tolist()
        sent_idx    = int(torch.argmax(sentiment_logits[0]).item())
        sent_label  = SENTIMENT_LABELS[sent_idx]
        sent_score  = round(sent_probs[sent_idx] * 100, 2)
        sent_prob_d = {l: round(p * 100, 2) for l, p in zip(SENTIMENT_LABELS, sent_probs)}

        # Convert "Positive → high score, Negative → low score" convention
        # so sentimentScore = 0..100 where 100 = maximally positive
        if sent_label == "Positive":
            normalised_score = round(50 + sent_probs[2] * 50, 2)   # 50–100
        elif sent_label == "Negative":
            normalised_score = round(sent_probs[0] * 50, 2)        # 0–50
        else:
            normalised_score = round(35 + sent_probs[1] * 30, 2)   # 35–65

        # ── Aspect ───────────────────────────────────────────────────────────
        asp_probs   = torch.softmax(aspect_logits[0], dim=-1).cpu().tolist()
        asp_idx     = int(torch.argmax(aspect_logits[0]).item())
        asp_label   = ASPECT_LABELS[asp_idx]
        asp_prob_d  = {l: round(p * 100, 2) for l, p in zip(ASPECT_LABELS, asp_probs)}

        return SentimentResult(
            text             = text,
            sentiment_label  = sent_label,
            sentiment_score  = normalised_score,
            sentiment_probs  = sent_prob_d,
            aspect_label     = asp_label,
            aspect_probs     = asp_prob_d,
            raw_label        = sent_label.upper(),
            raw_score        = sent_score,
        )

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        return [self.predict(t) for t in texts]
