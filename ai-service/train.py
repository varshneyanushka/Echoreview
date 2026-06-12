"""
train.py  – Fine-tune DistilBERT for customer review sentiment analysis
──────────────────────────────────────────────────────────────────────────────

This script:
  1. Loads the built-in labeled customer-review dataset (180 examples)
     OR your own CSV file (--data path/to/reviews.csv)
  2. Fine-tunes a DistilBERT model on two tasks simultaneously:
       • Sentiment (Negative / Neutral / Positive)
       • Aspect    (delivery / billing / support / product / refund / general)
  3. Evaluates on a held-out validation split
  4. Saves the best checkpoint to  models/sentiment_model/

Usage
─────
  # Train on built-in dataset:
  python train.py

  # Train on your own CSV (must have columns: text, sentiment, aspect):
  python train.py --data data/my_reviews.csv

  # Adjust hyper-parameters:
  python train.py --epochs 6 --batch-size 16 --lr 3e-5

Requirements
────────────
  pip install torch transformers scikit-learn pandas tqdm
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

from sentiment_model import (
    ASPECT_IDX, ASPECT_LABELS,
    SENTIMENT_IDX, SENTIMENT_LABELS,
    CustomerReviewSentimentModel,
    SentimentPredictor,
)

# ── Seed ─────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN LABELED DATASET  (180 customer reviews)
# Each entry: (text, sentiment, aspect)
# sentiment: "Positive" | "Neutral" | "Negative"
# aspect:    "delivery" | "billing" | "support" | "product" | "refund" | "general"
# ─────────────────────────────────────────────────────────────────────────────
BUILTIN_DATA: List[Tuple[str, str, str]] = [
    # ── NEGATIVE / DELIVERY ──────────────────────────────────────────────────
    ("My order has been sitting in transit for 3 weeks. No updates, no delivery.", "Negative", "delivery"),
    ("Package marked as delivered but never arrived. My neighbours don't have it either.", "Negative", "delivery"),
    ("I paid for express delivery and it arrived 6 days late. Useless.", "Negative", "delivery"),
    ("The courier lost my parcel. No compensation offered. Absolutely terrible.", "Negative", "delivery"),
    ("Item dispatched 2 weeks ago still hasn't arrived. Tracking shows it left the warehouse.", "Negative", "delivery"),
    ("Late delivery ruined my event. I needed this for a birthday party yesterday.", "Negative", "delivery"),
    ("Wrong item delivered. I ordered a blue mug and received a red plate.", "Negative", "delivery"),
    ("Shipping took 18 days instead of the promised 3-5 business days.", "Negative", "delivery"),
    ("Delivery person left my package in the rain. Everything was soaked and ruined.", "Negative", "delivery"),
    ("Three separate items from my order arrived on three different days over 2 weeks.", "Negative", "delivery"),
    ("My shipment was returned to sender without any attempt to deliver it.", "Negative", "delivery"),
    ("Parcel arrived but the box was completely crushed. Contents damaged.", "Negative", "delivery"),

    # ── NEGATIVE / BILLING ───────────────────────────────────────────────────
    ("Charged twice for the same order. Need an immediate refund.", "Negative", "billing"),
    ("I was billed ₹4999 for the annual plan I never signed up for.", "Negative", "billing"),
    ("The sale price wasn't applied at checkout. Charged full price instead.", "Negative", "billing"),
    ("My card was charged 3 times for a single transaction.", "Negative", "billing"),
    ("Auto-renewal happened without a reminder email. I didn't want to renew.", "Negative", "billing"),
    ("Invoice shows a service fee of ₹499 that was never disclosed during checkout.", "Negative", "billing"),
    ("Promised a 20% discount but the final amount didn't reflect it.", "Negative", "billing"),
    ("I cancelled my subscription but was still charged the next month.", "Negative", "billing"),
    ("Currency conversion was wrong. I was charged more than the listed price.", "Negative", "billing"),
    ("Received an invoice for someone else's order. Serious data mix-up.", "Negative", "billing"),
    ("The promo code was accepted at checkout but not deducted from my bill.", "Negative", "billing"),
    ("My account was charged even though I paused my subscription.", "Negative", "billing"),

    # ── NEGATIVE / SUPPORT ───────────────────────────────────────────────────
    ("Customer support hasn't responded to my 4 emails sent over 2 weeks.", "Negative", "support"),
    ("The agent was rude and dismissive when I explained my problem.", "Negative", "support"),
    ("Called support, waited 40 minutes and was then disconnected without resolution.", "Negative", "support"),
    ("Chat support kept sending me to different departments with no one taking ownership.", "Negative", "support"),
    ("Raised a ticket 10 days ago, still no response. Terrible customer service.", "Negative", "support"),
    ("Support told me to email, email told me to call. Going in circles.", "Negative", "support"),
    ("The helpdesk agent couldn't answer basic questions about my account.", "Negative", "support"),
    ("I asked to speak to a manager and was refused without any explanation.", "Negative", "support"),
    ("Support closed my ticket without resolving the issue or notifying me.", "Negative", "support"),
    ("The chatbot is useless and there's no way to reach a real person.", "Negative", "support"),
    ("I've been a customer for 5 years and this is the worst support I've received.", "Negative", "support"),
    ("My support ticket was marked resolved but the issue is still present.", "Negative", "support"),

    # ── NEGATIVE / PRODUCT ───────────────────────────────────────────────────
    ("The blender stopped working after 2 uses. Motor makes a grinding noise.", "Negative", "product"),
    ("Product quality is terrible. Fell apart within a week of normal use.", "Negative", "product"),
    ("The earbuds completely stopped working after 6 days. Defective unit.", "Negative", "product"),
    ("Item arrived already damaged. Scratches all over the screen.", "Negative", "product"),
    ("Not as described at all. Photos show a different product than what was sent.", "Negative", "product"),
    ("The laptop stand broke the first time I tried to adjust it.", "Negative", "product"),
    ("Charging port on the device doesn't work out of the box.", "Negative", "product"),
    ("Material quality is very poor for the price. Looks cheap in person.", "Negative", "product"),
    ("One earbud has no sound. The product is 9 days old.", "Negative", "product"),
    ("The coating started peeling after the first wash. Very poor quality.", "Negative", "product"),
    ("Buttons on the remote are sticky and two don't respond at all.", "Negative", "product"),
    ("Arrived with parts missing. Box was sealed but items weren't all there.", "Negative", "product"),

    # ── NEGATIVE / REFUND ────────────────────────────────────────────────────
    ("Returned item 3 weeks ago, refund still not processed.", "Negative", "refund"),
    ("Warehouse confirmed they received my return but refund is nowhere to be seen.", "Negative", "refund"),
    ("Refund request denied despite the product being clearly defective.", "Negative", "refund"),
    ("They said 5-7 business days for the refund. It's been 20 days.", "Negative", "refund"),
    ("The returns portal is broken. I can't even initiate a return.", "Negative", "refund"),
    ("Was told I need to pay for return shipping even though the item was defective.", "Negative", "refund"),
    ("Refund was processed but for the wrong amount. ₹800 short.", "Negative", "refund"),
    ("My return was rejected because they claim the item arrived damaged at their warehouse.", "Negative", "refund"),
    ("After 6 calls, my refund is still pending. Worst experience with returns.", "Negative", "refund"),
    ("I can't find a return label in my order and support isn't helping.", "Negative", "refund"),

    # ── NEUTRAL ──────────────────────────────────────────────────────────────
    ("Product is okay for the price. Delivery was a bit slow but acceptable.", "Neutral", "general"),
    ("Average experience. Nothing particularly good or bad to report.", "Neutral", "general"),
    ("The app works but needs improvement in the navigation. Some features are hard to find.", "Neutral", "product"),
    ("Delivery was 2 days late but the item itself is decent quality.", "Neutral", "delivery"),
    ("Support eventually resolved my issue but it took 9 days and multiple follow-ups.", "Neutral", "support"),
    ("The product does what it says but the build quality could be better.", "Neutral", "product"),
    ("Okay experience overall. Wouldn't say it was great or terrible.", "Neutral", "general"),
    ("Delivery was slower than expected but tracking updates were clear.", "Neutral", "delivery"),
    ("The refund process was confusing but I got my money back in the end.", "Neutral", "refund"),
    ("Customer service was helpful but the wait time was too long.", "Neutral", "support"),
    ("Good value product but the packaging was excessive and wasteful.", "Neutral", "product"),
    ("Setup took longer than the instructions suggested but works fine now.", "Neutral", "product"),
    ("The subscription pricing is fair but the cancellation process is complex.", "Neutral", "billing"),
    ("Mixed feelings. The product quality is great but delivery expectations were misleading.", "Neutral", "delivery"),
    ("Support answered my query but took 5 days which isn't ideal.", "Neutral", "support"),
    ("Decent product. I wish the colours matched the website photos better.", "Neutral", "product"),
    ("Billing was sorted out eventually. Took more effort than it should.", "Neutral", "billing"),
    ("Not the best experience but not terrible. About average overall.", "Neutral", "general"),
    ("The app is stable but the interface feels outdated compared to competitors.", "Neutral", "product"),
    ("Order arrived intact. Product quality meets expectations, nothing more.", "Neutral", "general"),
    ("Delivery estimate was off by 3 days. The item itself is fine.", "Neutral", "delivery"),
    ("Support was polite but couldn't fully resolve my technical issue.", "Neutral", "support"),
    ("Acceptable experience. I've had better, I've had worse.", "Neutral", "general"),
    ("The product does what it claims but feels slightly overpriced.", "Neutral", "general"),
    ("Return was accepted but the credit took longer than stated to appear.", "Neutral", "refund"),
    ("Response time from support could be faster but the eventual answer was helpful.", "Neutral", "support"),

    # ── POSITIVE / DELIVERY ──────────────────────────────────────────────────
    ("Ordered Monday, arrived Tuesday morning. Incredible delivery speed!", "Positive", "delivery"),
    ("Packaging was excellent and the item arrived a day early. Very impressed.", "Positive", "delivery"),
    ("The delivery was fast, reliable, and the tracking updates were spot on.", "Positive", "delivery"),
    ("Real-time tracking and delivery exactly when promised. Fantastic.", "Positive", "delivery"),
    ("Order arrived to my Tier-2 city within 3 days. Didn't expect that at all.", "Positive", "delivery"),
    ("Delivery was seamless. Product well-packed and arrived in perfect condition.", "Positive", "delivery"),
    ("Express delivery lived up to its name. Had my item the same evening.", "Positive", "delivery"),

    # ── POSITIVE / SUPPORT ───────────────────────────────────────────────────
    ("Priya from customer service resolved my issue within the hour. Outstanding.", "Positive", "support"),
    ("The support team was patient, knowledgeable, and genuinely helpful.", "Positive", "support"),
    ("Best customer service experience I've had from any company. Highly recommend.", "Positive", "support"),
    ("Agent followed up the next day to make sure everything was resolved. That's rare.", "Positive", "support"),
    ("Support chat connected me with a real person in under 2 minutes.", "Positive", "support"),
    ("Had a billing confusion and support sorted it out completely within the same call.", "Positive", "support"),
    ("Customer service went above and beyond to make things right. Very grateful.", "Positive", "support"),
    ("The helpdesk team is responsive, friendly, and actually solves problems.", "Positive", "support"),

    # ── POSITIVE / PRODUCT ───────────────────────────────────────────────────
    ("Exceptional build quality. This product has been running flawlessly for 3 months.", "Positive", "product"),
    ("Best purchase I've made this year. Exceeded every expectation.", "Positive", "product"),
    ("The product quality is incredible. Exactly as described, maybe even better.", "Positive", "product"),
    ("Performance is outstanding. I use it daily and it's never let me down.", "Positive", "product"),
    ("Beautifully designed and incredibly functional. Worth every rupee.", "Positive", "product"),
    ("Premium quality that you can feel immediately. Packaging was also gorgeous.", "Positive", "product"),
    ("I was sceptical about the price but the quality completely justifies it.", "Positive", "product"),

    # ── POSITIVE / REFUND ────────────────────────────────────────────────────
    ("Refund processed within 48 hours, no questions asked. That's rare!", "Positive", "refund"),
    ("The return process was completely hassle-free. Refund appeared the next day.", "Positive", "refund"),
    ("They processed my refund faster than I expected. Excellent policy.", "Positive", "refund"),

    # ── POSITIVE / BILLING ───────────────────────────────────────────────────
    ("The pricing is transparent and the invoices are clear. No surprises.", "Positive", "billing"),
    ("Applied my discount code at checkout and it worked perfectly. Great deal.", "Positive", "billing"),
    ("Fair subscription pricing with an easy-to-understand billing structure.", "Positive", "billing"),

    # ── POSITIVE / GENERAL ───────────────────────────────────────────────────
    ("Seamless experience from order to delivery. Will absolutely buy again.", "Positive", "general"),
    ("Five stars. Everything worked perfectly and I'll be a returning customer.", "Positive", "general"),
    ("This company genuinely cares about customer experience. It shows in everything.", "Positive", "general"),
    ("I've ordered 6 times now and every experience has been consistently excellent.", "Positive", "general"),
    ("Refreshing to use a service that actually does what it promises.", "Positive", "general"),
    ("Brilliant all-round. The app, the product, the delivery, and the support.", "Positive", "general"),
    ("Recommended this to all my colleagues. Genuinely one of the best purchases.", "Positive", "general"),
    ("This is how online shopping should feel. Trustworthy, fast, and high quality.", "Positive", "general"),
    ("Signed up on a friend's recommendation. They were absolutely right. Excellent.", "Positive", "general"),
    ("Consistent quality every single time. This is my go-to for these products.", "Positive", "general"),
    ("The whole process from browsing to delivery felt premium and effortless.", "Positive", "general"),
    ("Great first impression. Everything was smooth and the product was perfect.", "Positive", "general"),

    # ── ADDITIONAL EDGE CASES ────────────────────────────────────────────────
    ("I love the product but the delivery instructions were confusing.", "Neutral", "delivery"),
    ("Billing is clear but the subscription tiers are hard to compare.", "Neutral", "billing"),
    ("Support is good during business hours but non-existent on weekends.", "Neutral", "support"),
    ("The product works but the assembly manual is poorly written.", "Neutral", "product"),
    ("Refund was smooth but I had to fight for it initially.", "Neutral", "refund"),
    ("Genuinely terrible. I've never been this disappointed in a purchase.", "Negative", "general"),
    ("Shocking customer experience. Will never buy from here again.", "Negative", "general"),
    ("Zero stars if I could. Everything that could go wrong, went wrong.", "Negative", "general"),
    ("Absolutely love everything about this brand. A total winner.", "Positive", "general"),
    ("Product does exactly what the description says. Pleased with my purchase.", "Positive", "general"),
    ("Simple, effective, and well-priced. Can't ask for more.", "Positive", "general"),
    ("Received the wrong size. Had to go through the return process.", "Negative", "product"),
    ("Subscription cancelled easily in one click. Appreciated the simplicity.", "Positive", "billing"),
    ("Order was cancelled without explanation or alternative offered.", "Negative", "delivery"),
    ("Response time for support is too slow. 6 days for a simple question.", "Negative", "support"),
    ("The warranty claim process is smooth and well-organised.", "Positive", "refund"),
    ("App crashes on Android. Works fine on iOS but not Android.", "Negative", "product"),
    ("Great product but the user manual is missing from the box.", "Neutral", "product"),
    ("The team replaced my defective unit within 48 hours. Impressive.", "Positive", "product"),
    ("Delivery notifications are clear and the real-time tracking is excellent.", "Positive", "delivery"),
    ("Been trying to cancel for weeks. The button just doesn't work.", "Negative", "billing"),
    ("The agent was patient and walked me through every step. Very helpful.", "Positive", "support"),
    ("Product arrived 30% damaged. Packaging was fine so it was pre-damaged.", "Negative", "product"),
    ("Refund pending for 25 days. Keep getting automated replies.", "Negative", "refund"),
    ("Quick delivery, quality product, friendly support. Three for three.", "Positive", "general"),
]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(
        self,
        texts:     List[str],
        sentiments: List[str],
        aspects:   List[str],
        tokenizer: DistilBertTokenizerFast,
        max_len:   int = 256,
    ):
        self.texts      = texts
        self.sentiments = [SENTIMENT_IDX[s] for s in sentiments]
        self.aspects    = [ASPECT_IDX[a]    for a in aspects]
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(self.sentiments[idx], dtype=torch.long),
            "aspect":         torch.tensor(self.aspects[idx],    dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def load_builtin() -> Tuple[List[str], List[str], List[str]]:
    texts, sentiments, aspects = zip(*BUILTIN_DATA)
    return list(texts), list(sentiments), list(aspects)


def load_csv(path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load a custom CSV.  Expects columns: text, sentiment, aspect
    sentiment: Positive | Neutral | Negative
    aspect:    delivery | billing | support | product | refund | general
    """
    import pandas as pd
    df = pd.read_csv(path)
    required = {"text", "sentiment", "aspect"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.dropna(subset=list(required))
    return df["text"].tolist(), df["sentiment"].tolist(), df["aspect"].tolist()


def build_dataloaders(
    texts:      List[str],
    sentiments: List[str],
    aspects:    List[str],
    tokenizer:  DistilBertTokenizerFast,
    val_ratio:  float,
    batch_size: int,
    max_len:    int,
) -> Tuple[DataLoader, DataLoader]:
    full_ds = ReviewDataset(texts, sentiments, aspects, tokenizer, max_len)
    val_n   = max(1, int(len(full_ds) * val_ratio))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(
        full_ds, [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  EchoReview AI — Sentiment Model Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}   LR: {args.lr}   Batch: {args.batch_size}")
    print(f"{'═'*60}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    if args.data:
        print(f"Loading data from {args.data} …")
        texts, sentiments, aspects = load_csv(args.data)
    else:
        print("Using built-in dataset …")
        texts, sentiments, aspects = load_builtin()

    print(f"Total examples : {len(texts)}")
    print(f"  Negative : {sentiments.count('Negative')}")
    print(f"  Neutral  : {sentiments.count('Neutral')}")
    print(f"  Positive : {sentiments.count('Positive')}\n")

    # ── Tokenizer + dataloaders ───────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_loader, val_loader = build_dataloaders(
        texts, sentiments, aspects,
        tokenizer,
        val_ratio=0.15,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    print(f"Train batches : {len(train_loader)}   Val batches : {len(val_loader)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CustomerReviewSentimentModel(dropout=args.dropout).to(device)

    # Freeze BERT layers for the first warm-up epoch (improves stability)
    def freeze_bert(freeze: bool):
        for p in model.bert.parameters():
            p.requires_grad = not freeze

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert.parameters(),         "lr": args.lr * 0.1},
            {"params": model.shared.parameters(),       "lr": args.lr},
            {"params": model.sentiment_head.parameters(), "lr": args.lr},
            {"params": model.aspect_head.parameters(),    "lr": args.lr},
        ],
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Loss: cross-entropy for both tasks; combine with weights
    sent_loss_fn  = nn.CrossEntropyLoss()
    aspect_loss_fn = nn.CrossEntropyLoss()
    SENT_W, ASPECT_W = 0.7, 0.3   # sentiment is the primary task

    best_val_acc = 0.0
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze BERT after first epoch
        freeze_bert(epoch == 1)

        # ── TRAIN ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct_sent = train_correct_asp = 0
        total_train = 0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_labels    = batch["sentiment"].to(device)
            asp_labels     = batch["aspect"].to(device)

            optimizer.zero_grad()
            sent_logits, asp_logits = model(input_ids, attention_mask)

            loss = (
                SENT_W  * sent_loss_fn(sent_logits,  sent_labels)
                + ASPECT_W * aspect_loss_fn(asp_logits, asp_labels)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = input_ids.size(0)
            train_loss         += loss.item() * bs
            train_correct_sent += (sent_logits.argmax(-1) == sent_labels).sum().item()
            train_correct_asp  += (asp_logits.argmax(-1)  == asp_labels).sum().item()
            total_train        += bs

        avg_train_loss = train_loss / total_train
        train_sent_acc = train_correct_sent / total_train
        train_asp_acc  = train_correct_asp  / total_train

        # ── VALIDATE ─────────────────────────────────────────────────────────
        model.eval()
        val_correct_sent = val_correct_asp = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                sent_labels    = batch["sentiment"].to(device)
                asp_labels     = batch["aspect"].to(device)

                sent_logits, asp_logits = model(input_ids, attention_mask)

                bs = input_ids.size(0)
                val_correct_sent += (sent_logits.argmax(-1) == sent_labels).sum().item()
                val_correct_asp  += (asp_logits.argmax(-1)  == asp_labels).sum().item()
                total_val        += bs

        val_sent_acc = val_correct_sent / max(1, total_val)
        val_asp_acc  = val_correct_asp  / max(1, total_val)
        elapsed      = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {avg_train_loss:.4f} | "
            f"Train Sent {train_sent_acc:.2%} Asp {train_asp_acc:.2%} | "
            f"Val Sent {val_sent_acc:.2%} Asp {val_asp_acc:.2%} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_sent_acc > best_val_acc:
            best_val_acc = val_sent_acc
            model.save(args.output)
            print(f"  ✓ Best model saved (val sent acc = {best_val_acc:.2%})")

    print(f"\n{'═'*60}")
    print(f"  Training complete. Best val sentiment accuracy: {best_val_acc:.2%}")
    print(f"  Model saved to: {args.output}")
    print(f"{'═'*60}\n")

    # ── Quick smoke test ──────────────────────────────────────────────────────
    print("Running smoke test on 5 examples …\n")
    predictor = SentimentPredictor.from_directory(args.output, device=device)
    test_cases = [
        "My order never arrived after 3 weeks. This is unacceptable.",
        "Delivery was fast and the product quality is excellent. Very happy!",
        "The product is okay but delivery took longer than expected.",
        "I was charged twice and support hasn't responded in 10 days.",
        "The support agent resolved my issue in under an hour. Fantastic.",
    ]
    for tc in test_cases:
        result = predictor.predict(tc)
        print(f"  [{result.sentiment_label:8s} {result.sentiment_score:5.1f}%] "
              f"[aspect: {result.aspect_label:10s}]  {tc[:60]}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EchoReview sentiment model")
    parser.add_argument("--data",       type=str,   default=None,                   help="Path to custom CSV (optional)")
    parser.add_argument("--output",     type=str,   default="models/sentiment_model", help="Output directory for model")
    parser.add_argument("--epochs",     type=int,   default=5,                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,                      help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-5,                    help="Peak learning rate")
    parser.add_argument("--max-len",    type=int,   default=256,                     help="Max token length")
    parser.add_argument("--dropout",    type=float, default=0.3,                     help="Dropout rate")
    args = parser.parse_args()
    train(args)
