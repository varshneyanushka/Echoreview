"""
train_reply.py
──────────────────────────────────────────────────────────────────────────────
Fine-tune google/flan-t5-base on customer service reply generation.
After training, the model knows the exact voice, structure, and format we want:
  - Starts with "Hi <name>,"
  - Apologies for the specific problem mentioned
  - Gives a concrete next step
  - Stays between 60-100 words
  - Sounds human, not robotic

This solves the "I'm not a fan of this place" hallucination — FLAN was
mimicking Yelp-reviewer voice because it had no examples of business replies.

Usage:
  python train_reply.py
  python train_reply.py --epochs 5 --output models/flan_reply

After training, the fine-tuned model is loaded automatically by main.py
when FLAN_REPLY_MODEL_DIR is set (or defaults to models/flan_reply).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

# ── Gold examples ─────────────────────────────────────────────────────────────
# Format: (prompt, gold_reply)
# Prompts use the same format as main.py _build_flan_prompt_fewshot()
# so the fine-tuned model generalises correctly at inference time.

GOLD_EXAMPLES: List[Tuple[str, str]] = [

    # ── DELIVERY / NEGATIVE ──────────────────────────────────────────────────
    (
        "Write an apologetic customer service reply for Priya who left a 1-star delivery review: "
        "My order has been in transit for 3 weeks with no updates and no delivery.",
        "Hi Priya, we are truly sorry for the 3-week delay — this is completely unacceptable and not the experience we promise. "
        "Our logistics team has escalated your shipment right now as a priority. "
        "Please share your order number with us and we will arrange an immediate replacement or full refund, whichever you prefer. "
        "We apologise again for the inconvenience."
    ),
    (
        "Write an apologetic customer service reply for Rohit who left a 1-star delivery review: "
        "Package marked delivered but never arrived. Neighbours don't have it either.",
        "Hi Rohit, we sincerely apologise — a package marked delivered but never received is completely unacceptable. "
        "We have raised this with our courier partner immediately and are investigating. "
        "In the meantime, we will send a replacement today. "
        "Please confirm your delivery address and we will make this right as quickly as possible."
    ),
    (
        "Write an apologetic customer service reply for Anjali who left a 2-star delivery review: "
        "I paid for express delivery but the item arrived 6 days late. Completely useless.",
        "Hi Anjali, we are very sorry your express delivery arrived so late — that is not the service you paid for and we understand your frustration. "
        "We will refund your express delivery charge immediately, no questions asked. "
        "We have also flagged this with our courier team to prevent it happening again. "
        "Thank you for bringing this to our attention."
    ),

    # ── BILLING / NEGATIVE ───────────────────────────────────────────────────
    (
        "Write an apologetic customer service reply for Sunita who left a 1-star billing review: "
        "I was charged twice for the same subscription. 4999 appeared twice on my bank statement.",
        "Hi Sunita, we are so sorry about the duplicate charge of ₹4,999 — this is clearly an error on our part and should not have happened. "
        "Our finance team is reviewing your account right now and the duplicate charge will be fully refunded within 3 business days. "
        "We apologise for the stress this has caused and will ensure it does not happen again."
    ),
    (
        "Write an apologetic customer service reply for Sanjay who left a 1-star billing review: "
        "My subscription was cancelled but I was still charged the next month.",
        "Hi Sanjay, we sincerely apologise for charging you after cancellation — this is a billing error and we take full responsibility. "
        "Your account has been reviewed and a full refund for the incorrect charge is being processed now. "
        "You should see it within 3-5 business days. "
        "We are sorry for the inconvenience and thank you for bringing this to our attention."
    ),
    (
        "Write an apologetic customer service reply for Meera who left a 2-star billing review: "
        "The promo code was accepted at checkout but not applied to my final bill.",
        "Hi Meera, we apologise for this — your promo code should have been applied and we understand how frustrating it is to see the wrong amount charged. "
        "Our team has applied the discount to your account and the difference will be refunded within 2 business days. "
        "Thank you for flagging this and we are fixing the checkout issue so it does not affect other customers."
    ),

    # ── SUPPORT / NEGATIVE ───────────────────────────────────────────────────
    (
        "Write an apologetic customer service reply for Fatima who left a 1-star support review: "
        "Customer support hasn't responded to my 4 emails sent over 2 weeks.",
        "Hi Fatima, we are truly sorry for the silence — 4 unanswered emails over two weeks is completely unacceptable and not how we want to treat our customers. "
        "I am personally escalating your case right now and a senior team member will contact you within the next 2 hours. "
        "We will resolve your issue as a priority. Please accept our sincere apologies."
    ),
    (
        "Write an apologetic customer service reply for Kiran who left a 1-star support review: "
        "The agent was rude and dismissive. Was put on hold 40 minutes and then disconnected.",
        "Hi Kiran, we are deeply sorry for the way you were treated — the experience you described is completely unacceptable and does not reflect our standards. "
        "We are reviewing this call immediately and taking corrective action. "
        "A senior manager will reach out to you directly today to resolve your original issue and ensure you feel valued as a customer."
    ),
    (
        "Write an apologetic customer service reply for Deepak who left a 2-star support review: "
        "Support told me to email, email told me to call. Going in circles for 9 days.",
        "Hi Deepak, we are very sorry for the runaround you experienced — being bounced between channels for 9 days is unacceptable and we take full responsibility. "
        "We have assigned your case directly to our senior support lead who will contact you within 24 hours and see your issue through to resolution. "
        "Thank you for your patience and we are sorry this took so long."
    ),

    # ── PRODUCT / NEGATIVE ───────────────────────────────────────────────────
    (
        "Write an apologetic customer service reply for Arjun who left a 1-star product review: "
        "The blender stopped working after 2 uses. Motor makes a grinding noise.",
        "Hi Arjun, we are very sorry your blender failed after just two uses — this falls well below the quality we stand behind. "
        "We are shipping a brand-new replacement to you today, and you do not need to return the faulty unit. "
        "Your warranty has also been extended to 3 years as an apology. "
        "We are sorry for the inconvenience and will investigate this production issue."
    ),
    (
        "Write an apologetic customer service reply for Sneha who left a 1-star product review: "
        "The earbuds stopped working after 6 days. One earbud is completely silent.",
        "Hi Sneha, we are truly sorry your earbuds failed after just 6 days — this is a clear manufacturing defect and absolutely not acceptable. "
        "We are sending a replacement pair today under your warranty, no return required. "
        "Our quality team has been notified of this batch issue. "
        "Please expect the replacement within 2 days and accept our sincerest apologies."
    ),

    # ── REFUND / NEGATIVE ────────────────────────────────────────────────────
    (
        "Write an apologetic customer service reply for Mohan who left a 1-star refund review: "
        "Returned item 3 weeks ago. Warehouse confirmed receipt. Refund still not processed.",
        "Hi Mohan, we sincerely apologise for the 3-week delay on your refund — you should not have had to wait this long after we confirmed receipt. "
        "Our finance team is processing your refund as a priority today and it will appear within 2 business days. "
        "We are sorry for the stress this has caused and thank you for your patience."
    ),
    (
        "Write an apologetic customer service reply for Rekha who left a 1-star refund review: "
        "Refund request denied even though the product was clearly defective.",
        "Hi Rekha, we apologise — a refund for a defective product should never have been denied, and we are overturning that decision immediately. "
        "Your full refund is being processed today and will appear within 3 business days. "
        "We are also reviewing how this decision was made to prevent it happening to other customers. "
        "We are sorry for the inconvenience."
    ),

    # ── POSITIVE ─────────────────────────────────────────────────────────────
    (
        "Write a warm grateful customer service reply for Rahul who left a 5-star general review: "
        "Absolutely brilliant service. Ordered Monday, arrived Tuesday morning. Product exceeded expectations.",
        "Hi Rahul, thank you so much for this wonderful review — it truly made our team's day! "
        "Next-day delivery and a product that exceeds expectations is exactly what we aim for every single time. "
        "We are thrilled you had such a great experience and we look forward to serving you again soon."
    ),
    (
        "Write a warm grateful customer service reply for Kavya who left a 5-star support review: "
        "Arjun from customer service resolved my issue within the hour. Patient, proactive, and followed up.",
        "Hi Kavya, thank you for the kind words — we are so glad Arjun could resolve your issue so quickly! "
        "Proactive, patient support is exactly the standard we hold ourselves to, and it is wonderful to hear it came through. "
        "We will pass your feedback on to Arjun directly. Thank you for taking the time to share this."
    ),
    (
        "Write a warm grateful customer service reply for Suresh who left a 5-star product review: "
        "Best purchase this year. Build quality is exceptional. Has been running flawlessly for 3 months.",
        "Hi Suresh, thank you so much for this — 3 months of flawless performance is exactly what we built this product for! "
        "Reviews like yours remind us why we care so much about quality. "
        "We are thrilled you are happy with your purchase and hope to see you back soon."
    ),

    # ── NEUTRAL ──────────────────────────────────────────────────────────────
    (
        "Write an empathetic constructive customer service reply for Amit who left a 3-star general review: "
        "Product is decent but delivery was 12 days instead of 5. Also overcharged by 150.",
        "Hi Amit, thank you for the honest feedback. We are sorry the delivery took longer than promised and that you were overcharged — both of these should not have happened. "
        "We are refunding the ₹150 overcharge immediately and investigating the delivery delay with our courier. "
        "We appreciate you giving us a chance to improve and hope your next experience is much better."
    ),
    (
        "Write an empathetic constructive customer service reply for Ritu who left a 3-star support review: "
        "Support eventually resolved my issue but it took 9 days and 6 follow-up messages.",
        "Hi Ritu, thank you for the feedback and for your patience. We are glad the issue was resolved but 9 days and 6 follow-ups is far too long — you should not have had to work that hard to get help. "
        "We are reviewing our response processes to make sure this does not happen again. "
        "We appreciate your honest feedback and hope to do better for you next time."
    ),
    (
        "Write an empathetic constructive customer service reply for Vikram who left a 3-star product review: "
        "App works fine on iOS but crashes on Android when I open the reports section.",
        "Hi Vikram, thank you for flagging this — our Android team has been notified and is investigating the crash in the reports section as a priority. "
        "We expect a fix in the next app update, which we are targeting within the next 5 days. "
        "We apologise for the inconvenience and will notify you as soon as the fix is live."
    ),

    # ── MIXED / EDGE CASES ────────────────────────────────────────────────────
    (
        "Write an apologetic customer service reply for Nitin who left a 1-star support review: "
        "My account was locked without warning and I cannot access my active subscriptions.",
        "Hi Nitin, we are very sorry your account was locked without warning — this must have been extremely disruptive, especially with active subscriptions. "
        "Our technical team is reviewing your account right now as a priority and you will receive access within the next 2 hours. "
        "We will also extend your subscription by 7 days to compensate for the downtime. "
        "We sincerely apologise for the inconvenience."
    ),
    (
        "Write an apologetic customer service reply for Geeta who left a 3-star delivery review: "
        "Good product but I paid for express delivery and it arrived in standard time.",
        "Hi Geeta, thank you for flagging this — if you paid for express delivery, that is exactly what you should have received. "
        "We are refunding your express delivery charge in full, which you will see within 2 business days. "
        "We are glad you are happy with the product itself and we apologise for the delivery mix-up."
    ),
]


# ── Dataset ───────────────────────────────────────────────────────────────────
class ReplyDataset(Dataset):
    def __init__(self, examples, tokenizer, max_input=256, max_target=150):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, reply = self.examples[idx]
        enc = self.tokenizer(
            prompt, max_length=self.max_input,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        # as_target_tokenizer() was removed in transformers >= 4.23
        # For T5/FLAN, the target tokenizer is identical to the input tokenizer
        tgt = self.tokenizer(
            text_target=reply, max_length=self.max_target,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }

# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*58}")
    print(f"  EchoReview — FLAN-T5 Reply Fine-tuning")
    print(f"  Base model : {args.base_model}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}")
    print(f"{'═'*58}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.to(device)

    # Augment examples by shuffling order (helps generalisation on small data)
    augmented = GOLD_EXAMPLES * args.repeat
    random.shuffle(augmented)

    dataset    = ReplyDataset(augmented, tokenizer)
    loader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    total_steps = len(loader) * args.epochs
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"Training on {len(augmented)} examples ({len(GOLD_EXAMPLES)} gold × {args.repeat} repeats)\n")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0    = time.time()
        model.train()
        total_loss = 0.0

        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch:02d}/{args.epochs}  loss={avg:.4f}  ({time.time()-t0:.1f}s)")

        if avg < best_loss:
            best_loss = avg
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  ✓ Saved best model (loss={best_loss:.4f}) → {args.output}")

    print(f"\n{'═'*58}")
    print(f"  Done. Best loss: {best_loss:.4f}")
    print(f"  Model saved to: {args.output}")
    print(f"{'═'*58}\n")

    # Smoke test
    print("Smoke test:\n")
    model.eval()
    tests = [
        ("Write an apologetic customer service reply for Mohit who left a 1-star delivery review: "
         "Delivery was awful, the package was lost in transit, and support has not helped.", "Negative"),
        ("Write a warm grateful customer service reply for Pooja who left a 5-star general review: "
         "Seamless experience from order to delivery. Best online shopping ever!", "Positive"),
        ("Write an empathetic constructive customer service reply for Amit who left a 3-star billing review: "
         "Billing is unclear and I was confused by the charges on my invoice.", "Neutral"),
    ]
    for prompt, label in tests:
        enc     = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=120, num_beams=4,
                no_repeat_ngram_size=3, repetition_penalty=1.3, early_stopping=True,
            )
        reply = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  [{label}]\n  → {reply}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/flan-t5-base")
    parser.add_argument("--output",     default="models/flan_reply")
    parser.add_argument("--epochs",     type=int,   default=8)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--repeat",     type=int,   default=6,
                        help="Repeat gold examples N times for augmentation")
    args = parser.parse_args()
    train(args)