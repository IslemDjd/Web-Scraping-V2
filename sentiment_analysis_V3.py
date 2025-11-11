# =============================================================
# Arabic Reviews Sentiment Analysis (V2.3 with Post-Rules)
# Author: Islem Dj (enhanced with confidence + lexicon rules)
# =============================================================

import os
import math
import json
import re
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "./Results/clean_arabic_reviews.csv"
OUTPUT_CSV = "sentiment_reviews_arabic_bert_V3.csv"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

BATCH_SIZE = 32
MAX_LENGTH = 256
MIN_CONF_FOR_POLAR = 0.60   # scores below this are candidates for neutral unless lexicon overrides
SHORT_TOKEN_POSITIVE = True # treat one-word praise as positive; set False to force neutral
SHORT_LEN_THRESHOLD = 6     # tokens <= 6 considered short

USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else -1

# -----------------------------
# Arabic marketplace lexicon
# -----------------------------
POS_LEX = [
    "ممتاز","رائع","جميل","اصلي","أصلي","موصى","أنصح","كويس","حلو",
    "افضل","سريع التوصيل","توصيل سريع","خدمة ممتازة","ممتازه","رائعه","اصليه"
]
NEG_LEX = [
    "مزيف","غير اصلي","مو اصلي","غش","احتيال","يسخن","سخونه","حراره","خربان",
    "مشكل","مشاكل","تالف","مكسور","كسر","مرتجع","ارجاع","استرداد","رفض","يرفض",
    "مفعل مسبقاً","مفعل مسبقا","بطاريه تنفد","بطارية تنفد","سيئ","سئ","رديء","أسوأ","اسوا","كارث"
]

# simple Arabic tokenization fallback
AR_SPLIT = re.compile(r"[\\s\\u0600-\\u06FF]+")  # splits on whitespace; keeps Arabic blocks together

def contains_any(text: str, phrases):
    return any(p in text for p in phrases)

def tokenize_len(text: str) -> int:
    # rough token count; robust enough for short-text policy
    toks = re.split(r"\\s+", text.strip())
    toks = [t for t in toks if t]
    return len(toks)

# -----------------------------
# Load data
# -----------------------------
print("📂 Loading cleaned Arabic reviews...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8")
if "clean_description" not in df.columns:
    raise ValueError("Missing 'clean_description' column in input CSV")

texts = df["clean_description"].astype(str).fillna("").str.strip()

# -----------------------------
# Load model + tokenizer
# -----------------------------
print("🤖 Loading Arabic sentiment transformer model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print(f"Device set to use {'gpu' if USE_GPU else 'cpu'}")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    truncation=True
)

# Label normalization
id2label = getattr(model.config, "id2label", None)
label_map = {
    "positive": "positive", "pos": "positive",
    "negative": "negative", "neg": "negative",
    "neutral":  "neutral",  "neu": "neutral"
}

def normalize_label(lbl: str) -> str:
    if not isinstance(lbl, str):
        return "neutral"
    u = lbl.upper()
    if id2label and u.startswith("LABEL_"):
        try:
            idx = int(u.split("_")[-1])
            lbl = id2label.get(idx, lbl)
        except Exception:
            pass
    return label_map.get(lbl.lower(), "neutral")

def pick_top(result_item):
    # Supports dict or list[dict]
    if isinstance(result_item, dict):
        return result_item.get("label", "neutral"), float(result_item.get("score", 0.0))
    if isinstance(result_item, list):
        best = max(result_item, key=lambda d: float(d.get("score", 0.0)))
        return best.get("label", "neutral"), float(best.get("score", 0.0))
    return "neutral", 0.0

def predict_batched(text_list):
    results = sentiment_pipe(
        text_list,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        top_k=1,
        return_all_scores=False
    )
    if isinstance(results, dict):
        results = [results]
    labels, scores = [], []
    for r in results:
        lbl, sc = pick_top(r)
        labels.append(normalize_label(lbl))
        scores.append(sc)
    return labels, scores

# -----------------------------
# Inference + post-rules
# -----------------------------
print("🧠 Performing sentiment analysis on Arabic reviews...")
pred_labels = []
pred_scores = []

with torch.no_grad():
    total = len(texts)
    for i in tqdm(range(0, total, BATCH_SIZE), total=math.ceil(total / BATCH_SIZE)):
        batch = texts.iloc[i:i+BATCH_SIZE].tolist()

        if all(len(t) == 0 for t in batch):
            pred_labels.extend(["neutral"] * len(batch))
            pred_scores.extend([1.0] * len(batch))
            continue

        labels, scores = predict_batched(batch)

        for t, lbl, sc in zip(batch, labels, scores):
            raw = t

            # Short-text policy
            if tokenize_len(raw) <= SHORT_LEN_THRESHOLD:
                if SHORT_TOKEN_POSITIVE:
                    # keep the model decision; if low confidence but contains strong POS_LEX, boost
                    if lbl == "neutral" and contains_any(raw, POS_LEX):
                        lbl = "positive"
                        sc = max(sc, 0.75)
                else:
                    # force very short to neutral unless NEG_LEX triggers
                    if not contains_any(raw, NEG_LEX):
                        lbl = "neutral"
                        sc = max(sc, 0.9)

            # Confidence fallback
            if lbl in ("positive", "negative") and sc < MIN_CONF_FOR_POLAR:
                lbl = "neutral"

            # Lexicon overrides (high precision)
            if contains_any(raw, NEG_LEX) and not contains_any(raw, POS_LEX):
                lbl = "negative"
                sc = max(sc, 0.9)
            elif contains_any(raw, POS_LEX) and not contains_any(raw, NEG_LEX):
                # Only override to positive if model wasn't already negative
                if lbl != "negative":
                    lbl = "positive"
                    sc = max(sc, 0.85)

            pred_labels.append(lbl)
            pred_scores.append(sc)

# -----------------------------
# Save results
# -----------------------------
df["Sentiment"] = pred_labels
df["Sentiment_confidence"] = pred_scores

meta = {
    "model_name": MODEL_NAME,
    "transformers_version": getattr(__import__("transformers"), "__version__", "unknown"),
    "torch_version": torch.__version__,
    "device": "cuda" if USE_GPU else "cpu",
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "min_conf_for_polar": MIN_CONF_FOR_POLAR,
    "short_token_positive": SHORT_TOKEN_POSITIVE,
    "short_len_threshold": SHORT_LEN_THRESHOLD
}
df.attrs["meta"] = meta

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Sentiment analysis complete! Saved as '{OUTPUT_CSV}'")
print(json.dumps(meta, ensure_ascii=False, indent=2))
print(df[["clean_description", "Sentiment", "Sentiment_confidence"]].head(10).to_string(index=False))
