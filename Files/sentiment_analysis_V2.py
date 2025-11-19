# =============================================================
# Arabic Reviews Sentiment Analysis using Transformer (V2.2)
# Author: Islem Dj (refactor with batching, confidence, guards)
# =============================================================

import os
import math
import json
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "./Results/clean_arabic_reviews.csv"
OUTPUT_CSV = "sentiment_reviews_arabic_bert_V2.csv"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
BATCH_SIZE = 32            # tune: 16–64 based on VRAM/CPU
MAX_LENGTH = 256           # 256–512; 256 is faster and sufficient for most reviews
MIN_CONF_FOR_POLAR = 0.60  # < 0.60 => force to neutral
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else -1

print("📂 Loading cleaned Arabic reviews...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8")
if "clean_description" not in df.columns:
    raise ValueError("Missing 'clean_description' column in input CSV")

texts = df["clean_description"].astype(str).fillna("").str.strip()

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

# Map model labels consistently
# Some environments yield 'LABEL_0/1/2' — use id2label to normalize
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

# Robust extractor to handle dict vs list-of-dicts
def pick_top(result_item):
    # result_item can be dict or list[dict]
    if isinstance(result_item, dict):
        return result_item.get("label", "neutral"), float(result_item.get("score", 0.0))
    if isinstance(result_item, list):
        # choose the highest score among candidates
        best = max(result_item, key=lambda d: float(d.get("score", 0.0)))
        return best.get("label", "neutral"), float(best.get("score", 0.0))
    # unexpected type
    return "neutral", 0.0

# Batched inference with structure-stable output
def predict_batched(text_list):
    results = sentiment_pipe(
        text_list,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        top_k=1,                 # force single best prediction per input
        return_all_scores=False  # avoid list-of-candidates structure
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
# Inference loop
# -----------------------------
print("🧠 Performing sentiment analysis on Arabic reviews...")
pred_labels = []
pred_scores = []

with torch.no_grad():
    total = len(texts)
    for i in tqdm(range(0, total, BATCH_SIZE), total=math.ceil(total / BATCH_SIZE)):
        batch = texts.iloc[i:i+BATCH_SIZE].tolist()

        # If an entire batch is empty, short-circuit
        if all(len(t) == 0 for t in batch):
            pred_labels.extend(["neutral"] * len(batch))
            pred_scores.extend([1.0] * len(batch))
            continue

        labels, scores = predict_batched(batch)

        # Post-process: force low-confidence polar outputs to neutral
        for t, lbl, sc in zip(batch, labels, scores):
            if len(t) == 0:
                pred_labels.append("neutral")
                pred_scores.append(1.0)
                continue
            if lbl in ("positive", "negative") and sc < MIN_CONF_FOR_POLAR:
                pred_labels.append("neutral")
            else:
                pred_labels.append(lbl)
            pred_scores.append(sc)

# -----------------------------
# Save results
# -----------------------------
df["Sentiment"] = pred_labels
df["Sentiment_confidence"] = pred_scores

# Metadata for reproducibility
meta = {
    "model_name": MODEL_NAME,
    "transformers_version": getattr(__import__("transformers"), "__version__", "unknown"),
    "torch_version": torch.__version__,
    "device": "cuda" if USE_GPU else "cpu",
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "min_conf_for_polar": MIN_CONF_FOR_POLAR
}
df.attrs["meta"] = meta

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Sentiment analysis complete! Saved as '{OUTPUT_CSV}'")
print(json.dumps(meta, ensure_ascii=False, indent=2))
print(df[["clean_description", "Sentiment", "Sentiment_confidence"]].head(10).to_string(index=False))
