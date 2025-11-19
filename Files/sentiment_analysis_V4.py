# =============================================================
# Arabic Reviews Sentiment Analysis (V3.5: normalization + rules)
# Author: Islem Dj (enhanced with normalization, negation, logs)
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
OUTPUT_CSV = "sentiment_reviews_arabic_bert_V4.csv"
MISMATCH_CSV = "sentiment_V3_mismatches.csv"   # heuristic check export

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

BATCH_SIZE = 32
MAX_LENGTH = 256

# thresholds
MIN_CONF_FOR_POLAR = 0.60     # base
SHORT_LEN_THRESHOLD = 6        # tokens
SHORT_TOKEN_POSITIVE = True    # treat short praise as positive
SHORT_CONF_BONUS = 0.05        # require slightly higher confidence for short polar claims

# device
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else -1

# -----------------------------
# Arabic normalization
# -----------------------------
def normalize_ar(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r"[ـ]+", "", t)                 # remove tatweel
    t = re.sub(r"[إأآا]", "ا", t)              # normalize alef
    t = re.sub(r"ى", "ي", t)                   # alif maqsura -> ya
    t = re.sub(r"ة", "ه", t)                   # ta marbuta -> ha (choose: or keep and duplicate lexicon)
    t = re.sub(r"[ًٌٍَُِّْ~^`]", "", t)        # strip common diacritics
    t = re.sub(r"(.)\\1{2,}", r"\\1\\1", t)    # collapse elongations (رااائع -> رائع)
    return t.strip()

def tokenize_len(text: str) -> int:
    toks = re.split(r"\\s+", text.strip())
    toks = [t for t in toks if t]
    return len(toks)

# -----------------------------
# Domain lexicon and negation
# -----------------------------
POS_LEX = [
    "ممتاز","رائع","جميل","اصلي","موصى","انصح","كويس","حلو",
    "افضل","سريع التوصيل","توصيل سريع","خدمه ممتازه","خدمة ممتازة",
    "ممتازه","رائعه","اصليه","معتمد","راضي","مبسوط"
]
NEG_LEX = [
    "مزيف","غير اصلي","مو اصلي","غش","احتيال",
    "يسخن","سخونه","حراره","خربان","تعطل","تهنيج","علق",
    "مشكل","مشاكل","تالف","مكسور","كسر",
    "مرتجع","ارجاع","استرداد","رفض","يرفض",
    "مفعل مسبقا","مفعل مسبقاً","بطاريه تنفد","بطارية تنفد","سيئ","سئ","رديء","اسوأ","اسوا","كارث"
]
NEGATORS = ["ليس","مو","ما","لا","غير","بلا","بدون","مش"]

def contains_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)

def has_negated_cue(text: str, cue: str, window=4) -> bool:
    words = text.split()
    idxs = [i for i,w in enumerate(words) if cue in w]
    for idx in idxs:
        start = max(0, idx - window)
        if any(n in words[start:idx] for n in NEGATORS):
            return True
    return False

def polarity_cues(text: str):
    pos_hits = [p for p in POS_LEX if p in text and not has_negated_cue(text, p)]
    neg_hits = [n for n in NEG_LEX if n in text and not has_negated_cue(text, n)]
    return pos_hits, neg_hits

# -----------------------------
# Load data
# -----------------------------
print("📂 Loading cleaned Arabic reviews...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8")
if "clean_description" not in df.columns:
    raise ValueError("Missing 'clean_description' column in input CSV")

df["clean_description_norm"] = df["clean_description"].astype(str).fillna("").map(normalize_ar)
texts = df["clean_description_norm"]

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
# Inference + post-rules with logs
# -----------------------------
print("🧠 Performing sentiment analysis on Arabic reviews...")
pred_labels = []
pred_scores = []
rule_logs = []  # to audit decisions

with torch.no_grad():
    total = len(texts)
    for i in tqdm(range(0, total, BATCH_SIZE), total=math.ceil(total / BATCH_SIZE)):
        batch = texts.iloc[i:i+BATCH_SIZE].tolist()

        if all(len(t) == 0 for t in batch):
            pred_labels.extend(["neutral"] * len(batch))
            pred_scores.extend([1.0] * len(batch))
            rule_logs.extend(["empty->neutral"] * len(batch))
            continue

        labels, scores = predict_batched(batch)

        for idx_in_batch, (raw, lbl, sc) in enumerate(zip(batch, labels, scores)):
            applied = []
            L = tokenize_len(raw)
            base_lbl, base_sc = lbl, sc

            # Length-aware threshold
            min_conf = MIN_CONF_FOR_POLAR + (SHORT_CONF_BONUS if L <= SHORT_LEN_THRESHOLD else 0.0)

            pos_hits, neg_hits = polarity_cues(raw)

            # Short-text policy
            if L <= SHORT_LEN_THRESHOLD:
                if SHORT_TOKEN_POSITIVE:
                    if lbl == "neutral" and pos_hits:
                        lbl = "positive"; sc = max(sc, 0.75); applied.append("short_pos_override")
                else:
                    if not neg_hits:
                        lbl = "neutral"; sc = max(sc, 0.9); applied.append("short_force_neutral")

            # Confidence fallback to neutral
            if lbl in ("positive","negative") and sc < min_conf:
                lbl = "neutral"; applied.append("low_conf_neutral")

            # Mixed cues resolver
            if pos_hits and neg_hits:
                # Prefer negative unless positive is clearly higher in model confidence
                if base_lbl != "positive" or base_sc < (sc + 0.15):
                    lbl = "negative"; sc = max(sc, 0.85); applied.append("mixed_pref_negative")
                else:
                    lbl = "positive"; applied.append("mixed_positive_margin")

            # High-precision overrides
            if neg_hits and not pos_hits:
                lbl = "negative"; sc = max(sc, 0.9); applied.append("neg_lex_override")
            elif pos_hits and not neg_hits and lbl != "negative":
                lbl = "positive"; sc = max(sc, 0.85); applied.append("pos_lex_override")

            pred_labels.append(lbl)
            pred_scores.append(sc)
            rule_logs.append("|".join(applied) if applied else "none")

# -----------------------------
# Save results
# -----------------------------
df["Sentiment"] = pred_labels
df["Sentiment_confidence"] = pred_scores
df["rule_decisions"] = rule_logs

meta = {
    "model_name": MODEL_NAME,
    "transformers_version": getattr(__import__("transformers"), "__version__", "unknown"),
    "torch_version": torch.__version__,
    "device": "cuda" if USE_GPU else "cpu",
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "min_conf_for_polar": MIN_CONF_FOR_POLAR,
    "short_token_positive": SHORT_TOKEN_POSITIVE,
    "short_len_threshold": SHORT_LEN_THRESHOLD,
    "short_conf_bonus": SHORT_CONF_BONUS
}
df.attrs["meta"] = meta

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Sentiment analysis complete! Saved as '{OUTPUT_CSV}'")
print(json.dumps(meta, ensure_ascii=False, indent=2))
print(df[["clean_description", "Sentiment", "Sentiment_confidence", "rule_decisions"]].head(10).to_string(index=False))

# -----------------------------
# Optional: quick heuristic audit export
# -----------------------------
def heuristic_label(desc: str) -> str:
    if not isinstance(desc, str) or not desc.strip():
        return "neutral"
    d = desc
    pos = any(p in d for p in POS_LEX)
    neg = any(n in d for n in NEG_LEX)
    if pos and not neg: return "positive"
    if neg and not pos: return "negative"
    return "neutral"

audit = df.copy()
audit["heuristic"] = audit["clean_description_norm"].map(heuristic_label)
audit["match"] = (audit["heuristic"].str.lower() == audit["Sentiment"].str.lower())
mismatches = audit[~audit["match"]].copy()
mismatches[[
    "Clean_Product_Name","Brand","Customer_Name","Review_Date",
    "clean_title","clean_description","Sentiment","Sentiment_confidence",
    "rule_decisions","heuristic"
]].to_csv(MISMATCH_CSV, index=False)
print(f"🧪 Heuristic audit saved to '{MISMATCH_CSV}'")
