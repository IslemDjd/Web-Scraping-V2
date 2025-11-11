# =============================================================
# Arabic Reviews Sentiment Analysis (V5.0: Public Ensemble)
# Robust, public models + normalization + calibration + rules
# =============================================================

import os
import math
import json
import re
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "./Results/clean_arabic_reviews.csv"
OUTPUT_CSV = "sentiment_reviews_arabic_bert_V5.csv"
MISMATCH_CSV = "sentiment_V5_mismatches.csv"   # heuristic audit export

# Public ensemble candidates (A side)
CANDIDATES_A = [
    "asafaya/bert-base-arabic-sentiment",
    "akhooli/bert-base-arabic-camelbert-msa-sentiment",  # optional; if unavailable, will skip
    "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment" # fallback duplicate if needed
]
# Fixed B side (public)
MODEL_B = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

BATCH_SIZE = 32
MAX_LENGTH = 256

# Canonical label order
CANON_LABELS = ["negative", "neutral", "positive"]

# Calibrated thresholds (tune on dev)
THRESH_NEG = 0.45
THRESH_POS = 0.55

# Confidence and short-text policy
MIN_CONF_FOR_POLAR = 0.60
SHORT_LEN_THRESHOLD = 6
SHORT_CONF_BONUS = 0.05
SHORT_TOKEN_POSITIVE = True

# Chunking for long texts
ENABLE_CHUNKING = True
CHUNK_TOKENS = 180
CHUNK_OVERLAP = 30

# Device
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu")

# Optional HF token (for gated/public rate limits)
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

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
    t = re.sub(r"ة", "ه", t)                   # ta marbuta -> ha
    t = re.sub(r"[ًٌٍَُِّْ~^`]", "", t)        # strip diacritics
    t = re.sub(r"(.)\\1{2,}", r"\\1\\1", t)    # collapse elongations
    return t.strip()

def tokenize_len(text: str) -> int:
    toks = re.split(r"\\s+", text.strip())
    toks = [t for t in toks if t]
    return len(toks)

# -----------------------------
# Domain lexicon & negation
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

def polarity_cues(text: str) -> Tuple[List[str], List[str]]:
    pos_hits = [p for p in POS_LEX if p in text and not has_negated_cue(text, p)]
    neg_hits = [n for n in NEG_LEX if n in text and not has_negated_cue(text, n)]
    return pos_hits, neg_hits

# -----------------------------
# Data
# -----------------------------
print("📂 Loading cleaned Arabic reviews...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8")
if "clean_description" not in df.columns:
    raise ValueError("Missing 'clean_description' column in input CSV")

df["clean_description_norm"] = df["clean_description"].astype(str).fillna("").map(normalize_ar)
texts = df["clean_description_norm"].tolist()

# -----------------------------
# Model loading with fallback
# -----------------------------
def try_load(model_name, token=None):
    kw = {"use_auth_token": token} if token else {}
    tok = AutoTokenizer.from_pretrained(model_name, **kw)
    mod = AutoModelForSequenceClassification.from_pretrained(model_name, **kw)
    mod.eval().to(DEVICE)
    id2label = getattr(mod.config, "id2label", None)
    label2id = getattr(mod.config, "label2id", None)
    return tok, mod, id2label, label2id

def load_first_available(candidates, token=None):
    last_err = None
    for name in candidates:
        try:
            print(f"Loading: {name}")
            return name, *try_load(name, token)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            last_err = e
    raise RuntimeError(f"No candidate model could be loaded. Last error: {last_err}")

print("🤖 Loading ensemble models (public + fallback)...")
MODEL_A, tok_a, mod_a, id2label_a, label2id_a = load_first_available(CANDIDATES_A, HF_TOKEN)
tok_b, mod_b, id2label_b, label2id_b = try_load(MODEL_B, HF_TOKEN)

print(f"Ensemble members:\n - A: {MODEL_A}\n - B: {MODEL_B}")
print(f"Device: {'cuda' if USE_GPU else 'cpu'}")

# -----------------------------
# Inference helpers
# -----------------------------
def logits_for_texts(tokenizer, model, batch_texts):
    enc = tokenizer(
        batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**enc).logits
    return out

def chunk_text(text, tokenizer, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    enc = tokenizer(text, add_special_tokens=True, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0].tolist()
    if len(ids) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(ids):
        end = min(len(ids), start + max_tokens)
        sub_ids = ids[start:end]
        chunk = tokenizer.decode(sub_ids, skip_special_tokens=True)
        chunks.append(chunk)
        if end == len(ids): break
        start = max(0, end - overlap)
    return chunks

def map_to_canon(logits, id2label):
    if not id2label:
        return logits
    order = []
    for canon in CANON_LABELS:
        idx = None
        for k, v in id2label.items():
            if v.lower() == canon:
                idx = int(k)
                break
        if idx is None:
            return logits
        order.append(idx)
    return logits[:, order]

def ensemble_batch(batch_texts):
    logits_a_list, logits_b_list = [], []

    for t in batch_texts:
        if ENABLE_CHUNKING:
            chunks_a = chunk_text(t, tok_a, CHUNK_TOKENS, CHUNK_OVERLAP)
            chunks_b = chunk_text(t, tok_b, CHUNK_TOKENS, CHUNK_OVERLAP)
        else:
            chunks_a = [t]
            chunks_b = [t]

        la = logits_for_texts(tok_a, mod_a, chunks_a).mean(dim=0, keepdim=True)
        lb = logits_for_texts(tok_b, mod_b, chunks_b).mean(dim=0, keepdim=True)
        logits_a_list.append(la)
        logits_b_list.append(lb)

    LA = torch.cat(logits_a_list, dim=0)
    LB = torch.cat(logits_b_list, dim=0)

    LA_c = map_to_canon(LA, id2label_a)
    LB_c = map_to_canon(LB, id2label_b)

    L = (LA_c + LB_c) / 2.0
    probs = F.softmax(L, dim=-1)  # [batch, 3] in negative, neutral, positive order
    return probs

def probs_to_label(prob_vec, token_len=None):
    p_neg, p_neu, p_pos = prob_vec.tolist()

    # thresholded decision
    if p_pos >= THRESH_POS and p_pos > p_neg:
        lbl, conf = "positive", p_pos
    elif p_neg >= THRESH_NEG and p_neg > p_pos:
        lbl, conf = "negative", p_neg
    else:
        lbl, conf = "neutral", max(p_neu, max(p_pos, p_neg))

    # length-aware minimum confidence
    min_conf = MIN_CONF_FOR_POLAR + (SHORT_CONF_BONUS if (token_len is not None and token_len <= SHORT_LEN_THRESHOLD) else 0.0)
    if lbl in ("positive","negative") and conf < min_conf:
        lbl, conf = "neutral", conf
    return lbl, conf

# -----------------------------
# Inference + rules
# -----------------------------
print("🧠 Running ensemble inference with calibration and rules...")
pred_labels, pred_scores, rule_logs = [], [], []

total = len(texts)
for i in tqdm(range(0, total, BATCH_SIZE), total=math.ceil(total / BATCH_SIZE)):
    batch = texts[i:i+BATCH_SIZE]
    probs = ensemble_batch(batch)

    for t, pv in zip(batch, probs):
        L = tokenize_len(t)
        lbl, sc = probs_to_label(pv, token_len=L)
        applied = []

        pos_hits, neg_hits = polarity_cues(t)

        # Short-text policy
        if L <= SHORT_LEN_THRESHOLD:
            if SHORT_TOKEN_POSITIVE and lbl == "neutral" and pos_hits:
                lbl = "positive"; sc = max(sc, 0.75); applied.append("short_pos_override")
            elif not SHORT_TOKEN_POSITIVE and not neg_hits:
                lbl = "neutral"; sc = max(sc, 0.90); applied.append("short_force_neutral")

        # Mixed cues resolver
        if pos_hits and neg_hits:
            p_neg, p_neu, p_pos = pv.tolist()
            if p_neg + 0.10 >= p_pos:
                lbl = "negative"; sc = max(sc, max(p_neg, 0.85)); applied.append("mixed_pref_negative")
            else:
                applied.append("mixed_positive_margin")

        # High-precision overrides
        if neg_hits and not pos_hits:
            lbl = "negative"; sc = max(sc, 0.90); applied.append("neg_lex_override")
        elif pos_hits and not neg_hits and lbl != "negative":
            lbl = "positive"; sc = max(sc, 0.85); applied.append("pos_lex_override")

        pred_labels.append(lbl)
        pred_scores.append(float(sc))
        rule_logs.append("|".join(applied) if applied else "none")

# -----------------------------
# Save results
# -----------------------------
df["Sentiment"] = pred_labels
df["Sentiment_confidence"] = pred_scores
df["rule_decisions"] = rule_logs

meta = {
    "ensemble_A": MODEL_A,
    "ensemble_B": MODEL_B,
    "transformers_version": getattr(__import__("transformers"), "__version__", "unknown"),
    "torch_version": torch.__version__,
    "device": "cuda" if USE_GPU else "cpu",
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "thresholds": {"neg": THRESH_NEG, "pos": THRESH_POS},
    "min_conf_for_polar": MIN_CONF_FOR_POLAR,
    "short_len_threshold": SHORT_LEN_THRESHOLD,
    "short_conf_bonus": SHORT_CONF_BONUS,
    "short_token_positive": SHORT_TOKEN_POSITIVE,
    "enable_chunking": ENABLE_CHUNKING,
    "chunk_tokens": CHUNK_TOKENS,
    "chunk_overlap": CHUNK_OVERLAP
}
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Sentiment analysis complete! Saved as '{OUTPUT_CSV}'")
print(json.dumps(meta, ensure_ascii=False, indent=2))

# -----------------------------
# Optional: heuristic audit export
# -----------------------------
POS_HEUR, NEG_HEUR = POS_LEX, NEG_LEX
def heuristic_label(desc: str) -> str:
    if not isinstance(desc, str) or not desc.strip():
        return "neutral"
    d = desc
    pos = any(p in d for p in POS_HEUR)
    neg = any(n in d for n in NEG_HEUR)
    if pos and not neg: return "positive"
    if neg and not pos: return "negative"
    return "neutral"

audit = df.copy()
audit["heuristic"] = audit["clean_description_norm"].map(heuristic_label)
audit["match"] = (audit["heuristic"].str.lower() == audit["Sentiment"].str.lower())
mismatches = audit[~audit["match"]].copy()

mismatches[[
    "Clean_Product_Name","Brand","Customer_Name","Review_Date",
    "clean_title","clean_description","clean_description_norm",
    "Sentiment","Sentiment_confidence","rule_decisions","heuristic"
]].to_csv(MISMATCH_CSV, index=False)
print(f"🧪 Heuristic audit saved to '{MISMATCH_CSV}'")
