# =============================================================
#  Arabic Reviews Sentiment Analysis using Transformer
#  Author: Islem Dj | Version: 2.1
# =============================================================

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# ----------------------------------------------------------
# 1️⃣ Load cleaned Arabic reviews
# ----------------------------------------------------------
print("📂 Loading cleaned Arabic reviews...")
df = pd.read_csv("clean_arabic_reviews.csv", encoding="utf-8")

# ----------------------------------------------------------
# 2️⃣ Load Arabic sentiment model
# ----------------------------------------------------------
print("🤖 Loading Arabic sentiment transformer model...")
# Valid Arabic sentiment model
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=True)

# Create sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ----------------------------------------------------------
# 3️⃣ Apply sentiment analysis
# ----------------------------------------------------------
print("🧠 Performing sentiment analysis on Arabic reviews...")
tqdm.pandas()

def get_sentiment_arabic(text):
    if pd.isna(text) or text.strip() == "":
        return "neutral"
    try:
        result = sentiment_pipeline(text[:512])[0]  # truncate to 512 tokens
        label = result["label"].lower()
        # CAMeL-Lab model labels: 'positive', 'neutral', 'negative'
        if label in ["positive", "pos"]:
            return "positive"
        elif label in ["negative", "neg"]:
            return "negative"
        else:
            return "neutral"
    except Exception:
        return "neutral"

df["Sentiment"] = df["clean_description"].progress_apply(get_sentiment_arabic)

# ----------------------------------------------------------
# 4️⃣ Save results
# ----------------------------------------------------------
df.to_csv("sentiment_reviews_arabic_bert.csv", index=False, encoding="utf-8-sig")

print("✅ Sentiment analysis complete! Saved as 'sentiment_reviews_arabic_bert.csv'")
print(df[["clean_description", "Sentiment"]].head(10))
