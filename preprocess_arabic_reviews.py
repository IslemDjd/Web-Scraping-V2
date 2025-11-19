# =============================================================
#  Arabic E-commerce Reviews Preprocessing Pipeline
#  Author: Islem Dj  |  Version: 1.2
# =============================================================

import pandas as pd
import re
from langdetect import detect, DetectorFactory
import nltk
from deep_translator import GoogleTranslator
import time

# -----------------------------------------------
# 1️⃣ Load & Basic Cleaning
# -----------------------------------------------

print("📂 Loading dataset...")

df = pd.read_csv("./Data/Phone_8K.csv", encoding="utf-8")

# Drop duplicates & empty reviews
df.drop_duplicates(inplace=True)
df.dropna(subset=["Review_Description", "Review_Title"], inplace=True)

# Clean extra whitespace in text fields
df["Review_Title"] = df["Review_Title"].astype(str).str.strip()
df["Review_Description"] = df["Review_Description"].astype(str).str.strip()

# Combine for processing (but we'll keep them separate later)
df["text"] = (df["Review_Title"].fillna('') + " " + df["Review_Description"].fillna('')).str.replace(r'\s+', ' ', regex=True).str.strip()

print(f"✅ Total reviews loaded: {len(df)}")

# -----------------------------------------------
# 2️⃣ Keep Only Arabic Reviews
# -----------------------------------------------

print("🌍 Detecting Arabic reviews...")

DetectorFactory.seed = 0

def safe_detect(text):
    try:
        return detect(text)
    except:
        return "unknown"

df["language"] = df["text"].apply(safe_detect)
df = df[df["language"] == "ar"].reset_index(drop=True)

print(f"✅ Arabic reviews kept: {len(df)}")

# -----------------------------------------------
# 3️⃣ Arabic Text Normalization
# -----------------------------------------------

print("🧹 Normalizing Arabic text...")

def normalize_arabic(text):
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)  # Remove tashkeel
    text = re.sub(r"[إأآا]", "ا", text)  # Normalize alef
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[^ء-ي\s]", " ", text)  # Keep only Arabic letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_title"] = df["Review_Title"].apply(normalize_arabic)
df["clean_description"] = df["Review_Description"].apply(normalize_arabic)

# -----------------------------------------------
# 4️⃣ Stopword Removal (Arabic)
# -----------------------------------------------

print("🧠 Removing Arabic stopwords...")

nltk.download("stopwords")

try:
    from arabicstopwords.arabicstopwords import stopwords_list
    ar_stop = set(stopwords_list())
except:
    ar_stop = set(nltk.corpus.stopwords.words("arabic"))

def remove_arabic_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in ar_stop and len(w) > 1]
    return " ".join(words)

df["clean_title"] = df["clean_title"].apply(remove_arabic_stopwords)
df["clean_description"] = df["clean_description"].apply(remove_arabic_stopwords)

# -----------------------------------------------
# 🔤 NEW STEP: Translate Product Names to English
# -----------------------------------------------

print("🌐 Translating product names to English...")

def translate_to_english(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

df["Translated_Product_Name"] = df["Product_Name"].apply(translate_to_english)


print("✅ Product name translation completed")

# -----------------------------------------------
# 5️⃣ Product Name Cleaning
# -----------------------------------------------

print("🏷️ Cleaning product names...")

def clean_product_name(name):
    if pd.isna(name):
        return ""
    name = name.strip()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(
        r"\b("
        r"gb|tb|mhz|hz|inch|5g|4g|wifi|esim|dual sim|face ?time|"
        r"middle east|gulf|saudi|uae|version|original|with|unlocked|model|edition|color|colour|"
        r"global|imported|international|refurbished|official|for|and|series"
        r")\b", "", name, flags=re.IGNORECASE
    )
    name = re.sub(r"\b\d+\s?(gb|tb)\b", "", name, flags=re.IGNORECASE)
    colors = ["black","white","silver","gold","green","blue","orange","red","purple","pink","grey","gray","cosmic"]
    color_pattern = r"\b(" + "|".join(colors) + r")\b"
    name = re.sub(color_pattern, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[-,_/]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = " ".join(word.capitalize() for word in name.split())
    return name

df["Clean_Product_Name"] = df["Translated_Product_Name"].apply(clean_product_name)

# -----------------------------------------------
# 6️⃣ Brand Extraction (Improved)
# -----------------------------------------------

brands = {
    "Apple": ["iphone", "ipad", "macbook"],
    "Samsung": ["galaxy", "note", "samsung"],
    "Sony": ["xperia", "sony"],
    "Xiaomi": ["xiaomi", "redmi", "poco"],
    "Huawei": ["huawei", "honor"],
    "Oppo": ["oppo"],
    "Vivo": ["vivo"],
    "Realme": ["realme"],
    "OnePlus": ["oneplus"],
    "Nokia": ["nokia"],
    "Asus": ["asus", "rog"],
    "Lenovo": ["lenovo"],
    "HP": ["hp"],
    "Dell": ["dell"],
    "Microsoft": ["surface", "microsoft"]
}

def extract_brand(name):
    name_low = name.lower()
    for brand, keywords in brands.items():
        if any(k in name_low for k in keywords):
            return brand
    return "Unknown"

df["Brand"] = df["Clean_Product_Name"].apply(extract_brand)

# -----------------------------------------------
# 7️⃣ Fix Date Format (Remove Comma)
# -----------------------------------------------

def clean_date(date):
    if pd.isna(date):
        return ""
    date = str(date).replace(",", "").strip()
    return date

df["Review_Date"] = df["Review_Date"].apply(clean_date)

# -----------------------------------------------
# 8️⃣ Save Clean Dataset
# -----------------------------------------------

final_df = df[[
    "Clean_Product_Name",
    "Customer_Name",
    "Review_Date",
    "clean_title",
    "clean_description"
]]

final_df.to_csv("clean_arabic_reviews.csv", index=False, encoding="utf-8-sig")

print("✅ Arabic cleaned dataset saved as: clean_arabic_reviews.csv")
print(final_df.head(5))
