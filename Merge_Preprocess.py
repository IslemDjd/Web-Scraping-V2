import pandas as pd
import re

# =============================================================
# 1) LOAD BOTH FILES
# =============================================================

# First dataset (already cleaned)
df1 = pd.read_csv("./Results/clean_arabic_reviews.csv", encoding="utf-8")

# Second dataset (raw phone dataset)
# Choose the correct one depending on your file format:
df2 = pd.read_excel("./Data/Phone_8k.xlsx")       # If Excel
# df2 = pd.read_csv("Phone_8K.csv")        # If CSV


# =============================================================
# 2) BASIC CLEANING FOR SECOND FILE TO MATCH FIRST STRUCTURE
# =============================================================

# Remove duplicates & empty reviews
df2.drop_duplicates(inplace=True)
df2.dropna(subset=["Review_Description", "Review_Title"], inplace=True)

# Strip spaces
df2["Review_Title"] = df2["Review_Title"].astype(str).str.strip()
df2["Review_Description"] = df2["Review_Description"].astype(str).str.strip()

# -------------------------------------------------------------
# Arabic normalization
# -------------------------------------------------------------
def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[^ء-ي\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df2["clean_title"] = df2["Review_Title"].apply(normalize_arabic)
df2["clean_description"] = df2["Review_Description"].apply(normalize_arabic)


# -------------------------------------------------------------
# Stopwords removal (LIGHT VERSION → no NLTK needed)
# -------------------------------------------------------------
simple_stopwords = set("""
من في على عن الى إلى ما ماذا لم لن لا هذا هذه هناك هنا انا انت هو هي كان تكون نحن هم هن اذا حيث وهو وهي جدا فقط ثم او حتى كل بين مع قبل بعد حين كيف اي ايه لها له لهم بدون ضد مثل
""".split())

def remove_stops(text):
    words = text.split()
    return " ".join([w for w in words if w not in simple_stopwords and len(w) > 1])

df2["clean_title"] = df2["clean_title"].apply(remove_stops)
df2["clean_description"] = df2["clean_description"].apply(remove_stops)


# -------------------------------------------------------------
# Clean date (remove commas)
# -------------------------------------------------------------
df2["Review_Date"] = df2["Review_Date"].astype(str).str.replace(",", "").str.strip()

# -------------------------------------------------------------
# Product Name Cleanup (match first dataset)
# -------------------------------------------------------------
df2["Clean_Product_Name"] = df2["Product_Name"].astype(str).str.strip()


# =============================================================
# 3) CREATE same structure as your first CSV
# =============================================================

df2_final = df2[[
    "Clean_Product_Name",
    "Customer_Name",
    "Review_Date",
    "clean_title",
    "clean_description"
]].copy()


# =============================================================
# 4) REMOVE COMMAS FROM EVERY COLUMN TO PROTECT CSV FORMAT
# =============================================================

def remove_commas(df):
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", " ")
    return df

df1 = remove_commas(df1)
df2_final = remove_commas(df2_final)


# =============================================================
# 5) MERGE BOTH DATASETS
# =============================================================

merged = pd.concat([df1, df2_final], ignore_index=True)
merged.drop_duplicates(inplace=True)


# =============================================================
# 6) SAVE FINAL MERGED CLEAN CSV
# =============================================================

merged.to_csv("merged_clean_reviews.csv", index=False, encoding="utf-8-sig")

print("✅ Final merged file saved as: merged_clean_reviews.csv")
print("📌 Total rows:", len(merged))
