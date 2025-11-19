"""
Final Arabic E-commerce Reviews Preprocessing (XLSX only)
Author: Assistant (adapted for Islem Dj)

Output columns (exact):
product_name
review_title
review_description
customer_name
review_date
product_image_url
"""

import argparse
import re
from datetime import datetime
import pandas as pd

# -------------------------------
# Regex
# -------------------------------
ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

# -------------------------------
# Arabic Month Mapping
# -------------------------------
AR_MONTHS = {
    "يناير": "01", "فبراير": "02", "مارس": "03",
    "أبريل": "04", "ابريل": "04", "مايو": "05",
    "يونيو": "06", "يوليو": "07", "أغسطس": "08",
    "اغسطس": "08", "سبتمبر": "09", "أكتوبر": "10",
    "اكتوبر": "10", "نوفمبر": "11", "ديسمبر": "12"
}

# -------------------------------
# Unwanted Phrases
# -------------------------------
PROBLEMATIC_SUBSTRINGS = [
    "عرض النسخة الأصلية مترجم من جوجل",
    "عرض النسخة الأصلية مُترجم من جوجل",
    "عرض النسخة الأصليةمُترجم من جوجل",
    "عرض النسخة الاصلية مترجم من جوجل"
    "مفيدًا",
    "مفيد"
]

# -------------------------------
# Helpers
# -------------------------------
def remove_diacritics(text):
    if not isinstance(text, str):
        return ""
    return DIACRITICS_RE.sub("", text)


def is_mostly_arabic(text, thresh=0.6):
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False

    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False

    arabic = [c for c in letters if ARABIC_LETTER_RE.match(c)]
    return (len(arabic) / len(letters)) >= thresh


def parse_date_to_ddmmyyyy(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().replace(",", " ")

    # Replace Arabic month names
    for ar, num in AR_MONTHS.items():
        if ar in s:
            s = s.replace(ar, num)

    # Try multiple formats
    fmts = [
        "%d/%m/%Y", "%d-%m-%Y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%d %m %Y", "%m/%d/%Y",
        "%m-%d-%Y"
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%d/%m/%Y")
        except:
            pass

    # Pandas fallback
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if not pd.isna(dt):
            return dt.strftime("%d/%m/%Y")
    except:
        pass

    return ""


def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    t = remove_diacritics(text)
    t = re.sub(r"[إأآ]", "ا", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"[^\u0600-\u06FF0-9\s\-_,\.؟!]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def preprocess(input_path, output_path):

    print("📂 Reading XLSX...")
    df = pd.read_excel(input_path, dtype=str)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    required = [
        "Product_Name",
        "Review_Title",
        "Review_Description",
        "Customer_Name",
        "Review_Date",
        "Product_Image_URL"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"❌ ERROR: Missing required column: {col}")

    # Clean strings
    df["Review_Title"] = df["Review_Title"].fillna("").astype(str).str.strip()
    df["Review_Description"] = df["Review_Description"].fillna("").astype(str).str.strip()
    df["Review_Date"] = df["Review_Date"].fillna("").astype(str).str.strip()

    orig_count = len(df)

    print("🗑 Removing empty titles...")
    df = df[df["Review_Title"] != ""]

    print("🗑 Removing problematic description patterns...")
    desc_clean = df["Review_Description"].apply(remove_diacritics)
    unwanted_mask = df["Review_Description"].eq("")

    for sub in PROBLEMATIC_SUBSTRINGS:
        unwanted_mask |= desc_clean.str.contains(remove_diacritics(sub), na=False)

    df = df[~unwanted_mask]

    print("🌍 Filtering non-Arabic reviews...")
    title_ar = df["Review_Title"].apply(lambda x: is_mostly_arabic(remove_diacritics(x)))
    desc_ar = df["Review_Description"].apply(lambda x: is_mostly_arabic(remove_diacritics(x)))
    df = df[title_ar & desc_ar]

    print("📅 Standardizing dates...")
    df["Review_Date"] = df["Review_Date"].apply(parse_date_to_ddmmyyyy)

    print("🧹 Normalizing text...")
    df["Review_Title"] = df["Review_Title"].apply(normalize_arabic)
    df["Review_Description"] = df["Review_Description"].apply(normalize_arabic)

    # -------------------------
    # Output EXACT same format
    # -------------------------
    cols_out = [
        "Product_Name",
        "Review_Title",
        "Review_Description",
        "Customer_Name",
        "Review_Date",
        "Product_Image_URL"
    ]

    df_out = df[cols_out]

    print("💾 Saving XLSX...")
    df_out.to_excel(output_path, index=False)

    print(f"✅ Done! Rows: {orig_count} → {len(df_out)}")
    print(f"📁 Output saved to: {output_path}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="Clean_Reviews.xlsx")
    args = parser.parse_args()

    preprocess(args.input, args.output)
