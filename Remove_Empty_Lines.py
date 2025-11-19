import pandas as pd
import re

df = pd.read_csv("merged_clean_reviews.csv")

# ---------------------------------------------------------
# 1) REMOVE rows with empty Review_Title
# ---------------------------------------------------------
df["clean_title"] = df["clean_title"].astype(str).str.strip()

df = df[df["clean_title"].notna()]
df = df[df["clean_title"] != ""]
df = df[df["clean_title"] != "nan"]


# ---------------------------------------------------------
# 2) CLEAN Review_Description
# ---------------------------------------------------------
df["clean_description"] = df["clean_description"].astype(str).str.strip()

# Replace NaN and empty values with "/"
df.loc[df["clean_description"].isna(), "clean_description"] = "/"
df.loc[df["clean_description"] == "", "clean_description"] = "/"
df.loc[df["clean_description"] == "nan", "clean_description"] = "/"


# ---------------------------------------------------------
# 3) Remove Google-Translate related phrases by keyword
# ---------------------------------------------------------
# We remove ANY description containing ANY of these keywords:
keywords = [
    "مترجم",           # translated
    "جوجل",            # google
    "النسخة الاصلية",  # original version
    "عرض النسخة",      # preview original
    "عرض النسخة الاصلية",
    "google"
]

def clean_translation_markers(text):
    t = text.lower()
    for kw in keywords:
        if kw in t:
            return "/"       # replace ANY matching text with /
    return text

df["clean_description"] = df["clean_description"].apply(clean_translation_markers)


# ---------------------------------------------------------
# 4) Remove commas from ALL columns (to avoid breaking CSV)
# ---------------------------------------------------------
df = df.applymap(lambda x: str(x).replace(",", " ") if isinstance(x, str) else x)


# ---------------------------------------------------------
# 5) SAVE CLEAN CSV
# ---------------------------------------------------------
df.to_csv("merged_clean_reviews_v2.csv", index=False, encoding="utf-8-sig")

print("Done! File saved as Phone_8K_clean.csv")
