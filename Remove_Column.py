import os
import pandas as pd

os.chdir(os.path.dirname(__file__))  # Always run in script's folder

df = pd.read_csv("merged_clean_reviews.csv")


# Remove the Brand column
df = df.drop(columns=["Brand"], errors="ignore")

# Save new file
df.to_csv("sentiment_reviews_no_brand.csv", index=False, encoding="utf-8-sig")

print("Brand column removed and new file saved!")
