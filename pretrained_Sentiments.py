# Arabic Sentiment Auto-Labeling with Hugging Face Transformers

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# 1. Load your dataset
df = pd.read_csv("./Results/final_cleaned_reviews.csv")

# 2. Build the text field (combine clean_title and clean_description)
df["text"] = df["clean_title"].astype(str) + " " + df["clean_description"].astype(str)

# 3. Load a pretrained 3-label Arabic sentiment pipeline
sentiment_pipe = pipeline(
    "text-classification",
    model="AbdallahNasir/book-review-sentiment-classification",
    device=0  # Set to -1 for CPU or 0 for first GPU (Colab etc.)
)

# 4. Run sentiment prediction in batches
texts = df["text"].tolist()
pred_labels = []
batch_size = 32

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    outputs = sentiment_pipe(batch, truncation=True)
    pred_labels.extend([o["label"] for o in outputs])

# 5. Attach the predicted sentiment to the dataframe
df["sentiment"] = pred_labels

# 6. (Optional) Save the new file with sentiment labels
df.to_csv("final_reviews_with_sentiment.csv", index=False)

print(df[["text", "sentiment"]].head())
