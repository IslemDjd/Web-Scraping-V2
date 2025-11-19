import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load dataset with new sentiment labels
df = pd.read_csv("final_reviews_with_sentiment.csv")

df["text"] = df["clean_title"].astype(str) + " " + df["clean_description"].astype(str)

# Map sentiment strings to integers
label2id = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].str.lower().map(label2id)

# Drop missing labels (if any)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Train-test split with stratify
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


MODEL = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id=label2id,
)

train_data = ReviewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
test_data = ReviewsDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

model.save_pretrained("arabic_sentiment_model")
tokenizer.save_pretrained("arabic_sentiment_model")

print("Model saved to arabic_sentiment_model/")
