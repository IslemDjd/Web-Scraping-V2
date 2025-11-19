from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

label_names = {0: "negative", 1: "neutral", 2: "positive"}

tokenizer = AutoTokenizer.from_pretrained("arabic_sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("arabic_sentiment_model")
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    return label_names[pred], probs[0, pred].item()

test_texts = [
    "الهاتف ممتاز جدا وخدمة التوصيل رائعة",          # Positive
    "الخدمة كانت بطيئة ولم أكن راضيًا عن التجربة",     # Negative
    "المنتج جيد لكنه يحتاج إلى تحسين في التغليف",      # Neutral
    "التطبيق يعاني من الكثير من الأخطاء ويجب تحديثه",   # Negative
    "المطعم هادئ والطعام لذيذ ولكن السعر مرتفع قليلاً",  # Neutral
    "التجربة كانت أكثر من رائعة، أنصح به بشدة",         # Positive
    "الجهاز توقف عن العمل بعد أسبوع فقط",              # Negative
    "التوصيل كان سريعًا، ولكن المنتج لم يكن كما توقعت",  # Neutral
    "خدمة العملاء ممتازة ويسرني التعامل معهم",          # Positive
    "لم يستحق السعر المدفوع، تجربة سيئة",               # Negative
]

with open("sentiment_predictions.txt", "w", encoding="utf-8") as f:
    for text in test_texts:
        sentiment, confidence = predict_sentiment(text)
        f.write(f"Text: {text}\nPredicted sentiment: {sentiment} (confidence: {confidence:.2f})\n\n")

print("Predictions saved to sentiment_predictions.txt")
