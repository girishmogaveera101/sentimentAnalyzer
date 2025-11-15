import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./bert_sentiment_results"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Simple preprocessing
import re
def clean_text(s):
    s = str(s)
    s = re.sub(r"http\S+|www\.\S+", "", s)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"[^0-9A-Za-z\s#']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Predict function
def predict_text(text):
    inputs = tokenizer(clean_text(text), return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).cpu().item()
    return "positive" if pred == 1 else "negative"

# Evaluation on dataset
def evaluate_on_dataset(df):
    texts = df['text'].tolist()
    labels = df['target'].tolist()
    preds = []
    for t in texts:
        inputs = tokenizer(clean_text(t), return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.append(int(torch.argmax(logits, dim=-1).cpu().item()))
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    import pandas as pd
    CSV_PATH = "training.1600000.processed.noemoticon.csv"
    df = pd.read_csv(CSV_PATH, encoding='ISO-8859-1', header=None,
                     names=['target','ids','date','flag','user','text'])
    df = df[['target','text']].copy()
    df['target'] = df['target'].replace(4,1)
    df.dropna(subset=['text','target'], inplace=True)
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)  # smaller sample for quick eval
    evaluate_on_dataset(df)
