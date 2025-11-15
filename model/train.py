import os
import re
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# Settings
# ----------------------------
SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 2
OUTPUT_DIR = "./bert_sentiment_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SUBSAMPLE = True
SUBSAMPLE_SIZE = 50000

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)
print("Device:", DEVICE)

# ----------------------------
# Load CSV
# ----------------------------
def load_sentiment140(csv_path):
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', header=None,
                     names=['target','ids','date','flag','user','text'])
    df = df[['target','text']].copy()
    df['target'] = df['target'].replace(4, 1)
    df.dropna(subset=['text','target'], inplace=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df

CSV_PATH = "training.1600000.processed.noemoticon.csv"
df = load_sentiment140(CSV_PATH)

if USE_SUBSAMPLE:
    df = df.sample(n=SUBSAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

# ----------------------------
# Clean text
# ----------------------------
def clean_text(s):
    s = str(s)
    s = re.sub(r"http\S+|www\.\S+", "", s)
    s = re.sub(r"@\w+", "", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"[^0-9A-Za-z\s#']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df['text'] = df['text'].map(clean_text)

# ----------------------------
# Train / Val / Test split
# ----------------------------
train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df['target'])
train_df, val_df = train_test_split(train_df, test_size=0.1111, random_state=SEED, stratify=train_df['target'])

train_ds = Dataset.from_pandas(train_df[['text','target']])
val_ds   = Dataset.from_pandas(val_df[['text','target']])
dataset = DatasetDict({"train": train_ds, "validation": val_ds})

# ----------------------------
# Tokenizer + preprocess
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

tokenized = dataset.map(preprocess_function, batched=True, remove_columns=['text'])
tokenized = tokenized.rename_column("target", "labels")
tokenized.set_format("torch")

# ----------------------------
# Model
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ----------------------------
# Training
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    num_train_epochs=EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=3
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
