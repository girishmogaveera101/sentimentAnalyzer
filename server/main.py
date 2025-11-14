from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

model_id = "girizhhh/sentimentAnalyzer"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs).item()
    print("Logits : ",float(probs[0][0]),float(probs[0][1]))
    # label = "positive" if pred == 1 else "negative"
    # confidence = float(probs[0][pred])
    return {
        "negative": float(probs[0][0]),
        "positive": float(probs[0][1])
    }
