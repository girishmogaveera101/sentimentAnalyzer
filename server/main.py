from fastapi import FastAPI
import requests
import os

app = FastAPI()

API_URL = "https://router.huggingface.co/hf-inference/models/girizhhh/sentimentAnalyzer"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

@app.get("/predict")
def predict(text: str):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()
    except:
        return {"error": "HF returned non-JSON", "raw": response.text}
