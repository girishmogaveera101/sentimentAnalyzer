from fastapi import FastAPI
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/girizhhh/sentimentAnalyzer"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

@app.get("/predict")
def predict(text: str):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
