from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI(title="Sentiment Analyzer API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
print("Loading model...")
classifier = pipeline(
    "text-classification",
    model="girizhhh/sentimentAnalyzer",
    return_all_scores=True
)
print("Model loaded successfully!")

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "message": "Sentiment Analyzer API",
        "endpoints": {
            "POST /predict": "Analyze sentiment of text",
            "GET /health": "Check API health"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "sentimentAnalyzer"}

@app.post("/predict")
def predict(input: TextInput):
    if not input.text or not input.text.strip():
        return {
            "error": "Please provide text",
            "negative": 0,
            "positive": 0
        }
    
    try:
        results = classifier(input.text)[0]
        
        negative_score = results[0]['score']
        positive_score = results[1]['score']
        
        return {
            "negative": round(negative_score, 4),
            "positive": round(positive_score, 4),
            "text": input.text
        }
    except Exception as e:
        return {
            "error": str(e),
            "negative": 0,
            "positive": 0
        }

@app.get("/predict")
def predict_get(text: str):
    """Alternative GET endpoint for simple testing"""
    if not text or not text.strip():
        return {
            "error": "Please provide text parameter",
            "negative": 0,
            "positive": 0
        }
    
    try:
        results = classifier(text)[0]
        
        negative_score = results[0]['score']
        positive_score = results[1]['score']
        
        return {
            "negative": round(negative_score, 4),
            "positive": round(positive_score, 4),
            "text": text
        }
    except Exception as e:
        return {
            "error": str(e),
            "negative": 0,
            "positive": 0
        }