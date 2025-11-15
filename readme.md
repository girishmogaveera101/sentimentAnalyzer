# Sentiment Analysis Project

A BERT-based sentiment classifier trained on the **Sentiment140 dataset**. This project provides a **backend API** hosted on **Hugging Face Spaces** and a **frontend** hosted on **Vercel** for live sentiment predictions.

## Model

* **Architecture:** BERT (`bert-base-uncased`) for sequence classification
* **Training Dataset:** Sentiment140 (~1.6M tweets)
* **Labels:** Binary sentiment (`0 = negative`, `1 = positive`)
* **Performance:** Accuracy ~83%, F1 Score ~83%
* **Hugging Face Model Hub:** [`girizhhh/sentimentAnalyzer`](https://huggingface.co/girizhhh/sentimentAnalyzer)

## Server

* **Framework:** FastAPI / Flask
* **Purpose:** Serves the model for inference via an API endpoint
* **Example Endpoint:**

```
POST /predict
Content-Type: application/json
{
  "text": "I love this product!"
}
```

**Response:**

```json
{
  "prediction": "positive"
}
```

* **Deployment:** Hugging Face Spaces

## Frontend

* **Purpose:** User interface to input text and display sentiment
* **Integration:** Calls backend API (`/predict`) to get results
* **Hosting:** Vercel

## Quick Start (Locally)

1. Clone the repo:

```bash
git clone <repo-url>
cd project
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the backend server:

```bash
uvicorn server.main:app --reload
```

4. Run the frontend locally:

```bash
cd frontend
npm install
npm run dev
```

5. Test the API:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"I love this!"}'
```

## Notes

* Pretrained **model files** and **tokenizer** are included
* Training and evaluation scripts are in notebooks for reference
* Backend API is ready for Hugging Face Spaces deployment
* Frontend can be hosted separately on Vercel and connected to backend

## Diagram

```
User Input --> Frontend (Vercel) --> Backend API (HF Spaces) --> BERT Model --> Sentiment Prediction
```
