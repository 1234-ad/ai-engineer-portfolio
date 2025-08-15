from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import logging

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Load pre-trained BERT model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True
)

class TextInput(BaseModel):
    text: str
    language: str = "en"

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of input text
    """
    try:
        # Process text through model
        results = sentiment_pipeline(input_data.text)
        
        # Extract best prediction
        best_result = max(results[0], key=lambda x: x['score'])
        
        # Format response
        response = SentimentResponse(
            text=input_data.text,
            sentiment=best_result['label'],
            confidence=round(best_result['score'], 4),
            scores={result['label']: round(result['score'], 4) for result in results[0]}
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "twitter-roberta-base-sentiment"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)