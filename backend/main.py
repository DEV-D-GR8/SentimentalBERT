from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for input text
class TextRequest(BaseModel):
    text: str

# Response model for sentiment analysis
class SentimentResponse(BaseModel):
    label: str
    numeric_label: int
    confidence: float

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    """
    Analyze sentiment by sending a GET request to the Hugging Face Space API.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # URL of the Hugging Face Space
    API_URL = "https://devchopra-sentimentanalysis.hf.space/"

    try:
        logger.info(f"Sending request to Hugging Face API with text: {request.text[:50]}...")

        # Send GET request to the Hugging Face Space with the text as a query parameter
        response = requests.get(
            API_URL,
            params={"text": request.text},
            headers={
                "Accept": "application/json"
            },
            timeout=10  # Timeout in seconds
        )

        logger.info(f"Received response with status code: {response.status_code}")
        logger.debug(f"Response content preview: {response.text[:200]}")

        # Handle non-200 responses
        if response.status_code != 200:
            try:
                error_content = response.json()
                error_detail = error_content.get("detail", response.text)
            except json.JSONDecodeError:
                error_detail = response.text

            logger.error(f"Hugging Face API error: {error_detail}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Hugging Face API returned an error: {error_detail}"
            )

        # Parse the JSON response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from sentiment analysis service: {str(e)}"
            )

        logger.info(f"Full response: {json.dumps(result, indent=2)}")

        # Map the response to the SentimentResponse model
        return SentimentResponse(
            label=result.get("label"),
            numeric_label=result.get("numeric_label"),
            confidence=result.get("confidence")
        )

    except requests.exceptions.Timeout:
        logger.error("Request to Hugging Face API timed out")
        raise HTTPException(
            status_code=504,
            detail="Request to sentiment analysis service timed out"
        )

    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Hugging Face API")
        raise HTTPException(
            status_code=503,
            detail="Could not connect to sentiment analysis service"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}
