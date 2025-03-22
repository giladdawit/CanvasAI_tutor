from fastapi import APIRouter, HTTPException
from openai import OpenAI
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Retrieve OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Ensure API key is set before initializing OpenAI client
if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API Key not found. Please check your .env file.")

# ✅ Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Create FastAPI Router for Quiz & Flashcards
quiz_router = APIRouter()

@quiz_router.post("/generate_quiz")
async def generate_quiz(text: str):
    """
    Takes study material and returns AI-generated quiz questions (without storing data).
    """
    try:
        prompt = f"Generate 3 quiz questions (multiple-choice) from this study material:\n{text}"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI quiz generator."},
                {"role": "user", "content": prompt}
            ]
        )

        return {"quiz": response.choices[0].message.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")

@quiz_router.post("/generate_flashcards")
async def generate_flashcards(text: str):
    """
    Takes study material and returns AI-generated flashcards (without storing data).
    """
    try:
        prompt = f"Create 3 flashcards (Q&A format) from this text:\n{text}"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI flashcard generator."},
                {"role": "user", "content": prompt}
            ]
        )

        return {"flashcards": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
__all__= ["quiz_router"]