from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi import HTTPException

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure API key is set
if API_KEY is None:
    raise ValueError("❌ OpenAI API key not found. Set it in the .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

def get_ai_response(question: str):
    """
    Sends a request to OpenAI API and returns the AI-generated response.
    """
    try:
        print(f"🚀 Sending request to OpenAI: {question}")  # Debugging: See what is being sent
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI tutor."},
                {"role": "user", "content": question}
            ]
        )

        print("✅ Response received!")  # Debugging: Confirm response was received
        print("🔍 Full Response:", response)  # Debugging: Print full response object

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ OpenAI API error: {str(e)}")  # Debugging: Print error message
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
