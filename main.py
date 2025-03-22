import firebase_admin
from firebase_admin import auth, credentials
import requests
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Depends
from dotenv import load_dotenv
from chatbot import get_ai_response  # Ensure chatbot.py exists and has this function
from quiz_generator import quiz_router  # Import the FastAPI router correctly
from fastapi_limiter.depends import RateLimiter
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CANVAS_API_URL = "https://canvas.howard.edu/api/v1"
CANVAS_ACCESS_TOKEN = os.getenv("CANVAS_ACCESS_TOKEN")
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", "firebase-adminsdk.json")

# Initialize logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app (Only one instance of FastAPI should exist)
app = FastAPI()

# Include the Quiz Generator Router (Ensures quiz routes are registered)
app.include_router(quiz_router)

# Initialize Firebase Admin SDK for authentication (Uses service account credentials)
try:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred)
except Exception as e:
    logger.error(f"Firebase initialization error: {e}")
    raise HTTPException(status_code=500, detail="Firebase setup failed")

# Middleware: Verify Firebase Token
def verify_token(authorization: str = Header(None)):
    """
    Middleware to verify Firebase authentication token.
    Ensures only authorized users can access protected routes.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization token is missing")

    try:
        id_token = authorization.split("Bearer ")[1]  # Extract token from header
        decoded_token = auth.verify_id_token(id_token)  # Verify token using Firebase

        # Check if token is expired
        exp_time = decoded_token.get("exp")
        if exp_time and datetime.utcfromtimestamp(exp_time) < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

        return decoded_token["uid"]  # Return authenticated user ID
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# Login Route (Authenticates users via Firebase)
@app.post("/login")
async def login(authorization: str = Header(None)):
    """
    Authenticates user using Firebase ID token.
    Returns a success message and the authenticated user ID.
    """
    user_id = verify_token(authorization)
    return {"message": "Login successful", "userId": user_id}

# Fetch User Details (Protected Route)
@app.get("/user-details")
async def get_user_details(authorization: str = Header(None)):
    """
    Fetches user details for the authenticated user.
    This is a protected route requiring a valid Firebase token.
    """
    user_id = verify_token(authorization)
    return {"message": "User details fetched successfully", "userId": user_id}

# Fetch Courses (Canvas API)
@app.get("/courses")
async def get_courses():
    """
    Retrieves the list of enrolled courses for the authenticated user from Canvas API.
    No authentication is required for this request.
    """
    try:
        headers = {"Authorization": f"Bearer {CANVAS_ACCESS_TOKEN}"}
        response = requests.get(f"{CANVAS_API_URL}/courses", headers=headers)
        response.raise_for_status()
        return {"courses": response.json()}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch courses: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch courses")

# Background Task: Sync Data from Canvas API
def fetch_latest_data():
    """
    Fetches assignments and grades from Canvas API in the background.
    This ensures the latest data is available without user requests.
    """
    try:
        headers = {"Authorization": f"Bearer {CANVAS_ACCESS_TOKEN}"}
        requests.get(f"{CANVAS_API_URL}/assignments", headers=headers)
        requests.get(f"{CANVAS_API_URL}/users/self/enrollments", headers=headers)
        logger.info("Canvas API data synced successfully")
    except Exception as e:
        logger.error(f"Failed to sync Canvas data: {e}")

# API Endpoint: Trigger Background Sync
@app.get("/sync")
async def sync_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(fetch_latest_data)
    return {"message": "Background sync started"}

# Fetch Upcoming Assignments (Protected)
@app.get("/upcoming-tasks")
async def get_upcoming_tasks(authorization: str = Header(None)):
    """
    Fetches upcoming tasks for the authenticated user.
    This is a protected route requiring a valid Firebase token.
    """
    user_id = verify_token(authorization)
    return {
        "message": "Upcoming tasks fetched successfully",
        "userId": user_id,
        "tasks": [
            {"task_name": "Complete Firebase setup", "due_date": "2025-03-10"},
            {"task_name": "Build FastAPI backend", "due_date": "2025-03-12"}
        ]
    }

# Fetch Grades (Canvas API)
@app.get("/grades")
async def get_grades():
    """
    Retrieves the authenticated user's grades from the Canvas API.
    No authentication is required for this request.
    """
    try:
        headers = {"Authorization": f"Bearer {CANVAS_ACCESS_TOKEN}"}
        response = requests.get(f"{CANVAS_API_URL}/users/self/enrollments", headers=headers)
        response.raise_for_status()
        return {"grades": response.json()}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch grades: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch grades")

# Fetch Announcements (Canvas API)
@app.get("/announcements")
async def get_announcements():
    """
    Retrieves course announcements from the Canvas API.
    This provides students with important updates from their courses.
    """
    try:
        headers = {"Authorization": f"Bearer {CANVAS_ACCESS_TOKEN}"}
        response = requests.get(f"{CANVAS_API_URL}/announcements", headers=headers)
        response.raise_for_status()
        return {"announcements": response.json()}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch announcements: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch announcements")

# AI Chatbot API (Uses OpenAI API) with Rate Limiting
@app.post("/ai/chatbot", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def ai_chatbot(question: str):
    """
    AI-powered chatbot that answers academic questions.
    Uses OpenAI API to generate a response based on user input.
    Limited to 5 requests per minute per user.
    """
    return {"question": question, "answer": get_ai_response(question)}

# Centralized Exception Handler (Prevents crashes)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "An unexpected error occurred"})

# FastAPI automatically maps `/generate_flashcards` and `/generate_quiz` from quiz_router
