#config.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_google_api_key():
    """Retrieves the Google API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the .env file.")
    return api_key

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8