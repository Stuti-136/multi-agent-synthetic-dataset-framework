
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.8-flash"

# Agent temperature settings
GENERATOR_TEMPERATURE = 0.9
EXTRACTOR_TEMPERATURE = 0.0
CRITIC_TEMPERATURE = 0.0

# Pipeline settings
MAX_REVISIONS = 2
MAX_GENERATIONS = 3
SIMILARITY_THRESHOLD = 0.90

# Project directories
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = "data/exports"

# Validate API key
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )
