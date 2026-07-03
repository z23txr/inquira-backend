import os
import sys
import io
import logging
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger("inquira_rag")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI") or os.getenv("MONGO_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "youtube_rag_db")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "chat_history")

# --------------------------------------------------------------------------
# Fail fast: if a required key is missing, crash on startup with a clear
# message instead of failing confusingly deep inside a request later.
# MONGODB_ATLAS_URI is intentionally not required here — memory.py already
# has a safe in-memory fallback if it's missing.
# --------------------------------------------------------------------------
_required = {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
}
_missing = [name for name, value in _required.items() if not value]
if _missing:
    raise RuntimeError(f"Missing required environment variable(s): {', '.join(_missing)}. Check your .env file.")

if not MONGODB_ATLAS_URI:
    logger.warning("MONGODB_ATLAS_URI is not set — chat history will fall back to in-memory storage and won't persist across restarts.")

# Only used by main.py's standalone CLI test harness (`python main.py`),
# not by the API — each /load request there specifies its own video_id.
VIDEO_ID = os.getenv("TEST_VIDEO_ID", "zCNEngO4cfY")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "youtube-rag")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# NOTE: verify these model names are still active on Groq before deploying —
# Groq periodically deprecates/renames models (mixtral-8x7b-32768 in
# particular has been deprecated in the past). Check https://console.groq.com/docs/models
POWERFUL_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

FAST_MODELS = [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
]