import logging
import threading

import config
from langchain_core.chat_history import InMemoryChatMessageHistory

try:
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False

logger = logging.getLogger("inquira_rag")

# --------------------------------------------------------------------------
# Cache history objects per session_id instead of constructing a new
# MongoDBChatMessageHistory (which opens its own MongoClient connection)
# on every single call. Without this, every question asked would open a
# fresh MongoDB connection just to fetch history.
# --------------------------------------------------------------------------
_history_cache = {}
_cache_lock = threading.Lock()


def get_chat_history(session_id: str = "default_session"):
    """
    Fetches chat history for a session from MongoDB Atlas if configured,
    otherwise falls back to InMemoryChatMessageHistory.

    SECURITY NOTE: session_id is treated as an opaque cache key here. The
    caller (api.py) is responsible for making sure one user's session_id
    can't be guessed or reused to read another user's conversation — e.g.
    by scoping it as f"{user_id}:{video_id}" rather than trusting a raw
    client-supplied session_id directly.
    """
    with _cache_lock:
        if session_id in _history_cache:
            return _history_cache[session_id]

        history = None
        if getattr(config, "MONGODB_ATLAS_URI", None) and HAS_MONGODB:
            try:
                history = MongoDBChatMessageHistory(
                    connection_string=config.MONGODB_ATLAS_URI,
                    session_id=session_id,
                    database_name=getattr(config, "MONGODB_DB_NAME", "youtube_rag_db"),
                    collection_name=getattr(config, "MONGODB_COLLECTION_NAME", "chat_history"),
                )
            except Exception as e:
                logger.warning(f"MongoDB connection failed ({e}). Falling back to in-memory history.")
                history = None

        if history is None:
            history = InMemoryChatMessageHistory()

        _history_cache[session_id] = history
        return history


def format_chat_history(messages, max_messages=4):
    """
    Returns sliding window of recent conversation (Last N messages).
    Best practice for single chat memory to keep context clean and reduce token usage.
    """
    recent_messages = messages[-max_messages:] if messages else []
    formatted = []
    for msg in recent_messages:
        role = "User" if msg.type == "human" else "AI"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted) if formatted else "No previous conversation."