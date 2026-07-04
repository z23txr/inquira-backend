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

_history_cache = {}
_cache_lock = threading.Lock()


def get_chat_history(session_id: str = "default_session"):
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
    recent_messages = messages[-max_messages:] if messages else []
    formatted = []
    for msg in recent_messages:
        role = "User" if msg.type == "human" else "AI"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted) if formatted else "No previous conversation."