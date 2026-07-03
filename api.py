import os
import uuid
import logging
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load variables from .env into the environment BEFORE anything below
# tries to read them with os.getenv().
load_dotenv()

import config
from youtube_loader import load_youtube_chunks
from retriever import get_hybrid_retriever
from main import grounded_chain

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("inquira_api")

# --------------------------------------------------------------------------
# Required environment variables — fail fast instead of silently falling
# back to insecure defaults. Load these from a .env file (python-dotenv)
# or your hosting platform's secret manager, never commit them to git.
# --------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET")
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI", getattr(config, "MONGODB_ATLAS_URI", None))
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", getattr(config, "MONGODB_DB_NAME", "youtube_rag_db"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")  # comma-separated list, e.g. "https://app.example.com,https://staging.example.com"

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable must be set. Refusing to start with an insecure default.")

if not MONGODB_ATLAS_URI:
    raise RuntimeError("MONGODB_ATLAS_URI environment variable must be set.")

if not ALLOWED_ORIGINS:
    logger.warning("ALLOWED_ORIGINS is not set — CORS will block all cross-origin requests until configured.")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# --------------------------------------------------------------------------
# App + middleware setup
# --------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Inquira AI RAG Engine")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # restricted to known frontend domains, not "*"
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# --------------------------------------------------------------------------
# MongoDB connection
# --------------------------------------------------------------------------
client = MongoClient(MONGODB_ATLAS_URI, serverSelectionTimeoutMS=5000)
db = client[MONGODB_DB_NAME]
users_collection = db["users"]
history_collection = db["user_history"]

# Enforce uniqueness at the database level too (not just an app-level check).
# This closes the race condition where two simultaneous /auth/register
# requests for the same email could both pass the find_one() check.
users_collection.create_index("email", unique=True)
history_collection.create_index([("user_id", 1), ("videoId", 1)], unique=True)


# --------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------
class RegisterReq(BaseModel):
    name: str
    email: EmailStr          # validates proper email format automatically
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class LoginReq(BaseModel):
    email: EmailStr
    password: str


class ForgotReq(BaseModel):
    email: EmailStr
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class RefreshReq(BaseModel):
    refresh_token: str


class LoadReq(BaseModel):
    video_id: str


class AskReq(BaseModel):
    question: str
    video_id: str             # now required — ties the question to a specific loaded video
    session_id: str = "default_session"


class SaveHistoryReq(BaseModel):
    videoId: str
    title: str
    date: str
    chat: list[dict[str, Any]]


# --------------------------------------------------------------------------
# Per-video retriever cache (replaces unsafe global main.chunks/main.retriever)
#
# NOTE: this in-memory dict works for a single server process. If you scale
# to multiple worker processes or servers, replace this with a shared cache
# like Redis so every instance sees the same loaded videos.
# --------------------------------------------------------------------------
retriever_cache: dict[str, object] = {}


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "exp": datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ", 1)[1]

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = users_collection.find_one({"_id": payload["sub"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# --------------------------------------------------------------------------
# Auth endpoints
# --------------------------------------------------------------------------
@app.post("/auth/register")
@limiter.limit("5/minute")
def register(req: RegisterReq, request: Request):
    email = req.email.lower()
    user_id = str(uuid.uuid4())
    new_user = {
        "_id": user_id,
        "name": req.name,
        "email": email,
        "password": hash_password(req.password),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        users_collection.insert_one(new_user)
    except DuplicateKeyError:
        # Relies on the unique index rather than a separate find_one() check,
        # so it's safe even under concurrent requests.
        raise HTTPException(status_code=400, detail="Email already registered")

    access_token = create_access_token(user_id, email)
    refresh_token = create_refresh_token(user_id)

    logger.info(f"New user registered: {email}")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {"id": user_id, "name": req.name, "email": email},
    }


@app.post("/auth/login")
@limiter.limit("5/minute")
def login(req: LoginReq, request: Request):
    email = req.email.lower()
    user = users_collection.find_one({"email": email})

    # Same generic error whether the email doesn't exist or the password is
    # wrong — prevents attackers from discovering which emails are registered.
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(user["_id"], user["email"])
    refresh_token = create_refresh_token(user["_id"])

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {"id": user["_id"], "name": user["name"], "email": user["email"]},
    }


@app.post("/auth/refresh")
@limiter.limit("10/minute")
def refresh_access_token(req: RefreshReq, request: Request):
    try:
        payload = jwt.decode(req.refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token has expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = users_collection.find_one({"_id": payload["sub"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    new_access_token = create_access_token(user["_id"], user["email"])
    return {"access_token": new_access_token}


@app.post("/auth/forgot")
@limiter.limit("3/minute")
def forgot_password(req: ForgotReq, request: Request):
    # NOTE: This still only checks email ownership by knowledge of the email
    # address, which is weak. For real production use, replace this with:
    #   1. Generate a short-lived, single-use reset token
    #   2. Email it to the user as a link (e.g. via SendGrid/SES)
    #   3. A separate endpoint that accepts the reset token + new password
    # Keeping the simpler flow here since it mirrors your original design,
    # but flagging it clearly as the next thing to upgrade.
    email = req.email.lower()
    user = users_collection.find_one({"email": email})
    if not user:
        # Generic response to avoid confirming which emails are registered.
        return {"message": "If that email is registered, the password has been reset."}

    users_collection.update_one(
        {"email": email},
        {"$set": {"password": hash_password(req.new_password)}},
    )
    logger.info(f"Password reset for user: {email}")
    return {"message": "If that email is registered, the password has been reset."}


@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return {"id": user["_id"], "name": user["name"], "email": user["email"]}


# --------------------------------------------------------------------------
# Video RAG endpoints — now require authentication and use a per-video
# retriever cache instead of mutating shared global state on `main`.
# --------------------------------------------------------------------------
@app.post("/load")
@limiter.limit("10/minute")
def load_video(req: LoadReq, request: Request, user=Depends(get_current_user)):
    try:
        chunks = load_youtube_chunks(req.video_id)
        retriever_cache[req.video_id] = get_hybrid_retriever(chunks, req.video_id)
    except Exception as e:
        logger.error(f"Failed to load video {req.video_id} for user {user['email']}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Could not load this video. Please check the video ID and that it has captions available.",
        )

    logger.info(f"Video {req.video_id} indexed by {user['email']}")
    return {"title": f"Video Index Ready ({req.video_id})", "status": "success"}


@app.post("/ask")
@limiter.limit("30/minute")
def ask_question(req: AskReq, request: Request, user=Depends(get_current_user)):
    retriever = retriever_cache.get(req.video_id)
    if not retriever:
        raise HTTPException(
            status_code=400,
            detail="This video hasn't been loaded yet. Call /load with the video_id first.",
        )

    # Scope the session key to this user + video, rather than trusting the
    # client-supplied session_id directly. Without this, a user could pass
    # another user's session_id and read/write their chat history.
    scoped_session_id = f"{user['_id']}:{req.video_id}:{req.session_id}"

    try:
        answer = grounded_chain(req.question, retriever=retriever, session_id=scoped_session_id)
    except Exception as e:
        logger.error(f"grounded_chain failed for user {user['email']}, video {req.video_id}: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong while generating the answer. Please try again.")

    return {"answer": answer}


# --------------------------------------------------------------------------
# User Chat History endpoints (stored per user in MongoDB)
# --------------------------------------------------------------------------
@app.get("/history")
@limiter.limit("30/minute")
def get_user_history(request: Request, user=Depends(get_current_user)):
    docs = history_collection.find({"user_id": user["_id"]}).sort("updated_at", -1)
    history = []
    for doc in docs:
        history.append({
            "videoId": doc.get("videoId", ""),
            "title": doc.get("title", doc.get("videoId", "")),
            "date": doc.get("date", ""),
            "chat": doc.get("chat", []),
        })
    return history


@app.post("/history")
@limiter.limit("30/minute")
def save_user_history(req: SaveHistoryReq, request: Request, user=Depends(get_current_user)):
    history_collection.update_one(
        {"user_id": user["_id"], "videoId": req.videoId},
        {
            "$set": {
                "user_id": user["_id"],
                "videoId": req.videoId,
                "title": req.title,
                "date": req.date,
                "chat": req.chat,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
    return {"status": "success"}


@app.delete("/history/{video_id}")
@limiter.limit("20/minute")
def delete_user_history(video_id: str, request: Request, user=Depends(get_current_user)):
    history_collection.delete_one({"user_id": user["_id"], "videoId": video_id})
    return {"status": "success"}


# --------------------------------------------------------------------------
# Health check — useful for uptime monitoring / load balancer probes
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok", "database": "connected"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


if __name__ == "__main__":
    import uvicorn
    # In production, run with a process manager instead, e.g.:
    #   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(app, host="0.0.0.0", port=8000)