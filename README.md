#Inquira Backend  Intelligent YouTube Video RAG Engine & API
**Inquira** is an advanced AI-powered Retrieval-Augmented Generation (RAG) backend designed to index YouTube videos, extract transcripts, and answer user questions with surgical precision. It leverages hybrid search (Vector + Keyword) and smart reranking to provide 100% timestamp-verified answers.
---
## Key Features
- **Hybrid Retrieval System:** Combines **Pinecone** dense vector embeddings (semantic search) with **BM25** sparse keyword matching (exact terminology search) for maximum retrieval accuracy.
- **FlashRank Reranking:** Re-evaluates and re-ranks retrieved document chunks to filter out noise and send only the highest-quality context to the LLM.
- **Failover LLM Rotator:** Automatically switches between powerful **Groq AI models** (`llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, and `llama-3.1-8b-instant`) to prevent rate limit bottlenecks and ensure zero downtime.
- **Anti-Hallucination Guard:** Implements a strict verification loop that double-checks generated responses against actual video timestamps and transcripts.
- **Secure Authentication:** Full JWT-based authentication (Access & Refresh tokens) with password hashing via `passlib` and user management.
- **☁️ Cloud & Local Memory:** Stores user chat sessions and search history in **MongoDB Atlas** with fallback support for offline resilience.
---
## Technology Stack
- **Framework:** Python 3.10+, FastAPI, Uvicorn
- **AI & RAG:** LangChain, Pinecone Vector Database, BM25, FlashRank, Groq API (Llama 3.3 / Mixtral)
- **Database & Storage:** MongoDB Atlas (`pymongo`), In-Memory Cache
- **Authentication:** PyJWT, Passlib (Bcrypt/Argon2)
- **Video Processing:** `youtube-transcript-api`, pytube/custom loaders
---
##  Prerequisites & Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/z23txr/inquira-backend.git
   cd inquira-backend
   ```
2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   
   ```
3. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn langchain pinecone-client pymongo pyjwt passlib groq requests
   ```
---
## ⚙️ Environment Variables (`.env`)
Create a `.env` file in the root directory and add your API keys:
```env
# Groq AI API Key
GROQ_API_KEY="your_groq_api_key_here"
# Pinecone Vector DB
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="youtube-rag"
PINECONE_ENVIRONMENT="us-east-1-aws"
# MongoDB Atlas
MONGODB_ATLAS_URI="mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"
# Security
JWT_SECRET="your_super_secret_jwt_key_here"
```
---
## Running the Server
Start the FastAPI development server:
```bash
uvicorn api:app --reload --port 8000
```
The API documentation (Swagger UI) will be available at:
**http://localhost:8000/docs**
---
## API Endpoints Overview
### Authentication (`/auth`)
- `POST /auth/register` — Register a new user account.
- `POST /auth/login` — Log in and receive JWT access & refresh tokens.
- `POST /auth/refresh` — Generate a new access token using a refresh token.
- `POST /auth/forgot` — Reset account password.
### Video Indexing & RAG
- `POST /load` — Index a YouTube video (fetches transcript, creates vector & BM25 indexes).
  ```json
  { "video_id": "dQw4w9WgXcQ" }
  ```
- `POST /ask` — Ask a question against the indexed video.
  ```json
  {
    "question": "What is the main summary of this video?",
    "video_id": "dQw4w9WgXcQ",
    "session_id": "default_session"
  }
  ```
### History & Memory
- `GET /history` — Retrieve authenticated user's saved chat sessions.
- `POST /history` — Save a new chat session to MongoDB cloud storage.
---
## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/z23txr/inquira-backend/issues).

## 📋 Prerequisites & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/z23txr/inquira-backend.git
   cd inquira-backend
