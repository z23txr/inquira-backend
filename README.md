#  Inquira Backend  Intelligent YouTube Video RAG Engine & API

**Inquira** is an advanced AI-powered Retrieval-Augmented Generation (RAG) backend designed to index YouTube videos, extract transcripts, and answer user questions with surgical precision. It leverages hybrid search (Vector + Keyword) and smart reranking to provide 100% timestamp-verified answers.

---

##  Key Features

- ** Hybrid Retrieval System:** Combines **Pinecone** dense vector embeddings (semantic search) with **BM25** sparse keyword matching (exact terminology search) for maximum retrieval accuracy.
- ** FlashRank Reranking:** Re-evaluates and re-ranks retrieved document chunks to filter out noise and send only the highest-quality context to the LLM.
- ** Failover LLM Rotator:** Automatically switches between powerful **Groq AI models** (`llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, and `llama-3.1-8b-instant`) to prevent rate limit bottlenecks and ensure zero downtime.
- ** Anti-Hallucination Guard:** Implements a strict verification loop that double-checks generated responses against actual video timestamps and transcripts.
- ** Secure Authentication:** Full JWT-based authentication (Access & Refresh tokens) with password hashing via `passlib` and user management.
- ** Cloud & Local Memory:** Stores user chat sessions and search history in **MongoDB Atlas** with fallback support for offline resilience.

---

## Technology Stack

- **Framework:** Python 3.10+, FastAPI, Uvicorn
- **AI & RAG:** LangChain, Pinecone Vector Database, BM25, FlashRank, Groq API (Llama 3.3 / Mixtral)
- **Database & Storage:** MongoDB Atlas (`pymongo`), In-Memory Cache
- **Authentication:** PyJWT, Passlib (Bcrypt/Argon2)
- **Video Processing:** `youtube-transcript-api`, pytube/custom loaders

---

## 📋 Prerequisites & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/z23txr/inquira-backend.git
   cd inquira-backend
