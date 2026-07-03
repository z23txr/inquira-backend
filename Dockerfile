# Use a slim Python base image matching the Python version we tested with (3.11)
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout (so logs show up immediately)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed to build some Python packages (e.g. tokenizers, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — Docker caches this layer, so rebuilds are
# fast unless requirements.txt itself changes (code changes won't re-trigger
# this slow pip install step).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install the CPU-only build of torch FIRST. Without this, sentence-transformers
# (a dependency of langchain-huggingface) pulls in the full CUDA/GPU version of
# torch by default, which downloads several extra GB of NVIDIA packages that
# are completely useless here since this container has no GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Render/Railway set PORT via env variable; default to 8000 for local Docker runs
ENV PORT=8000
EXPOSE 8000

# Simple healthcheck so the platform knows when the container is actually ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use shell form so ${PORT} gets expanded — required for platforms like
# Render/Railway that inject a dynamic PORT value
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}