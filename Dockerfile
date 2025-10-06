# Dockerfile (robust for Render)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so we can use Docker layer cache
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

EXPOSE 8000

# Optional healthcheck — useful in Render logs
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS --retry 2 http://127.0.0.1:${PORT:-8000}/health || exit 1

# Use python -m uvicorn (module invocation is more robust than relying on available PATH script).
CMD ["sh", "-c", "exec python -m uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
