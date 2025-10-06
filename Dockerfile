# Dockerfile (improved for Render)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps (kept minimal). Add/remove packages if you need them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker layer caching helps during iterative builds
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY . /app

EXPOSE 8000

# Optional healthcheck — Render will still use the PORT env variable to route traffic,
# but this makes it easier to see container health in logs.
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS --retry 2 http://127.0.0.1:${PORT:-8000}/health || exit 1

# Use PORT env variable provided by Render. Default to 8000 if not provided.
CMD ["sh", "-c", "exec uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
