FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed for building some wheels (xgboost/catboost may need compilers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository files (backend.py and support files)
COPY . /app

EXPOSE 8000

# Use uvicorn as the server
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
