# ==============================================================================
# Production Dockerfile for Customer Churn MLOps Prediction Service
# Complies with container security standards (Non-root user, slim base image)
# ==============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, application, and model artifacts
COPY src/ /app/src/
COPY app/ /app/app/
COPY models/ /app/models/
COPY data/ /app/data/

# Create a non-privileged user and switch to it
RUN addgroup --system appgroup && adduser --system --group appuser \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
