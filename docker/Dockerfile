# MLOps Credit Score Prediction - Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim as production

# Labels
LABEL maintainer="MLOps Team"
LABEL version="1.0.0"
LABEL description="MLOps Credit Score Prediction API"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=5000 \
    MODEL_DIR=/app/models \
    LOG_FILE=/var/log/mlops/app.log \
    JSON_LOGGING=true

WORKDIR /app

# Create non-root user for security
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Create necessary directories
RUN mkdir -p /var/log/mlops /app/models /app/data && \
    chown -R mlops:mlops /var/log/mlops /app

# Copy Python packages from builder
COPY --from=builder --chown=mlops:mlops /root/.local /home/mlops/.local
ENV PATH=/home/mlops/.local/bin:$PATH

# Copy application code
COPY --chown=mlops:mlops app/ /app/app/
COPY --chown=mlops:mlops data/ /app/data/

# --- ADD THIS LINE ---
COPY --chown=mlops:mlops models/ /app/models/
# ---------------------

# Switch to non-root user
USER mlops

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app.main:app"]
