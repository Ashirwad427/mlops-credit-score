#!/bin/bash

# =============================================================================
# MLOps Credit Score Prediction - Model Serving Script
# =============================================================================
# Starts the Flask API server for model serving.
# =============================================================================

set -e

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-4}"
ENV="${ENV:-development}"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --workers) WORKERS="$2"; shift ;;
        --env) ENV="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST      Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT      Port to bind to (default: 5000)"
            echo "  --workers NUM    Number of workers (default: 4)"
            echo "  --env ENV        Environment: development|production (default: development)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================"
echo "MLOps Credit Score Prediction - Model Server"
echo "============================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Environment: $ENV"
echo "============================================"

# Check for models
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "Warning: No trained models found in models/ directory"
    echo "Run 'python scripts/run_training.py' first to train models"
fi

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Start server based on environment
if [ "$ENV" = "production" ]; then
    echo "Starting production server with Gunicorn..."
    exec gunicorn \
        --bind "${HOST}:${PORT}" \
        --workers "$WORKERS" \
        --timeout 120 \
        --access-logfile - \
        --error-logfile - \
        --capture-output \
        "app.main:app"
else
    echo "Starting development server..."
    export FLASK_APP=app.main
    export FLASK_ENV=development
    exec python -m flask run --host="$HOST" --port="$PORT"
fi
