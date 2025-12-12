# MLOps Credit Score Prediction - Complete Guide

This is the consolidated documentation for the MLOps Credit Score Prediction project. It covers setup, testing, and demonstration of all components.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Component Testing](#3-component-testing)
   - [3.1 ML Training Pipeline](#31-ml-training-pipeline)
   - [3.2 Flask API](#32-flask-api)
   - [3.3 Docker Containers](#33-docker-containers)
   - [3.4 MLflow Experiment Tracking](#34-mlflow-experiment-tracking)
   - [3.5 Prometheus Metrics](#35-prometheus-metrics)
   - [3.6 Elasticsearch (ELK)](#36-elasticsearch-elk)
   - [3.7 Unit Tests](#37-unit-tests)
4. [Full Demo Flow](#4-full-demo-flow)
5. [API Reference](#5-api-reference)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | ML training and API |
| Docker | 20.10+ | Containerization |
| Docker Compose | 2.0+ | Multi-container orchestration |
| Git | 2.30+ | Version control |

### Setup Virtual Environment

```bash
cd mlops-credit-score

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Always activate the virtual environment before running any Python commands.

### Verify Installations

```bash
# Ensure venv is activated (you should see (venv) in your prompt)
source venv/bin/activate

python3 --version        # Python 3.10+
docker --version         # Docker 20.10+
docker compose version   # Docker Compose 2.0+
```

---

## 2. Quick Start

### Option A: Run Everything with Docker (Recommended)

```bash
# Start all services (includes MLflow, Elasticsearch, Prometheus, Grafana)
docker compose -f docker/docker-compose.yml up -d

# Check status
docker ps

# Test API
curl http://localhost:5000/health

# Access MLflow UI
open http://localhost:5001
```

### Option B: Run Locally

```bash
# Activate virtual environment first!
source venv/bin/activate

# Train model first (logs to mlruns/ automatically)
python3 scripts/run_training.py

# Start MLflow UI to view training runs
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001 &

# Start Flask API
python3 -m flask --app app.main run --host 0.0.0.0 --port 5000
```

---

## 3. Component Testing
> **Important:** For all local Python commands, always activate the virtual environment first:
> ```bash
> source venv/bin/activate
> ```

### 3.1 ML Training Pipeline

**What it does:** Trains 10 model combinations (2 classifiers × 5 sampling methods) and selects the best one. Automatically logs all runs to MLflow.

#### Run Training

```bash
source venv/bin/activate
python3 scripts/run_training.py
```

> **Note:** Training automatically logs to `mlruns/` directory. All experiments, metrics, parameters, and models are tracked.

#### Expected Output

```
======================================================================
MLOps Credit Score Prediction - Model Training Pipeline
======================================================================
Training data: .../data/train.csv
Models directory: .../models
Cross-validation folds: 5
======================================================================
Training 10 model combinations...
Training Baseline_RandomForest...
Baseline_RandomForest - F1 Macro: 0.7856
...
Best Model: Over_RandomForest
F1 Macro: 0.7981
Model saved: models/Over_RandomForest_v_YYYYMMDD_HHMMSS.joblib
======================================================================
Training completed successfully!
```

#### Verify Results

```bash
# Check saved model
ls -lh models/

# View model metadata
cat models/latest_model.json
```

---

### 3.2 Flask API

**What it does:** Serves predictions via REST API with health checks, metrics, and model info.

#### Start API (if not using Docker)

```bash
python3 -m flask --app app.main run --host 0.0.0.0 --port 5000
```

#### Test Endpoints

**Health Check:**
```bash
curl http://localhost:5000/health
```
Expected: `{"status": "healthy", "model_loaded": true, ...}`

**Model Info:**
```bash
curl http://localhost:5000/model/info
```
Expected: `{"model_name": "Over_RandomForest", "metrics": {...}, ...}`

**Make Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Annual_Income": 50000,
    "Monthly_Inhand_Salary": 4000,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 2,
    "Interest_Rate": 10,
    "Num_of_Loan": 2,
    "Delay_from_due_date": 5,
    "Num_of_Delayed_Payment": 3,
    "Changed_Credit_Limit": 10,
    "Num_Credit_Inquiries": 2,
    "Credit_Mix": "Standard",
    "Outstanding_Debt": 1000,
    "Credit_Utilization_Ratio": 30,
    "Credit_History_Age": "10 Years and 5 Months",
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 500,
    "Amount_invested_monthly": 200,
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Monthly_Balance": 500
  }'
```
Expected: `{"prediction": "Standard", "probabilities": [0.21, 0.60, 0.19], ...}`

---

### 3.3 Docker Containers

**What it does:** Runs the entire stack in containers (API, MLflow, Elasticsearch, Vault, Prometheus, Grafana).

#### Start Containers

```bash
docker compose -f docker/docker-compose.yml up -d
```

#### Check Container Status

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:
```
NAMES           STATUS          PORTS
mlops-api       Up (healthy)    0.0.0.0:5000->5000/tcp
elasticsearch   Up (healthy)    0.0.0.0:9200->9200/tcp
vault           Up              0.0.0.0:8200->8200/tcp
prometheus      Up              0.0.0.0:9090->9090/tcp
grafana         Up              0.0.0.0:3000->3000/tcp
```

#### View Container Logs

```bash
docker logs mlops-api --tail 20
docker logs mlflow --tail 20
docker logs elasticsearch --tail 20
```

#### Stop Containers

```bash
docker compose -f docker/docker-compose.yml down
```

---

### 3.4 MLflow Experiment Tracking

**What it does:** Tracks all training experiments, logs metrics, parameters, and models. All training runs are automatically logged to the `mlruns/` directory.

#### View Training Runs

**Option A: Start MLflow UI Locally**

```bash
# Activate venv first
source venv/bin/activate

# Start MLflow UI pointing to local mlruns directory
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001
```

Then open: **http://localhost:5001**

**Option B: Using Docker (Alternative)**

MLflow can also run in Docker, sharing the same `mlruns/` directory:

```bash
# Start docker stack (shares mlruns volume)
docker compose -f docker/docker-compose.yml up -d mlflow

# Access MLflow UI
open http://localhost:5001
```

> **Note:** Both options read from the same `mlruns/` directory, so all training runs are visible regardless of where MLflow UI is running.

#### What You'll See

- **Experiment:** `credit-score-training` with all training runs
- **Metrics:** Best model's F1, accuracy, precision, recall + all 10 model F1 scores for comparison
- **Parameters:** CV folds, number of samples, features, data path, best model name
- **Artifacts:** Logged model (best model saved as artifact)
- **Run Name:** `training_YYYYMMDD_HHMMSS` with unique Run ID

#### Query MLflow Data (Python)

```python
import mlflow

# List experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")

# Get runs from an experiment
runs = mlflow.search_runs(experiment_ids=["1"])
print(runs[['run_id', 'metrics.best_f1_macro', 'params.cv_folds']])
```

---

### 3.5 Prometheus Metrics

**What it does:** Exposes application metrics for monitoring (predictions, latency, errors).

#### Test Metrics Endpoint

```bash
curl http://localhost:5000/metrics
```

#### Key Metrics Exposed

| Metric | Type | Description |
|--------|------|-------------|
| `predictions_total` | Counter | Total predictions by status and class |
| `prediction_latency_seconds` | Histogram | Prediction latency distribution |
| `model_info` | Gauge | Current model name and version |
| `http_requests_total` | Counter | Total HTTP requests |

#### Access Prometheus UI

If running with Docker:
```
http://localhost:9090
```

Sample query: `rate(predictions_total[5m])`

---

### 3.6 Elasticsearch (ELK)

**What it does:** Stores structured JSON logs for analysis and visualization.

#### Check Elasticsearch Health

```bash
curl http://localhost:9200/_cluster/health
```

Expected: `{"status": "yellow" or "green", ...}`

#### View Indices

```bash
curl http://localhost:9200/_cat/indices?v
```

#### Search Logs

```bash
curl -X GET "http://localhost:9200/mlops-logs/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{"query": {"match_all": {}}, "size": 5}'
```

#### Access Kibana (if running)

```
http://localhost:5601
```

---

### 3.7 Unit Tests

**What it does:** Runs automated tests for API and model components.

#### Install Test Dependencies

```bash
# Activate venv first
source venv/bin/activate

pip install pytest pytest-cov
```

#### Run All Tests

```bash
source venv/bin/activate
python3 -m pytest app/tests/ -v
```

#### Run with Coverage

```bash
source venv/bin/activate
python3 -m pytest app/tests/ --cov=app --cov-report=html
```

View coverage report: `open htmlcov/index.html`

#### Test Files

| File | Tests |
|------|-------|
| `app/tests/test_api.py` | API endpoints, health, predictions |
| `app/tests/test_model.py` | Training, preprocessing, model loading |

---

## 4. Full Demo Flow

Follow this sequence to demonstrate all components working together:

### Step 1: Setup Environment

```bash
cd mlops-credit-score

# Activate virtual environment
source venv/bin/activate

# Clean up previous runs (optional)
rm -rf models/*.joblib models/*.json mlruns/

# Keep the .gitkeep file
echo "# Placeholder" > models/.gitkeep
```

### Step 2: Train the Model

```bash
python3 scripts/run_training.py
```

**✓ Checkpoint:** 
- Verify `models/latest_model.json` exists
- See `MLflow tracking enabled. Run ID: <run_id>` message
- Check `mlruns/` directory created with experiment data

### Step 3: Start Docker Stack

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

**✓ Checkpoint:** Run `docker ps` and verify all containers are running

### Step 4: Test API

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model/info

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "Annual_Income": 50000, "Monthly_Inhand_Salary": 4000, "Num_Bank_Accounts": 3, "Num_Credit_Card": 2, "Interest_Rate": 10, "Num_of_Loan": 2, "Delay_from_due_date": 5, "Num_of_Delayed_Payment": 3, "Changed_Credit_Limit": 10, "Num_Credit_Inquiries": 2, "Credit_Mix": "Standard", "Outstanding_Debt": 1000, "Credit_Utilization_Ratio": 30, "Credit_History_Age": "10 Years and 5 Months", "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 500, "Amount_invested_monthly": 200, "Payment_Behaviour": "Low_spent_Small_value_payments", "Monthly_Balance": 500}'
```

**✓ Checkpoint:** Prediction returns `{"prediction": "Standard", ...}`

### Step 5: View Metrics

```bash
curl http://localhost:5000/metrics | grep predictions_total
```

**✓ Checkpoint:** Counter shows at least 1 prediction

### Step 6: Check Elasticsearch Logs

```bash
curl http://localhost:9200/_cluster/health
```

**✓ Checkpoint:** Status is "yellow" or "green"

### Step 7: View MLflow Training Runs

Start MLflow UI to view the logged training runs:

```bash
# In a new terminal, activate venv
source venv/bin/activate

# Start MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001

# Or use Docker (alternative)
# docker compose -f docker/docker-compose.yml up -d mlflow

# Open in browser
open http://localhost:5001
```

**✓ Checkpoint:** 
- See experiment `credit-score-training`
- View the training run with all metrics (best_f1_macro, best_accuracy, etc.)
- See all 10 model comparison metrics
- Model artifact is logged

### Step 8: Cleanup

```bash
docker compose -f docker/docker-compose.yml down
```

---

## 5. API Reference

### Base URL

```
http://localhost:5000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with model status |
| GET | `/model/info` | Current model details and metrics |
| POST | `/predict` | Make credit score prediction |
| GET | `/metrics` | Prometheus metrics |
| POST | `/train` | Trigger model retraining |
| POST | `/model/reload` | Reload model from disk |

### Prediction Request Schema

```json
{
  "Age": 35,
  "Annual_Income": 50000,
  "Monthly_Inhand_Salary": 4000,
  "Num_Bank_Accounts": 3,
  "Num_Credit_Card": 2,
  "Interest_Rate": 10,
  "Num_of_Loan": 2,
  "Delay_from_due_date": 5,
  "Num_of_Delayed_Payment": 3,
  "Changed_Credit_Limit": 10,
  "Num_Credit_Inquiries": 2,
  "Credit_Mix": "Standard",
  "Outstanding_Debt": 1000,
  "Credit_Utilization_Ratio": 30,
  "Credit_History_Age": "10 Years and 5 Months",
  "Payment_of_Min_Amount": "Yes",
  "Total_EMI_per_month": 500,
  "Amount_invested_monthly": 200,
  "Payment_Behaviour": "Low_spent_Small_value_payments",
  "Monthly_Balance": 500
}
```

### Prediction Response Schema

```json
{
  "request_id": "uuid",
  "prediction": "Standard",
  "prediction_code": 1.0,
  "probabilities": [0.21, 0.60, 0.19],
  "class_labels": ["Poor", "Standard", "Good"],
  "latency_ms": 58.06
}
```

---

## 6. Troubleshooting

### Training Crashes / High Memory Usage

**Symptom:** System freezes or runs out of memory during training

**Solution:**
1. MLflow is disabled by default to prevent this
2. If you enabled MLflow and experience issues, run without it:
   ```bash
   python3 scripts/run_training.py  # MLflow disabled by default
   ```
3. Close other applications to free up memory
4. Training 10 models requires ~4GB RAM

### MLflow Shows No Runs

**Symptom:** MLflow UI is empty, no experiments visible

**Solution:**
1. Ensure you've run training at least once:
   ```bash
   source venv/bin/activate
   python3 scripts/run_training.py
   ```
2. Check mlruns directory exists and has data:
   ```bash
   ls -la mlruns/
   ls -la mlruns/*/  # View experiment folders
   ```
3. Verify MLflow UI is pointing to correct directory:
   ```bash
   mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001
   ```
4. Clean up corrupted experiments if needed:
   ```bash
   # Remove old/corrupted experiment folders
   rm -rf mlruns/1 mlruns/2
   ```

### Model Not Loading in Docker

**Symptom:** `{"message": "No model is currently loaded"}`

**Solution:**
1. Ensure model is trained locally first: `python3 scripts/run_training.py`
2. Rebuild container: `docker compose -f docker/docker-compose.yml up -d --build mlops-api`
3. Check mount: `docker exec mlops-api ls /app/models/`

### Elasticsearch Connection Refused

**Symptom:** `connection refused` on port 9200

**Solution:**
1. Wait 30-60 seconds for Elasticsearch to start
2. Check logs: `docker logs elasticsearch`
3. Verify memory: Elasticsearch needs at least 2GB RAM

### MLflow Not Tracking

**Symptom:** No runs appear in MLflow UI

**Solution:**
1. Verify MLflow installed: `python3 -c "import mlflow; print(mlflow.__version__)"`
2. Check mlruns directory exists: `ls mlruns/`
3. Ensure training completed successfully

### Prediction Errors

**Symptom:** `500 Internal Server Error` on `/predict`

**Solution:**
1. Check all required fields are provided
2. Verify field names match exactly (case-sensitive)
3. Check logs: `docker logs mlops-api`

### Port Already in Use

**Symptom:** `Address already in use`

**Solution:**
```bash
# Find process using port
lsof -i :5000

# Kill it
kill -9 <PID>

# Or stop all Docker containers
docker compose -f docker/docker-compose.yml down
```

---

## Summary

**Training automatically logs to MLflow** - All experiments are stored in the `mlruns/` directory and can be viewed using the MLflow UI.

| Component | Port | Test Command |
|-----------|------|--------------|
| Flask API | 5000 | `curl http://localhost:5000/health` |
| MLflow UI | 5001 | `mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001` |
| Prometheus | 9090 | Open http://localhost:9090 |
| Grafana | 3000 | Open http://localhost:3000 |
| Elasticsearch | 9200 | `curl http://localhost:9200/_cluster/health` |
| Kibana | 5601 | Open http://localhost:5601 |
| Vault | 8200 | `curl http://localhost:8200/v1/sys/health` |

> **Note:** MLflow UI can run locally or in Docker - both read from the same `mlruns/` directory

---

**Project:** MLOps Credit Score Prediction  
**Version:** 1.0.0  
**Last Updated:** December 2025
