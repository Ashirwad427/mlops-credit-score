# MLOps DevOps Framework - Credit Score Prediction

## 🎓 CSE 816: Software Production Engineering - Final Project

A complete MLOps DevOps framework implementing automated SDLC for Machine Learning pipelines using industry-standard DevOps tools.

### 👥 Team Members
- Ashirwad Mishra : IMT2022108
- Ishan Singh : IMT2022124


## 📋 Project Overview

This project implements a complete DevOps pipeline for an ML application that predicts credit scores using RandomForest and XGBoost classifiers with various sampling techniques (SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler).

### 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   GitHub    │───▶   Jenkins    │───▶   Docker    │───▶  Kubernetes  │
│  (Git Push) │    │  (CI/CD)     │    │  (Build)    │    │  (Deploy)    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                                       │
                          ▼                                       ▼
                   ┌──────────────┐                       ┌──────────────┐
                   │   Ansible    │                       │  ELK Stack   │
                   │ (Config Mgmt)│                       │ (Monitoring) │
                   └──────────────┘                       └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │    Vault     │
                   │  (Secrets)   │
                   └──────────────┘
```

---

## 🛠️ Technology Stack

| Component                | Tool                      | Purpose                             |
|--------------------------|---------------------------|-------------------------------------|
| Version Control          | Git & GitHub              | Source code management              |
| CI/CD                    | Jenkins + GitHub Webhooks | Automated build & deployment        |
| Containerization         | Docker & Docker Compose   | Application packaging               |
| Configuration Management | Ansible                   | Infrastructure automation           |
| Orchestration            | Kubernetes (K8s)          | Container orchestration & scaling   |
| Monitoring & Logging     | ELK Stack                 | Centralized logging & visualization |
| Secret Management        | HashiCorp Vault           | Secure credential storage           |

---

## 📁 Project Structure

```
mlops-credit-score/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Flask API
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py            # Model training
│   │   ├── predict.py          # Prediction logic
│   │   └── preprocessing.py    # Data preprocessing
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py           # ELK-compatible logging
│   └── tests/
│       ├── __init__.py
│       ├── test_model.py
│       └── test_api.py
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── preprocessed/
├── models/                     # Saved model artifacts
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.train
│   └── docker-compose.yml
├── jenkins/
│   ├── Jenkinsfile
│   └── jenkins-config.yaml
├── ansible/
│   ├── playbook.yml
│   ├── inventory/
│   │   ├── hosts
│   │   └── group_vars/
│   └── roles/
│       ├── docker/
│       ├── kubernetes/
│       └── elk/
├── kubernetes/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   └── secrets.yaml
├── elk/
│   ├── elasticsearch/
│   ├── logstash/
│   └── kibana/
├── vault/
│   └── vault-config.hcl
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Kubernetes cluster (minikube for local development)
- Jenkins server
- Ansible
- HashiCorp Vault

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mlops-credit-score.git
cd mlops-credit-score
```

### 2. Local Development
```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# Access the API
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

### 3. Deploy to Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/

# Check deployment status
kubectl get pods -n mlops-credit
```

---

## 🔧 Features

### ✅ Core Requirements
- [x] Git version control with GitHub
- [x] Jenkins CI/CD with GitHub webhooks
- [x] Docker containerization
- [x] Ansible configuration management
- [x] Kubernetes orchestration
- [x] ELK Stack monitoring

### ⭐ Advanced Features (3 Marks)
- [x] **Vault Integration**: Secure storage of credentials
- [x] **Ansible Roles**: Modular playbook design
- [x] **Kubernetes HPA**: Horizontal Pod Autoscaling

### 🎯 Domain-Specific (MLOps - 5 Marks)
- [x] ML Model Training Pipeline
- [x] Model versioning
- [x] A/B Testing support
- [x] Model performance monitoring
- [x] Automated retraining triggers

### 💡 Innovation (2 Marks)
- [x] Blue-Green deployment strategy
- [x] Canary releases
- [x] ML model rollback capability
- [x] Automated data drift detection

---

## 📊 ML Models

### Classifiers
- **RandomForest Classifier**
- **XGBoost Classifier**

### Sampling Techniques
- Baseline (No sampling)
- Random Oversampling
- SMOTE
- ADASYN
- BorderlineSMOTE

### Evaluation Metrics
- F1 Score (Macro)
- Accuracy
- Precision
- Recall
- Confusion Matrix

---

## 🔄 CI/CD Pipeline

### Pipeline Stages
1. **Checkout**: Fetch code from GitHub
2. **Build**: Install dependencies
3. **Test**: Run unit and integration tests
4. **Train**: Train ML models (if data changed)
5. **Build Docker Image**: Package application
6. **Push to Registry**: Push to Docker Hub
7. **Deploy**: Deploy to Kubernetes cluster
8. **Verify**: Health checks and smoke tests

### Webhook Configuration
Jenkins is configured with GitHub webhook to trigger builds on:
- Push to main branch
- Pull request creation
- Tag creation

---

## 📈 Monitoring with ELK Stack

### Log Types
- Application logs
- Model prediction logs
- Training metrics
- Error tracking

### Kibana Dashboards
- Real-time prediction monitoring
- Model performance trends
- Error rate visualization
- Resource utilization

---

## 🔐 Security Features

### Vault Integration
- API keys storage
- Database credentials
- Docker Hub credentials
- Kubernetes secrets

### Best Practices
- No hardcoded secrets
- RBAC for Kubernetes
- Network policies
- Image scanning

---

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/model/info` | GET | Model information |
| `/model/retrain` | POST | Trigger retraining |
| `/metrics` | GET | Prometheus metrics |

---

## 🧪 Testing

```bash
# Run unit tests
pytest app/tests/ -v

# Run with coverage
pytest app/tests/ --cov=app --cov-report=html

# Integration tests
pytest app/tests/test_api.py -v
```

---

## 📝 License

This project is part of CSE 816: Software Production Engineering course.

---

## 📧 Contact

For questions or support, contact the team members.
