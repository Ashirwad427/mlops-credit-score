#!/bin/bash

# =============================================================================
# MLOps Credit Score Prediction - Local Development Setup Script
# =============================================================================
# This script sets up the local development environment for the project.
# It creates a virtual environment, installs dependencies, and prepares
# the project for development.
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

print_header "MLOps Credit Score Prediction - Development Setup"

echo "Project root: $PROJECT_ROOT"

# Check Python version
print_header "Checking Python Version"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_success "Found $PYTHON_VERSION"
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Create virtual environment
print_header "Setting Up Virtual Environment"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_header "Upgrading pip"
pip install --upgrade pip
print_success "pip upgraded"

# Install dependencies
print_header "Installing Dependencies"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed from requirements.txt"
else
    print_warning "requirements.txt not found, installing from setup.py"
    pip install -e .
fi

# Create necessary directories
print_header "Creating Directory Structure"
mkdir -p models logs data
print_success "Directories created"

# Check for data files
print_header "Checking Data Files"
if [ -f "data/train.csv" ] && [ -f "data/test.csv" ]; then
    print_success "Data files found"
else
    print_warning "Data files not found in data/ directory"
    echo "Please ensure train.csv and test.csv are in the data/ directory"
fi

# Set up pre-commit hooks (optional)
print_header "Setting Up Git Hooks"
if [ -d ".git" ]; then
    if command -v pre-commit &> /dev/null; then
        pre-commit install 2>/dev/null || true
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not installed, skipping hooks setup"
    fi
else
    print_warning "Not a git repository, skipping hooks setup"
fi

# Run tests to verify setup
print_header "Verifying Setup"
echo "Running quick test..."
python -c "
import sys
try:
    import pandas
    import numpy
    import sklearn
    import xgboost
    import flask
    print('All core dependencies imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"
print_success "Setup verification passed"

# Print summary
print_header "Setup Complete!"
echo "
To get started:

1. Activate the virtual environment:
   ${GREEN}source venv/bin/activate${NC}

2. Train the models:
   ${GREEN}python scripts/run_training.py${NC}

3. Start the Flask API:
   ${GREEN}python -m app.main${NC}

4. Run tests:
   ${GREEN}pytest app/tests/ -v${NC}

5. Build Docker image:
   ${GREEN}docker build -t mlops-credit-score -f docker/Dockerfile .${NC}

For full deployment, refer to docs/SETUP.md
"
