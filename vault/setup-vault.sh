#!/bin/bash
# Vault Setup Script for MLOps Application
# Run this script after Vault is initialized

set -e

# Configuration
VAULT_ADDR=${VAULT_ADDR:-"http://localhost:8200"}
VAULT_TOKEN=${VAULT_TOKEN:-"mlops-dev-token"}

export VAULT_ADDR
export VAULT_TOKEN

echo "🔐 Setting up Vault for MLOps Application..."
echo "Vault Address: $VAULT_ADDR"

# Check if Vault is accessible
echo "📡 Checking Vault connectivity..."
vault status || { echo "❌ Cannot connect to Vault"; exit 1; }

# Enable secrets engine (KV v2)
echo "🔧 Enabling KV secrets engine..."
vault secrets enable -path=secret kv-v2 2>/dev/null || echo "KV engine already enabled"

# Create policies
echo "📜 Creating policies..."

vault policy write mlops-readonly - <<EOF
path "secret/data/mlops/*" {
  capabilities = ["read"]
}
path "secret/metadata/mlops/*" {
  capabilities = ["list"]
}
EOF

vault policy write mlops-cicd - <<EOF
path "secret/data/mlops/*" {
  capabilities = ["create", "read", "update", "delete"]
}
path "secret/metadata/mlops/*" {
  capabilities = ["list", "read"]
}
path "secret/data/docker/*" {
  capabilities = ["read"]
}
EOF

# Store application secrets
echo "🔒 Storing application secrets..."

# Application configuration
vault kv put secret/mlops/config \
  app_name="mlops-credit-score" \
  log_level="INFO" \
  json_logging="true" \
  environment="production"

# Docker Hub credentials (replace with actual values)
vault kv put secret/mlops/docker \
  username="${DOCKER_USERNAME:-yourusername}" \
  password="${DOCKER_PASSWORD:-changeme}"

# Database credentials (replace with actual values)
vault kv put secret/mlops/database \
  host="${DB_HOST:-localhost}" \
  port="${DB_PORT:-5432}" \
  username="${DB_USERNAME:-mlops}" \
  password="${DB_PASSWORD:-changeme}" \
  database="${DB_NAME:-mlops_db}"

# API keys
vault kv put secret/mlops/api-keys \
  prediction_api_key="${API_KEY:-$(openssl rand -hex 32)}"

# Create AppRole for application authentication
echo "🔑 Setting up AppRole authentication..."
vault auth enable approle 2>/dev/null || echo "AppRole already enabled"

vault write auth/approle/role/mlops-app \
  secret_id_ttl=24h \
  token_ttl=1h \
  token_max_ttl=4h \
  policies="mlops-readonly"

vault write auth/approle/role/mlops-cicd \
  secret_id_ttl=0 \
  token_ttl=1h \
  token_max_ttl=4h \
  policies="mlops-cicd"

# Get role IDs and secret IDs
echo "📋 Generating AppRole credentials..."

MLOPS_APP_ROLE_ID=$(vault read -field=role_id auth/approle/role/mlops-app/role-id)
MLOPS_APP_SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/mlops-app/secret-id)

MLOPS_CICD_ROLE_ID=$(vault read -field=role_id auth/approle/role/mlops-cicd/role-id)
MLOPS_CICD_SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/mlops-cicd/secret-id)

echo ""
echo "✅ Vault setup complete!"
echo ""
echo "==================================="
echo "AppRole Credentials (save securely)"
echo "==================================="
echo ""
echo "MLOps Application:"
echo "  Role ID: $MLOPS_APP_ROLE_ID"
echo "  Secret ID: $MLOPS_APP_SECRET_ID"
echo ""
echo "MLOps CI/CD:"
echo "  Role ID: $MLOPS_CICD_ROLE_ID"
echo "  Secret ID: $MLOPS_CICD_SECRET_ID"
echo ""
echo "==================================="
echo ""
echo "To authenticate, use:"
echo "  vault write auth/approle/login role_id=<role_id> secret_id=<secret_id>"
