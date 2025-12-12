# MLOps CI/CD Policy
# This policy grants access for Jenkins/CI-CD pipelines

# Allow reading and writing application configuration
path "secret/data/mlops/*" {
  capabilities = ["create", "read", "update", "delete"]
}

# Allow listing secrets
path "secret/metadata/mlops/*" {
  capabilities = ["list", "read"]
}

# Allow reading Docker Hub credentials
path "secret/data/docker/*" {
  capabilities = ["read"]
}

# Allow managing Kubernetes secrets
path "secret/data/kubernetes/*" {
  capabilities = ["create", "read", "update", "delete"]
}

# Allow generating dynamic database credentials
path "database/creds/mlops-*" {
  capabilities = ["read"]
}
