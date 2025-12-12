# MLOps Application Policy
# This policy grants access to secrets needed by the MLOps application

# Allow reading application configuration
path "secret/data/mlops/config" {
  capabilities = ["read"]
}

# Allow reading Docker Hub credentials
path "secret/data/mlops/docker" {
  capabilities = ["read"]
}

# Allow reading database credentials
path "secret/data/mlops/database" {
  capabilities = ["read"]
}

# Allow reading API keys
path "secret/data/mlops/api-keys" {
  capabilities = ["read"]
}

# Allow listing secrets metadata
path "secret/metadata/mlops/*" {
  capabilities = ["list"]
}
