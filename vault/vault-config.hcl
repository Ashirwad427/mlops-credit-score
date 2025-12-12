# HashiCorp Vault Configuration
# Secure credential storage for MLOps application

# Storage backend (file-based for simplicity, use Consul/etcd for production)
storage "file" {
  path = "/vault/data"
}

# Listener configuration
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1  # Enable TLS in production
  
  # TLS configuration (uncomment for production)
  # tls_cert_file = "/vault/certs/server.crt"
  # tls_key_file  = "/vault/certs/server.key"
}

# API address
api_addr = "http://0.0.0.0:8200"
cluster_addr = "https://0.0.0.0:8201"

# Enable UI
ui = true

# Disable mlock (enable in production with proper permissions)
disable_mlock = true

# Telemetry (for monitoring)
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname          = true
}

# Seal configuration (for auto-unseal, uncomment one of these)
# AWS KMS
# seal "awskms" {
#   region     = "us-east-1"
#   kms_key_id = "alias/vault-key"
# }

# Azure Key Vault
# seal "azurekeyvault" {
#   tenant_id     = "your-tenant-id"
#   client_id     = "your-client-id"
#   client_secret = "your-client-secret"
#   vault_name    = "your-vault-name"
#   key_name      = "your-key-name"
# }
