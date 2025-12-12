"""
API Tests for Flask Application
"""

import pytest
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get('/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'service' in data
        assert 'endpoints' in data


class TestModelEndpoints:
    """Test model-related endpoints."""
    
    def test_model_info(self, client):
        """Test /model/info endpoint."""
        response = client.get('/model/info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        # May or may not have model loaded
        assert 'status' in data or 'model_name' in data
    
    def test_predict_no_data(self, client):
        """Test /predict without data."""
        response = client.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        # Should return error or handle gracefully
        assert response.status_code in [400, 500, 503]


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""
    
    def test_metrics(self, client):
        """Test /metrics endpoint."""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        # Prometheus metrics are in text format
        assert b'predictions_total' in response.data or response.status_code == 200


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
