"""
Flask API for Credit Score Prediction MLOps Service
"""

import os
import time
import uuid
import logging
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

from app.model.predict import get_predictor, Predictor
from app.model.train import run_training_pipeline
from app.utils.logger import setup_logging, get_prediction_logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
# Use local logs directory for development, /var/log/mlops for production
default_log_file = os.path.join(os.getcwd(), 'logs', 'app.log')
log_file = os.environ.get('LOG_FILE', default_log_file)
json_logging = os.environ.get('JSON_LOGGING', 'true').lower() == 'true'
logger = setup_logging(
    app_name="mlops-credit-score",
    log_level=os.environ.get('LOG_LEVEL', 'INFO'),
    log_file=log_file,
    json_format=json_logging
)
prediction_logger = get_prediction_logger()

# Model directory
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')

# Initialize predictor
predictor: Predictor = None

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total number of predictions',
    ['status', 'prediction_class']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_INFO = Gauge(
    'model_info',
    'Model information',
    ['model_name', 'version']
)

REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)


def init_predictor():
    """Initialize the predictor."""
    global predictor
    try:
        predictor = get_predictor(MODEL_DIR)
        if predictor.is_ready():
            model_info = predictor.get_model_info()
            MODEL_INFO.labels(
                model_name=model_info['model_name'],
                version=model_info['version']
            ).set(1)
            logger.info(f"Predictor initialized with model: {model_info['model_name']}")
        else:
            logger.warning("Predictor initialized but no model loaded")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        predictor = None


def request_logger(f):
    """Decorator to log requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = str(uuid.uuid4())
        request.request_id = request_id
        
        start_time = time.time()
        
        logger.info(f"Request started: {request.method} {request.path} (ID: {request_id})")
        
        response = f(*args, **kwargs)
        
        latency = time.time() - start_time
        status_code = response[1] if isinstance(response, tuple) else 200
        
        REQUEST_COUNTER.labels(
            method=request.method,
            endpoint=request.path,
            status_code=status_code
        ).inc()
        
        logger.info(f"Request completed: {request.method} {request.path} - {status_code} ({latency:.3f}s)")
        
        return response
    
    return decorated_function


@app.route('/health', methods=['GET'])
@request_logger
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_loaded': predictor.is_ready() if predictor else False,
        'version': '1.0.0'
    }
    
    if predictor and predictor.is_ready():
        model_info = predictor.get_model_info()
        status['model'] = {
            'name': model_info['model_name'],
            'version': model_info['version']
        }
    
    return jsonify(status), 200


@app.route('/ready', methods=['GET'])
@request_logger
def readiness_check():
    """Readiness check for Kubernetes."""
    if predictor and predictor.is_ready():
        return jsonify({'status': 'ready'}), 200
    return jsonify({'status': 'not_ready', 'reason': 'model_not_loaded'}), 503


@app.route('/predict', methods=['POST'])
@request_logger
def predict():
    """Make prediction endpoint."""
    if not predictor or not predictor.is_ready():
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train a model first using /model/retrain endpoint'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        start_time = time.time()
        
        # Make prediction
        if isinstance(data, list):
            result = predictor.predict(data)
        else:
            result = predictor.predict_single(data)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_LATENCY.observe(latency_ms / 1000)
        
        if 'prediction' in result:
            PREDICTION_COUNTER.labels(status='success', prediction_class=result['prediction']).inc()
            prediction_logger.log_prediction(
                request_id=request_id,
                input_data=data,
                prediction=result['prediction'],
                probability=result.get('probabilities'),
                latency_ms=latency_ms,
                model_version=predictor.metadata.get('version')
            )
        else:
            for pred in result.get('predictions', []):
                PREDICTION_COUNTER.labels(status='success', prediction_class=pred).inc()
        
        result['request_id'] = request_id
        result['latency_ms'] = round(latency_ms, 2)
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        PREDICTION_COUNTER.labels(status='error', prediction_class='none').inc()
        
        prediction_logger.log_error(
            error_type='prediction_error',
            error_message=str(e),
            request_id=getattr(request, 'request_id', None),
            context={'data_type': str(type(data))}
        )
        
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/model/info', methods=['GET'])
@request_logger
def model_info():
    """Get model information."""
    if not predictor or not predictor.is_ready():
        return jsonify({
            'status': 'no_model',
            'message': 'No model is currently loaded'
        }), 200
    
    info = predictor.get_model_info()
    return jsonify(info), 200


@app.route('/model/retrain', methods=['POST'])
@request_logger
def retrain_model():
    """Trigger model retraining."""
    try:
        data = request.get_json() or {}
        
        train_data_path = data.get('train_data_path', 'data/train.csv')
        cv = data.get('cv', 5)
        
        logger.info(f"Starting model retraining with data: {train_data_path}")
        
        start_time = time.time()
        
        result = run_training_pipeline(
            train_data_path=train_data_path,
            model_dir=MODEL_DIR,
            cv=cv
        )
        
        duration = time.time() - start_time
        
        # Reload predictor with new model
        predictor.reload_model()
        
        # Update Prometheus metrics
        if predictor.is_ready():
            model_info = predictor.get_model_info()
            MODEL_INFO.labels(
                model_name=model_info['model_name'],
                version=model_info['version']
            ).set(1)
        
        prediction_logger.log_training(
            model_name=result['best_model'],
            metrics=result['metrics'],
            duration_seconds=duration,
            data_size=len(result.get('feature_columns', []))
        )
        
        return jsonify({
            'status': 'success',
            'best_model': result['best_model'],
            'metrics': result['metrics'],
            'model_path': result['model_path'],
            'duration_seconds': round(duration, 2)
        }), 200
    
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        prediction_logger.log_error(
            error_type='training_error',
            error_message=str(e)
        )
        
        return jsonify({
            'error': 'Retraining failed',
            'message': str(e)
        }), 500


@app.route('/model/reload', methods=['POST'])
@request_logger
def reload_model():
    """Reload the model from disk (for blue-green deployments)."""
    try:
        predictor.reload_model()
        
        if predictor.is_ready():
            model_info = predictor.get_model_info()
            return jsonify({
                'status': 'success',
                'model': model_info
            }), 200
        else:
            return jsonify({
                'status': 'warning',
                'message': 'Model reloaded but no valid model found'
            }), 200
    
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({
            'error': 'Reload failed',
            'message': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        prometheus_client.generate_latest(),
        mimetype='text/plain'
    )


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'service': 'MLOps Credit Score Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/ready': 'Readiness check',
            '/predict': 'Make prediction (POST)',
            '/model/info': 'Get model information',
            '/model/retrain': 'Trigger retraining (POST)',
            '/model/reload': 'Reload model (POST)',
            '/metrics': 'Prometheus metrics'
        }
    }), 200


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


# Initialize predictor on startup
with app.app_context():
    init_predictor()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting MLOps API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
