"""
Logging Module - ELK Stack Compatible
Provides structured JSON logging for integration with ELK Stack.
"""

import os
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for ELK Stack integration."""
    
    def __init__(self, app_name: str = "mlops-credit-score"):
        super().__init__()
        self.app_name = app_name
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "app": self.app_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class PredictionLogger:
    """Logger for ML predictions - useful for monitoring and debugging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_prediction(
        self,
        request_id: str,
        input_data: Dict,
        prediction: Any,
        probability: Optional[list] = None,
        latency_ms: Optional[float] = None,
        model_version: Optional[str] = None
    ):
        """Log a prediction event."""
        extra_data = {
            "event_type": "prediction",
            "request_id": request_id,
            "input_features_count": len(input_data) if isinstance(input_data, dict) else 0,
            "prediction": prediction,
            "model_version": model_version
        }
        
        if probability is not None:
            extra_data["probability"] = probability
        
        if latency_ms is not None:
            extra_data["latency_ms"] = latency_ms
        
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            "",
            0,
            f"Prediction: {prediction}",
            (),
            None
        )
        record.extra_data = extra_data
        self.logger.handle(record)
    
    def log_training(
        self,
        model_name: str,
        metrics: Dict,
        duration_seconds: float,
        data_size: int
    ):
        """Log a training event."""
        extra_data = {
            "event_type": "training",
            "model_name": model_name,
            "metrics": metrics,
            "duration_seconds": duration_seconds,
            "data_size": data_size
        }
        
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            "",
            0,
            f"Training complete: {model_name}",
            (),
            None
        )
        record.extra_data = extra_data
        self.logger.handle(record)
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        request_id: Optional[str] = None,
        context: Optional[Dict] = None
    ):
        """Log an error event."""
        extra_data = {
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message
        }
        
        if request_id:
            extra_data["request_id"] = request_id
        
        if context:
            extra_data["context"] = context
        
        record = self.logger.makeRecord(
            self.logger.name,
            logging.ERROR,
            "",
            0,
            f"Error: {error_type} - {error_message}",
            (),
            None
        )
        record.extra_data = extra_data
        self.logger.handle(record)


def setup_logging(
    app_name: str = "mlops-credit-score",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True
) -> logging.Logger:
    """
    Setup application logging.
    
    Args:
        app_name: Application name for log identification
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: Use JSON format for ELK Stack compatibility
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter(app_name)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
        except PermissionError:
            # Fall back to local logs directory if permission denied
            local_log_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(local_log_dir, exist_ok=True)
            log_file = os.path.join(local_log_dir, 'app.log')
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.warning(f"Permission denied for configured log path. Using local logs: {log_file}")
        logger.addHandler(file_handler)
    
    return logger


def get_prediction_logger(app_name: str = "mlops-credit-score") -> PredictionLogger:
    """Get a prediction logger instance."""
    logger = logging.getLogger(f"{app_name}.predictions")
    return PredictionLogger(logger)


# Create default logger
default_logger = setup_logging()
prediction_logger = get_prediction_logger()
