"""
Prediction Module for Credit Score Prediction
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union

from .train import load_model
from .preprocessing import preprocess, align_columns

logger = logging.getLogger(__name__)

# Label mapping
LABEL_MAPPING = {0: 'Poor', 1: 'Standard', 2: 'Good'}
REVERSE_LABEL_MAPPING = {'Poor': 0, 'Standard': 1, 'Good': 2}


class Predictor:
    """Predictor class for making predictions."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the predictor with a trained model."""
        self.model = None
        self.metadata = None
        self.model_dir = model_dir
        self.feature_columns = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.model, self.metadata = load_model(self.model_dir)
            self.feature_columns = self.metadata.get('feature_columns', [])
            logger.info(f"Model loaded: {self.metadata['model_name']} ({self.metadata['version']})")
        except FileNotFoundError as e:
            logger.warning(f"No model found: {e}")
            self.model = None
            self.metadata = None
    
    def reload_model(self):
        """Reload the model (useful for hot-reloading after retraining)."""
        self._load_model()
    
    def is_ready(self) -> bool:
        """Check if the predictor is ready to make predictions."""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_ready():
            return {'status': 'no_model_loaded'}
        
        return {
            'model_name': self.metadata['model_name'],
            'version': self.metadata['version'],
            'timestamp': self.metadata['timestamp'],
            'metrics': self.metadata['metrics'],
            'feature_count': len(self.feature_columns)
        }
    
    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict:
        """
        Make predictions on input data.
        
        Args:
            data: Input data as dict, list of dicts, or DataFrame
        
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Please train a model first.")
        
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        logger.info(f"Making predictions for {len(df)} samples")
        
        # Preprocess
        df_processed = preprocess(df, is_training=False)
        
        # Align columns with training data
        df_aligned = align_columns(self.feature_columns, df_processed)
        
        # Make predictions
        predictions = self.model.predict(df_aligned)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(df_aligned)
            except Exception:
                pass
        elif hasattr(self.model, 'named_steps'):
            # For pipeline models
            try:
                probabilities = self.model.predict_proba(df_aligned)
            except Exception:
                pass
        
        # Convert numeric predictions to labels
        prediction_labels = [LABEL_MAPPING.get(p, str(p)) for p in predictions]
        
        result = {
            'predictions': prediction_labels,
            'prediction_codes': predictions.tolist(),
            'count': len(predictions)
        }
        
        if probabilities is not None:
            result['probabilities'] = probabilities.tolist()
            result['class_labels'] = ['Poor', 'Standard', 'Good']
        
        logger.info(f"Predictions complete: {result['count']} samples")
        return result
    
    def predict_single(self, data: Dict) -> Dict:
        """
        Make prediction for a single sample.
        
        Args:
            data: Single sample as dictionary
        
        Returns:
            Dictionary with prediction result
        """
        result = self.predict(data)
        
        return {
            'prediction': result['predictions'][0],
            'prediction_code': result['prediction_codes'][0],
            'probabilities': result.get('probabilities', [[]])[0] if result.get('probabilities') else None,
            'class_labels': result.get('class_labels')
        }


# Global predictor instance (lazy loaded)
_predictor = None


def get_predictor(model_dir: str = "models") -> Predictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor(model_dir)
    return _predictor


def predict(data: Union[Dict, List[Dict], pd.DataFrame], model_dir: str = "models") -> Dict:
    """Convenience function for making predictions."""
    predictor = get_predictor(model_dir)
    return predictor.predict(data)


def predict_single(data: Dict, model_dir: str = "models") -> Dict:
    """Convenience function for single prediction."""
    predictor = get_predictor(model_dir)
    return predictor.predict_single(data)
