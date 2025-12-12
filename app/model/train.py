"""
Model Training Module for Credit Score Prediction
Implements RandomForest and XGBoost with various sampling techniques.
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

# MLflow integration - logs to mlruns directory
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Run: pip install mlflow")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

from .preprocessing import preprocess, align_columns

logger = logging.getLogger(__name__)

# Model configurations
RANDOM_STATE = 42

# Classifiers to train
CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'),
}

# Sampling techniques
SAMPLERS = {
    "Baseline": None,
    "Over": RandomOverSampler(random_state=RANDOM_STATE),
    "SMOTE": SMOTE(random_state=RANDOM_STATE),
    "ADASYN": ADASYN(random_state=RANDOM_STATE),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=RANDOM_STATE),
}


def create_pipeline(classifier, sampler=None) -> Any:
    """Create a pipeline with optional sampler."""
    if sampler is None:
        return classifier
    
    return ImbPipeline([
        ('sampler', sampler),
        ('classifier', classifier)
    ])


def get_all_models() -> Dict[str, Any]:
    """Get all model combinations (classifiers x samplers)."""
    models = {}
    
    for clf_name, clf in CLASSIFIERS.items():
        for sampler_name, sampler in SAMPLERS.items():
            model_name = f"{sampler_name}_{clf_name}"
            # Clone the classifier for each combination
            if clf_name == "RandomForest":
                classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
            else:
                classifier = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss')
            
            # Clone sampler if needed
            if sampler is not None:
                if sampler_name == "Over":
                    sampler_clone = RandomOverSampler(random_state=RANDOM_STATE)
                elif sampler_name == "SMOTE":
                    sampler_clone = SMOTE(random_state=RANDOM_STATE)
                elif sampler_name == "ADASYN":
                    sampler_clone = ADASYN(random_state=RANDOM_STATE)
                elif sampler_name == "BorderlineSMOTE":
                    sampler_clone = BorderlineSMOTE(random_state=RANDOM_STATE)
                else:
                    sampler_clone = None
            else:
                sampler_clone = None
            
            models[model_name] = create_pipeline(classifier, sampler_clone)
    
    return models


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """Evaluate model using cross-validation."""
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    scores = cross_validate(
        model, X, y,
        scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'],
        cv=stratified_kfold,
        n_jobs=-1,
        return_train_score=True
    )
    
    metrics = {
        'f1_macro': float(np.mean(scores['test_f1_macro'])),
        'accuracy': float(np.mean(scores['test_accuracy'])),
        'precision_macro': float(np.mean(scores['test_precision_macro'])),
        'recall_macro': float(np.mean(scores['test_recall_macro'])),
        'train_f1_macro': float(np.mean(scores['train_f1_macro'])),
    }
    
    return metrics


def train_all_models(
    X: pd.DataFrame, 
    y: pd.Series, 
    cv: int = 5
) -> Dict[str, Dict]:
    """Train and evaluate all model combinations."""
    models = get_all_models()
    results = {}
    
    logger.info(f"Training {len(models)} model combinations...")
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        try:
            metrics = evaluate_model(model, X, y, cv)
            results[model_name] = {
                'metrics': metrics,
                'status': 'success'
            }
            logger.info(f"{model_name} - F1 Macro: {metrics['f1_macro']:.4f}")
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {
                'metrics': {},
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def select_best_model(results: Dict[str, Dict]) -> str:
    """Select the best model based on F1 macro score."""
    best_model = None
    best_score = -1
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            score = result['metrics']['f1_macro']
            if score > best_score:
                best_score = score
                best_model = model_name
    
    logger.info(f"Best model: {best_model} with F1 Macro: {best_score:.4f}")
    return best_model


def save_model(
    model: Any, 
    model_name: str, 
    metrics: Dict, 
    feature_columns: list,
    model_dir: str = "models"
) -> str:
    """Save model and metadata, keeping only the latest version."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Clean up old model files before saving new one
    for file in os.listdir(model_dir):
        if file.endswith('.joblib') or file.endswith('_metadata.json'):
            old_file_path = os.path.join(model_dir, file)
            try:
                os.remove(old_file_path)
                logger.info(f"Removed old model file: {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {str(e)}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v_{timestamp}"
    
    # Use only filenames (relative paths)
    model_filename = f"{model_name}_{version}.joblib"
    metadata_filename = f"{model_name}_{version}_metadata.json"
    
    # Save model
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'version': version,
        'timestamp': timestamp,
        'metrics': metrics,
        'feature_columns': feature_columns,
        'model_path': model_filename  # Store only filename
    }
    
    metadata_path = os.path.join(model_dir, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest model symlink/reference with filenames only
    latest_path = os.path.join(model_dir, "latest_model.json")
    with open(latest_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'version': version,
            'model_path': model_filename,  # Store only filename
            'metadata_path': metadata_filename  # Store only filename
        }, f, indent=2)
    
    logger.info(f"Model saved: {model_path}")
    return model_path


def load_model(model_dir: str = "models") -> Tuple[Any, Dict]:
    """Load the latest model."""
    latest_path = os.path.join(model_dir, "latest_model.json")
    
    if not os.path.exists(latest_path):
        raise FileNotFoundError("No model found. Please train a model first.")
    
    with open(latest_path, 'r') as f:
        latest_info = json.load(f)
    
    # Join model_dir with the filename to get the full path
    model_path = os.path.join(model_dir, latest_info['model_path'])
    metadata_path = os.path.join(model_dir, latest_info['metadata_path'])
    
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded model: {latest_info['model_name']} ({latest_info['version']})")
    return model, metadata


def run_training_pipeline(
    train_data_path: str,
    model_dir: str = "models",
    cv: int = 5,
    select_best: bool = True,
    mlflow_tracking_uri: str = None
) -> Dict:
    """Run the complete training pipeline with optional MLflow tracking."""
    
    # Setup MLflow tracking
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            # Use mlruns relative to current working directory
            # This works both locally and in Docker when cwd is /app
            mlflow.set_tracking_uri("mlruns")
        
        mlflow.set_experiment("credit-score-training")
        mlflow_run = mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logger.info(f"MLflow tracking enabled. Run ID: {mlflow_run.info.run_id}")
    
    logger.info("Starting training pipeline...")
    
    # Load and preprocess data
    logger.info(f"Loading data from {train_data_path}")
    df = pd.read_csv(train_data_path)
    df = preprocess(df, is_training=True)
    
    # Split features and target
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']
    
    logger.info(f"Training data shape: {X.shape}")
    
    # Log dataset info to MLflow
    if MLFLOW_AVAILABLE and mlflow_run:
        mlflow.log_params({
            "cv_folds": cv,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "data_path": train_data_path
        })
    
    # Train all models with cross-validation
    results = train_all_models(X, y, cv)
    
    if select_best:
        # Select best model based on CV scores
        best_model_name = select_best_model(results)
        
        # Train best model on full dataset once for deployment
        models = get_all_models()
        best_model = models[best_model_name]
        
        logger.info(f"Training {best_model_name} on full dataset for deployment")
        best_model.fit(X, y)
        
        # Use CV metrics (the real performance metrics)
        cv_metrics = results[best_model_name]['metrics']
        
        # Log best model to MLflow
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metrics({
                "best_f1_macro": cv_metrics['f1_macro'],
                "best_accuracy": cv_metrics['accuracy'],
                "best_precision": cv_metrics['precision_macro'],
                "best_recall": cv_metrics['recall_macro']
            })
            # Log all model scores for comparison
            for name, res in results.items():
                if res['status'] == 'success':
                    mlflow.log_metric(f"{name}_f1", res['metrics']['f1_macro'])
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.end_run()
            logger.info("MLflow run completed and logged.")
        
        # Save the trained model with CV metrics
        model_path = save_model(
            best_model, 
            best_model_name, 
            cv_metrics,
            X.columns.tolist(),
            model_dir
        )
        
        return {
            'all_results': results,
            'best_model': best_model_name,
            'metrics': cv_metrics,
            'model_path': model_path,
            'feature_columns': X.columns.tolist()
        }
    
    # End MLflow run if not selecting best model
    if MLFLOW_AVAILABLE and mlflow_run:
        mlflow.end_run()
    
    return {'all_results': results}


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training
    result = run_training_pipeline(
        train_data_path="data/train.csv",
        model_dir="models",
        cv=5
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Model: {result['best_model']}")
    print(f"F1 Macro: {result['metrics']['f1_macro']:.4f}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Model saved to: {result['model_path']}")
