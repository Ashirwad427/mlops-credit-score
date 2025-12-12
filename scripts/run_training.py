#!/usr/bin/env python3
"""
Training script entry point for MLOps Credit Score Prediction project.
This script trains all model combinations (RF/XGBoost x 5 sampling methods).
"""

import os
import sys
import argparse
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model.train import run_training_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train ML models for Credit Score Prediction'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing train.csv'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--no-select-best',
        action='store_true',
        help='Do not select and train best model, only evaluate all'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='credit-score-training',
        help='MLflow experiment name for tracking'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    models_dir = os.path.join(project_root, args.models_dir)
    
    train_path = os.path.join(data_dir, 'train.csv')
    
    # Validate paths
    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        sys.exit(1)
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MLOps Credit Score Prediction - Model Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Training data: {train_path}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Cross-validation folds: {args.cv}")
    logger.info(f"Select best model: {not args.no_select_best}")
    logger.info(f"MLflow experiment: {args.experiment_name}")
    logger.info("=" * 70)
    
    try:
        # Run training pipeline
        result = run_training_pipeline(
            train_data_path=train_path,
            model_dir=models_dir,
            cv=args.cv,
            select_best=not args.no_select_best
        )
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("Training Summary")
        logger.info("=" * 70)
        
        all_results = result.get('all_results', {})
        successful = {k: v for k, v in all_results.items() if v['status'] == 'success'}
        failed = {k: v for k, v in all_results.items() if v['status'] == 'failed'}
        
        logger.info(f"Total models trained: {len(all_results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if successful:
            logger.info("\n" + "-" * 70)
            logger.info(f"{'Model':<30} {'F1 Macro':<12} {'Accuracy':<12} {'Precision':<12}")
            logger.info("-" * 70)
            for model_name, res in sorted(successful.items(), 
                                         key=lambda x: x[1]['metrics']['f1_macro'], 
                                         reverse=True):
                metrics = res['metrics']
                logger.info(
                    f"{model_name:<30} "
                    f"{metrics['f1_macro']:<12.4f} "
                    f"{metrics['accuracy']:<12.4f} "
                    f"{metrics['precision_macro']:<12.4f}"
                )
        
        if failed:
            logger.warning("\nFailed models:")
            for model_name, res in failed.items():
                logger.warning(f"  {model_name}: {res.get('error', 'Unknown error')}")
        
        if 'best_model' in result:
            logger.info("\n" + "=" * 70)
            logger.info(f"Best Model: {result['best_model']}")
            logger.info(f"F1 Macro: {result['metrics']['f1_macro']:.4f}")
            logger.info(f"Accuracy: {result['metrics']['accuracy']:.4f}")
            logger.info(f"Model saved: {result['model_path']}")
            logger.info("=" * 70)
        
        logger.info("\nTraining completed successfully!")
        return 0 if not failed else 1
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
