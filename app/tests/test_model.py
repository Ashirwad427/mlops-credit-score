"""
Unit Tests for ML Model
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.model.preprocessing import (
    clean_dirty, drop_cols, ohe_loan_types, 
    convert_credit_history_to_months, label_encode_credit_mix,
    preprocess, cap_outliers_iqr
)
from app.model.train import (
    get_all_models, create_pipeline, evaluate_model,
    CLASSIFIERS, SAMPLERS
)


class TestPreprocessing:
    """Test cases for preprocessing functions."""
    
    def test_drop_cols(self):
        """Test column dropping."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Month': ['Jan', 'Feb', 'Mar'],
            'Name': ['A', 'B', 'C'],
            'Number': [1, 2, 3],
            'Age': [25, 30, 35]
        })
        
        result = drop_cols(df)
        
        assert 'ID' not in result.columns
        assert 'Month' not in result.columns
        assert 'Name' not in result.columns
        assert 'Number' not in result.columns
        assert 'Age' in result.columns
    
    def test_convert_credit_history_to_months(self):
        """Test credit history conversion."""
        df = pd.DataFrame({
            'Credit_History_Age': [
                '2 Years and 6 Months',
                '5 Years and 0 Months',
                None
            ]
        })
        
        result = convert_credit_history_to_months(df)
        
        assert result['Credit_History_Age'].iloc[0] == 30  # 2*12 + 6
        assert result['Credit_History_Age'].iloc[1] == 60  # 5*12 + 0
        assert pd.isnull(result['Credit_History_Age'].iloc[2])
    
    def test_label_encode_credit_mix(self):
        """Test credit mix label encoding."""
        df = pd.DataFrame({
            'Credit_Mix': ['Bad', 'Standard', 'Good', 'Bad']
        })
        
        result = label_encode_credit_mix(df)
        
        assert result['Credit_Mix'].iloc[0] == 0
        assert result['Credit_Mix'].iloc[1] == 1
        assert result['Credit_Mix'].iloc[2] == 2
    
    def test_cap_outliers_iqr(self):
        """Test outlier capping."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100, -50]  # 100 and -50 are outliers
        })
        
        result = cap_outliers_iqr(df, ['values'])
        
        # Outliers should be capped
        assert result['values'].max() < 100
        assert result['values'].min() > -50


class TestModels:
    """Test cases for model functions."""
    
    def test_get_all_models(self):
        """Test model generation."""
        models = get_all_models()
        
        # Should have 2 classifiers * 5 samplers = 10 models
        assert len(models) == 10
        
        # Check model names
        expected_names = [
            'Baseline_RandomForest', 'Baseline_XGBoost',
            'Over_RandomForest', 'Over_XGBoost',
            'SMOTE_RandomForest', 'SMOTE_XGBoost',
            'ADASYN_RandomForest', 'ADASYN_XGBoost',
            'BorderlineSMOTE_RandomForest', 'BorderlineSMOTE_XGBoost'
        ]
        
        for name in expected_names:
            assert name in models
    
    def test_create_pipeline_no_sampler(self):
        """Test pipeline creation without sampler."""
        from sklearn.ensemble import RandomForestClassifier
        
        clf = RandomForestClassifier(random_state=42)
        pipeline = create_pipeline(clf, None)
        
        # Should return the classifier directly
        assert isinstance(pipeline, RandomForestClassifier)
    
    def test_create_pipeline_with_sampler(self):
        """Test pipeline creation with sampler."""
        from sklearn.ensemble import RandomForestClassifier
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        
        clf = RandomForestClassifier(random_state=42)
        sampler = SMOTE(random_state=42)
        pipeline = create_pipeline(clf, sampler)
        
        # Should return an imblearn Pipeline
        assert isinstance(pipeline, ImbPipeline)
    
    def test_classifiers_defined(self):
        """Test that classifiers are properly defined."""
        assert 'RandomForest' in CLASSIFIERS
        assert 'XGBoost' in CLASSIFIERS
    
    def test_samplers_defined(self):
        """Test that samplers are properly defined."""
        assert 'Baseline' in SAMPLERS
        assert 'Over' in SAMPLERS
        assert 'SMOTE' in SAMPLERS
        assert 'ADASYN' in SAMPLERS
        assert 'BorderlineSMOTE' in SAMPLERS


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'Age': np.random.randint(18, 70, n_samples),
            'Income_Annual': np.random.uniform(20000, 200000, n_samples),
            'Base_Salary_PerMonth': np.random.uniform(1000, 15000, n_samples),
            'Total_Bank_Accounts': np.random.randint(1, 10, n_samples),
            'Total_Credit_Cards': np.random.randint(0, 10, n_samples),
            'Rate_Of_Interest': np.random.uniform(5, 25, n_samples),
            'Delay_from_due_date': np.random.randint(0, 30, n_samples),
            'Credit_Mix': np.random.choice(['Bad', 'Standard', 'Good'], n_samples),
            'Credit_Score': np.random.choice(['Poor', 'Standard', 'Good'], n_samples)
        })
    
    def test_label_encode_preserves_shape(self, sample_data):
        """Test that encoding preserves data shape."""
        original_len = len(sample_data)
        
        result = label_encode_credit_mix(sample_data.copy())
        
        assert len(result) == original_len


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
