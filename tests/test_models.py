import pytest
import pandas as pd
import numpy as np
from services.ml_models import FlightDelayModels

@pytest.fixture
def sample_data():
    """Create sample flight data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'AIRLINE': np.random.choice(['AA', 'DL', 'UA', 'SW'], n_samples),
        'ORIGIN_AIRPORT': np.random.choice(['ATL', 'LAX', 'ORD'], n_samples),
        'DESTINATION_AIRPORT': np.random.choice(['JFK', 'SFO', 'DEN'], n_samples),
        'DISTANCE': np.random.randint(100, 3000, n_samples),
        'SCHEDULED_DEPARTURE': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'DEPARTURE_DELAY': np.random.normal(10, 20, n_samples),
        'ARRIVAL_DELAY': np.random.normal(15, 25, n_samples),
        'IS_DELAYED': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)

class TestMLModels:
    """Test ML model training and prediction"""
    
    def test_feature_preparation(self, sample_data):
        """Test feature preparation"""
        models = FlightDelayModels()
        X, feature_cols = models.prepare_features(sample_data, fit_scaler=True)
        
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == len(feature_cols)
        assert len(feature_cols) > 0
    
    def test_classification_model_training(self, sample_data):
        """Test classification model training"""
        models = FlightDelayModels()
        metrics = models.train_classification_model(sample_data)
        
        assert 'random_forest' in metrics
        assert 'logistic_regression' in metrics
        assert 'accuracy' in metrics['random_forest']
        assert 'f1' in metrics['random_forest']
    
    def test_regression_model_training(self, sample_data):
        """Test regression model training"""
        models = FlightDelayModels()
        metrics = models.train_regression_model(sample_data)
        
        assert 'gradient_boosting' in metrics
        assert 'rmse' in metrics['gradient_boosting']
        assert 'r2' in metrics['gradient_boosting']
    
    def test_clustering_model_training(self, sample_data):
        """Test clustering model training"""
        models = FlightDelayModels()
        results = models.train_clustering_model(sample_data)
        
        assert 'clusters' in results
        assert 'n_clusters' in results
        assert results['n_clusters'] == 5
    
    def test_feature_importance(self, sample_data):
        """Test feature importance calculation"""
        models = FlightDelayModels()
        importance = models.get_feature_importance(sample_data)
        
        assert isinstance(importance, list)
        assert len(importance) > 0
        assert 'feature' in importance[0]
        assert 'importance' in importance[0]
