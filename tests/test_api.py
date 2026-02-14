import pytest
import json
from datetime import datetime, timedelta
from app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestPredictionAPI:
    """Test prediction endpoints"""
    
    def test_predict_endpoint(self, client):
        """Test flight delay prediction"""
        payload = {
            'airline': 'AA',
            'origin': 'ATL',
            'dest': 'LAX',
            'distance': 2100,
            'flight_date': datetime.now().strftime('%Y-%m-%d'),
            'crs_dep_time': 1400
        }
        
        response = client.post('/api/predict', 
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'classification' in data
        assert 'regression' in data
        assert 'flight_info' in data
    
    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input"""
        payload = {
            'airline': 'INVALID',
            'origin': 'XXX',
            'dest': 'YYY'
        }
        
        response = client.post('/api/predict',
                              data=json.dumps(payload),
                              content_type='application/json')
        
        assert response.status_code in [200, 400]

class TestAnalyticsAPI:
    """Test analytics endpoints"""
    
    def test_overall_stats(self, client):
        """Test overall statistics endpoint"""
        response = client.get('/api/analytics/overall')
        assert response.status_code == 200
        data = response.get_json()
        assert 'total_flights' in data
        assert 'delay_percentage' in data
    
    def test_airline_stats(self, client):
        """Test airline statistics endpoint"""
        response = client.get('/api/analytics/by-airline')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
    
    def test_route_stats(self, client):
        """Test route statistics endpoint"""
        response = client.get('/api/analytics/by-route?limit=10')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
    
    def test_hour_stats(self, client):
        """Test hourly statistics endpoint"""
        response = client.get('/api/analytics/by-hour')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
    
    def test_day_stats(self, client):
        """Test day of week statistics endpoint"""
        response = client.get('/api/analytics/by-day')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
    
    def test_month_stats(self, client):
        """Test monthly statistics endpoint"""
        response = client.get('/api/analytics/by-month')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

class TestClusteringAPI:
    """Test clustering endpoints"""
    
    def test_airport_clusters(self, client):
        """Test airport clustering endpoint"""
        response = client.get('/api/clustering/airport-clusters')
        assert response.status_code == 200
        data = response.get_json()
        assert 'clusters' in data or 'error' in data

class TestModelAPI:
    """Test model endpoints"""
    
    def test_feature_importance(self, client):
        """Test feature importance endpoint"""
        response = client.get('/api/feature-importance')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list) or isinstance(data, dict)
    
    def test_model_metrics(self, client):
        """Test model metrics endpoint"""
        response = client.get('/api/model-metrics')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, dict)

class TestDataAPI:
    """Test data endpoints"""
    
    def test_airlines_list(self, client):
        """Test airlines list endpoint"""
        response = client.get('/api/airlines')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
    
    def test_airports_list(self, client):
        """Test airports list endpoint"""
        response = client.get('/api/airports')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
