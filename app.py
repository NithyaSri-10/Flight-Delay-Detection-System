from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
from collections import defaultdict
from config.database import SessionLocal, engine
from models.warehouse_schema import Base
from services.olap_queries import OLAPAnalytics
from services.ml_models import FlightDelayModels
from config.auth import require_auth, auth_manager
from config.weather_service import weather_service
import uuid

from services.ml_models import train_classification_models
from config.database import SessionLocal
from scripts.etl_pipeline import load_and_clean_data
import os
import json

app = Flask(__name__)

@app.route("/api/model-metrics", methods=["GET"])
def get_model_metrics():
    """
    Returns model metrics including confusion matrices.
    If metrics file exists, load it; otherwise train models.
    """
    metrics_path = "model/metrics.json"

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        # Load data and train models
        df = load_and_clean_data("data/flight_delay_data.csv")
        metrics = train_classification_models(df)
        os.makedirs("model", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return jsonify(metrics)


Base.metadata.create_all(bind=engine)

# Load trained models
def load_models():
    """Load all trained ML models"""
    models_dir = Path("models")
    
    try:
        rf_model = joblib.load(models_dir / "random_forest_classifier.pkl")
        gb_model = joblib.load(models_dir / "gradient_boosting_regressor.pkl")
        label_encoders = joblib.load(models_dir / "label_encoders.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        
        return {
            'rf_classifier': rf_model,
            'gb_regressor': gb_model,
            'label_encoders': label_encoders,
            'scaler': scaler
        }
    except FileNotFoundError:
        print("Warning: Models not found. Please train models first using: python scripts/train_all_models.py")
        return None

models = load_models()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

prediction_history = []

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        db = SessionLocal()
        from models.user_schema import User
        
        # Check if user exists
        existing_user = db.query(User).filter_by(email=email).first()
        if existing_user:
            db.close()
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        user_id = str(uuid.uuid4())
        new_user = User(
            id=user_id,
            email=email,
            password_hash=auth_manager.hash_password(password),
            full_name=full_name
        )
        db.add(new_user)
        db.commit()
        
        token = auth_manager.generate_token(user_id, email)
        db.close()
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user_id,
                'email': email,
                'full_name': full_name
            }
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        db = SessionLocal()
        from models.user_schema import User
        
        user = db.query(User).filter_by(email=email).first()
        if not user or not auth_manager.verify_password(password, user.password_hash):
            db.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        token = auth_manager.generate_token(user.id, user.email)
        db.close()
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/auth/verify', methods=['GET'])
@require_auth
def verify_token():
    """Verify authentication token"""
    return jsonify({
        'valid': True,
        'user': request.user
    }), 200

@app.route('/api/weather/<airport_code>', methods=['GET'])
@require_auth
def get_weather(airport_code):
    """Get weather data for airport"""
    try:
        weather = weather_service.get_weather_by_airport(airport_code.upper())
        if not weather:
            return jsonify({'error': f'Weather data not available for {airport_code}'}), 404
        
        impact = weather_service.get_weather_impact_on_delay(weather)
        return jsonify({
            'weather': weather,
            'impact': impact
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/weather/impact', methods=['POST'])
@require_auth
def get_weather_impact():
    """Get weather impact on flight delays"""
    try:
        data = request.json
        origin = data.get('origin')
        destination = data.get('destination')
        
        origin_weather = weather_service.get_weather_by_airport(origin)
        dest_weather = weather_service.get_weather_by_airport(destination)
        
        origin_impact = weather_service.get_weather_impact_on_delay(origin_weather)
        dest_impact = weather_service.get_weather_impact_on_delay(dest_weather)
        
        return jsonify({
            'origin': {
                'airport': origin,
                'weather': origin_weather,
                'impact': origin_impact
            },
            'destination': {
                'airport': destination,
                'weather': dest_weather,
                'impact': dest_impact
            },
            'combined_risk': (origin_impact['risk_factor'] + dest_impact['risk_factor']) / 2
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
@require_auth
def predict():
    """Predict flight delay using classification and regression models"""
    try:
        if not models:
            return jsonify({'error': 'Models not loaded. Please train models first.'}), 500
        
        data = request.json
        
        # Extract input data
        airline = data.get('airline', '').upper()
        origin = data.get('origin', '').upper()
        dest = data.get('dest', '').upper()
        distance = float(data.get('distance', 0))
        crs_dep_time = int(data.get('crs_dep_time', 0))
        flight_date = data.get('flight_date')
        
        # Parse flight date
        date_obj = datetime.strptime(flight_date, '%Y-%m-%d')
        dep_hour = crs_dep_time // 100
        dow = date_obj.weekday()
        month = date_obj.month
        
        # Prepare features
        features = pd.DataFrame([{
            'AIRLINE': airline,
            'ORIGIN_AIRPORT': origin,
            'DESTINATION_AIRPORT': dest,
            'DISTANCE': distance,
            'HOUR': dep_hour,
            'DAY_OF_WEEK': dow,
            'MONTH': month
        }])
        
        # Encode categorical variables
        for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
            if col in models['label_encoders']:
                try:
                    features[col] = models['label_encoders'][col].transform(features[col].astype(str))
                except ValueError:
                    features[col] = 0
        
        # Scale features
        X = models['scaler'].transform(features)
        
        # Classification prediction (delayed or not)
        classification_pred = models['rf_classifier'].predict(X)[0]
        classification_proba = models['rf_classifier'].predict_proba(X)[0]
        
        # Regression prediction (exact delay in minutes)
        regression_pred = models['gb_regressor'].predict(X)[0]
        
        result = {
            'classification': {
                'prediction': 'Delayed' if classification_pred == 1 else 'On Time',
                'delay_probability': float(classification_proba[1]),
                'on_time_probability': float(classification_proba[0])
            },
            'regression': {
                'predicted_delay_minutes': float(max(0, regression_pred))
            },
            'flight_info': {
                'airline': airline,
                'origin': origin,
                'destination': dest,
                'distance': distance,
                'departure_hour': dep_hour,
                'flight_date': flight_date
            }
        }
        
        # Store prediction history
        prediction_history.append({
            'airline': airline,
            'origin': origin,
            'dest': dest,
            'prediction': result['classification']['prediction']
        })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/overall', methods=['GET'])
@require_auth
def get_overall_stats():
    """Get overall delay statistics from data warehouse"""
    try:
        db = SessionLocal()
        stats = OLAPAnalytics.get_overall_statistics(db)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/by-airline', methods=['GET'])
@require_auth
def get_airline_stats():
    """Get delay statistics by airline"""
    try:
        db = SessionLocal()
        stats = OLAPAnalytics.get_delay_by_airline(db)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/by-airport', methods=['GET'])
@require_auth
def get_airport_stats():
    """Get delay statistics by airport"""
    try:
        airport_type = request.args.get('type', 'origin')
        db = SessionLocal()
        stats = OLAPAnalytics.get_delay_by_airport(db, airport_type)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/by-route', methods=['GET'])
@require_auth
def get_route_stats():
    """Get delay statistics by route"""
    try:
        limit = request.args.get('limit', 20, type=int)
        db = SessionLocal()
        stats = OLAPAnalytics.get_delay_by_route(db, limit)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/by-hour', methods=['GET'])
@require_auth
def get_hour_stats():
    """Get delay statistics by hour of day"""
    try:
        db = SessionLocal()
        stats = OLAPAnalytics.get_delay_by_hour(db)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/by-day', methods=['GET'])
@require_auth
def get_day_stats():
    """Get delay statistics by day of week"""
    try:
        db = SessionLocal()
        stats = OLAPAnalytics.get_delay_by_day_of_week(db)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/by-month', methods=['GET'])
@require_auth
def get_month_stats():
    """Get delay statistics by month"""
    try:
        db = SessionLocal()
        stats = OLAPAnalytics.get_delay_by_month(db)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics/drill-down/<route_code>', methods=['GET'])
@require_auth
def drill_down_route(route_code):
    """Drill-down analysis for specific route"""
    try:
        db = SessionLocal()
        stats = OLAPAnalytics.drill_down_route_details(db, route_code)
        db.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feature-importance', methods=['GET'])
@require_auth
def get_feature_importance():
    """Get feature importance from trained models"""
    try:
        models_dir = Path("models")
        with open(models_dir / "metrics.json", 'r') as f:
            import json
            metrics = json.load(f)
            return jsonify(metrics.get('feature_importance', []))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model-metrics', methods=['GET'])
@require_auth
def get_model_metrics():
    """Get all model performance metrics"""
    try:
        models_dir = Path("models")
        with open(models_dir / "metrics.json", 'r') as f:
            import json
            metrics = json.load(f)
            return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/statistics', methods=['GET'])
@require_auth
def get_statistics():
    """Get prediction statistics"""
    if not prediction_history:
        return jsonify({
            'total_predictions': 0,
            'delayed_count': 0,
            'on_time_count': 0,
            'delay_rate': 0,
            'by_airline': {},
            'by_route': {}
        })
    
    total = len(prediction_history)
    delayed = sum(1 for p in prediction_history if p['prediction'] == 'Delayed')
    on_time = total - delayed
    
    # Statistics by airline
    by_airline = defaultdict(lambda: {'total': 0, 'delayed': 0})
    for p in prediction_history:
        by_airline[p['airline']]['total'] += 1
        if p['prediction'] == 'Delayed':
            by_airline[p['airline']]['delayed'] += 1
    
    airline_stats = {}
    for airline, stats in by_airline.items():
        airline_stats[airline] = {
            'total': stats['total'],
            'delayed': stats['delayed'],
            'delay_rate': round(stats['delayed'] / stats['total'] * 100, 1)
        }
    
    # Statistics by route
    by_route = defaultdict(lambda: {'total': 0, 'delayed': 0})
    for p in prediction_history:
        route = f"{p['origin']}-{p['dest']}"
        by_route[route]['total'] += 1
        if p['prediction'] == 'Delayed':
            by_route[route]['delayed'] += 1
    
    route_stats = {}
    for route, stats in by_route.items():
        route_stats[route] = {
            'total': stats['total'],
            'delayed': stats['delayed'],
            'delay_rate': round(stats['delayed'] / stats['total'] * 100, 1)
        }
    
    return jsonify({
        'total_predictions': total,
        'delayed_count': delayed,
        'on_time_count': on_time,
        'delay_rate': round(delayed / total * 100, 1),
        'by_airline': airline_stats,
        'by_route': route_stats
    })

@app.route('/api/airlines', methods=['GET'])
@require_auth
def get_airlines():
    """Get list of available airlines from database"""
    try:
        db = SessionLocal()
        from models.warehouse_schema import DimAirline
        airlines = db.query(DimAirline.airline_code).all()
        db.close()
        return jsonify([a[0] for a in airlines])
    except Exception as e:
        return jsonify(['AA', 'DL', 'UA', 'SW', 'B6', 'AS'])

@app.route('/api/airports', methods=['GET'])
@require_auth
def get_airports():
    """Get list of available airports from database"""
    try:
        db = SessionLocal()
        from models.warehouse_schema import DimAirport
        airports = db.query(DimAirport.airport_code).all()
        db.close()
        return jsonify([a[0] for a in airports])
    except Exception as e:
        return jsonify(['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MIA'])

@app.route('/api/clustering/airport-clusters', methods=['GET'])
@require_auth
def get_airport_clusters():
    """Get airport clustering results"""
    try:
        models_dir = Path("models")
        with open(models_dir / "metrics.json", 'r') as f:
            import json
            metrics = json.load(f)
            clusters = metrics.get('clustering', {}).get('clusters', {})
            
            # Group airports by cluster
            cluster_groups = defaultdict(list)
            for airport, cluster_id in clusters.items():
                cluster_groups[cluster_id].append(airport)
            
            return jsonify({
                'clusters': dict(cluster_groups),
                'n_clusters': metrics.get('clustering', {}).get('n_clusters', 5)
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/clustering/airport-details/<airport_code>', methods=['GET'])
@require_auth
def get_airport_cluster_details(airport_code):
    """Get detailed cluster information for an airport"""
    try:
        db = SessionLocal()
        from models.warehouse_schema import DimAirport
        
        airport = db.query(DimAirport).filter_by(airport_code=airport_code).first()
        if not airport:
            db.close()
            return jsonify({'error': 'Airport not found'}), 404
        
        # Get delay statistics for this airport
        stats = OLAPAnalytics.get_delay_by_airport(db, 'origin')
        airport_stats = next((s for s in stats if s['airport_code'] == airport_code), None)
        
        db.close()
        
        return jsonify({
            'airport_code': airport_code,
            'airport_name': airport.airport_name,
            'city': airport.city,
            'state': airport.state,
            'statistics': airport_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analysis/top-delay-factors', methods=['GET'])
@require_auth
def get_top_delay_factors():
    """Get top factors contributing to delays"""
    try:
        models_dir = Path("models")
        with open(models_dir / "metrics.json", 'r') as f:
            import json
            metrics = json.load(f)
            importance = metrics.get('feature_importance', [])
            
            # Return top 5 features
            return jsonify({
                'top_factors': importance[:5],
                'all_factors': importance
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analysis/delay-patterns', methods=['GET'])
@require_auth
def get_delay_patterns():
    """Get comprehensive delay patterns"""
    try:
        db = SessionLocal()
        
        patterns = {
            'by_hour': OLAPAnalytics.get_delay_by_hour(db),
            'by_day': OLAPAnalytics.get_delay_by_day_of_week(db),
            'by_month': OLAPAnalytics.get_delay_by_month(db),
            'top_airlines': OLAPAnalytics.get_delay_by_airline(db)[:5],
            'top_routes': OLAPAnalytics.get_delay_by_route(db, 10)
        }
        
        db.close()
        return jsonify(patterns)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
