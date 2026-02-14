import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.etl_pipeline import run_etl, load_and_clean_data
from services.ml_models import FlightDelayModels
from config.database import SessionLocal
import json
import numpy as np
import os

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def train_all_models():
    """Train all ML models"""
    print("=" * 60)
    print("FLIGHT DELAY PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Run ETL
    print("\n[STEP 1] Running ETL Pipeline...")
    run_etl()
    
    # Step 2: Load data
    print("\n[STEP 2] Loading cleaned data...")
    result = load_and_clean_data()
    
    # handle both possible return shapes:
    # - older versions returned (flights_df, airlines_df, airports_df)
    # - current pipeline returns flights_df
    if result is None:
        print("Error: load_and_clean_data() returned None. Check data files.")
        return

    if isinstance(result, tuple) or isinstance(result, list):
        # pick first element that looks like a DataFrame
        flights_df = None
        for item in result:
            # crude check for pandas DataFrame without importing pandas here
            try:
                # pandas DataFrame has 'shape' attribute and 'columns'
                if hasattr(item, "shape") and hasattr(item, "columns"):
                    flights_df = item
                    break
            except Exception:
                continue
        if flights_df is None:
            print("Error: load_and_clean_data() returned a tuple but no DataFrame found.")
            return
    else:
        flights_df = result

    # Validate flights_df
    try:
        # lazy import of pandas to check
        import pandas as pd
        if not isinstance(flights_df, pd.DataFrame):
            print("Error: Loaded object is not a pandas DataFrame.")
            return
        if flights_df.empty:
            print("Error: flights DataFrame is empty after cleaning.")
            return
    except Exception as e:
        print(f"Error validating DataFrame: {e}")
        return

    # Step 3: Train models
    print("\n[STEP 3] Training ML Models...")
    ml_models = FlightDelayModels()
    metrics = {}

    # Ensure output folder exists
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train classification model
    try:
        classification_metrics = ml_models.train_classification_model(flights_df)
        metrics['classification'] = classification_metrics
        print("✓ Classification model trained.")
    except Exception as e:
        print(f"⚠️ Classification training failed: {e}")
        metrics['classification'] = {'error': str(e)}

    # Train regression model
    try:
        regression_metrics = ml_models.train_regression_model(flights_df)
        metrics['regression'] = regression_metrics
        print("✓ Regression model trained.")
    except Exception as e:
        print(f"⚠️ Regression training failed: {e}")
        metrics['regression'] = {'error': str(e)}

    # Train clustering model
    try:
        clustering_results = ml_models.train_clustering_model(flights_df)
        metrics['clustering'] = clustering_results
        print("✓ Clustering model trained.")
    except Exception as e:
        print(f"⚠️ Clustering training failed: {e}")
        metrics['clustering'] = {'error': str(e)}

    # Feature importance
    try:
        feature_importance = ml_models.get_feature_importance(flights_df)
        metrics['feature_importance'] = feature_importance
        print("✓ Feature importance extracted.")
    except Exception as e:
        print(f"⚠️ Feature importance extraction failed: {e}")
        metrics['feature_importance'] = {'error': str(e)}

    # Save metrics as JSON (convert numpy types)
    metrics_serializable = convert_to_serializable(metrics)
    metrics_path = models_dir / "metrics.json"
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_path}")
    except Exception as e:
        print(f"⚠️ Failed to write metrics.json: {e}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModels saved to: models/ (if training succeeded)")
    print("Metrics saved to: models/metrics.json")

if __name__ == "__main__":
    train_all_models()
