import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------
# TRAIN AND EVALUATE MODELS
# -----------------------
def train_classification_models(df):
    """
    Train Logistic Regression and Random Forest models,
    store performance metrics + confusion matrices.
    """

    # --- Feature & Target ---
    features = ['airline_code', 'origin_code', 'dest_code', 'dep_hour', 'dow', 'month']
    X = df[features]
    y = df['IsDelayed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {'classification': {}}

    # --- Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    results['classification']['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'confusion_matrix': cm_lr.tolist()  # ✅ Convert numpy array → list
    }

    joblib.dump(log_reg, "model/logistic_regression.pkl")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    results['classification']['random_forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'confusion_matrix': cm_rf.tolist()  # ✅
    }

    joblib.dump(rf, "model/random_forest.pkl")

    print("✅ Models trained and saved successfully.")
    return results


class FlightDelayModels:
    """Machine Learning models for flight delay prediction"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def normalize_columns(self, df: pd.DataFrame):
        """Normalize column names to handle both sample data and warehouse schema"""
        df = df.copy()
        
        # Map sample data column names to standard names
        column_mapping = {
            'Airline': 'AIRLINE',
            'Origin': 'ORIGIN_AIRPORT',
            'Destination': 'DESTINATION_AIRPORT',
            'DepDelay': 'DEPARTURE_DELAY',
            'ArrDelay': 'ARRIVAL_DELAY',
            'ScheduledDeparture': 'SCHEDULED_DEPARTURE',
            'Distance': 'DISTANCE',
            'FlightDate': 'FLIGHT_DATE'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Create IS_DELAYED column if it doesn't exist
        if 'IS_DELAYED' not in df.columns:
            if 'ARRIVAL_DELAY' in df.columns:
                df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
            else:
                raise ValueError("Cannot create IS_DELAYED: ARRIVAL_DELAY column not found")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler=False):
        """Prepare features for ML models"""
        df = self.normalize_columns(df)
        df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
        for col in categorical_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe")
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unknown categories
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # If unknown category, use the first category
                    df[col] = self.label_encoders[col].transform(df[col].astype(str).fillna(df[col].astype(str).iloc[0]))
        
        # Extract time features
        if 'SCHEDULED_DEPARTURE' in df.columns:
            df['SCHEDULED_DEPARTURE'] = pd.to_datetime(df['SCHEDULED_DEPARTURE'], errors='coerce')
            df['HOUR'] = df['SCHEDULED_DEPARTURE'].dt.hour
            df['DAY_OF_WEEK'] = df['SCHEDULED_DEPARTURE'].dt.dayofweek
            df['MONTH'] = df['SCHEDULED_DEPARTURE'].dt.month
        else:
            # If no datetime column, create dummy time features
            df['HOUR'] = np.random.randint(0, 24, len(df))
            df['DAY_OF_WEEK'] = np.random.randint(0, 7, len(df))
            df['MONTH'] = np.random.randint(1, 13, len(df))
        
        # Select features
        feature_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE', 'HOUR', 'DAY_OF_WEEK', 'MONTH']
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols].fillna(0)
        
        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, feature_cols
    
    def train_classification_model(self, df: pd.DataFrame):
        """Train classification model (Random Forest + Logistic Regression)"""
        print("Training Classification Models...")
        
        X, feature_cols = self.prepare_features(df, fit_scaler=True)
        df = self.normalize_columns(df)
        y = df['IS_DELAYED'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest
        print("  - Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred)
        }
        print(f"    Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}")
        
        # Logistic Regression
        print("  - Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        lr_metrics = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1': f1_score(y_test, lr_pred)
        }
        print(f"    Logistic Regression - Accuracy: {lr_metrics['accuracy']:.4f}, F1: {lr_metrics['f1']:.4f}")
        
        # Save models
        joblib.dump(rf_model, self.models_dir / "random_forest_classifier.pkl")
        joblib.dump(lr_model, self.models_dir / "logistic_regression_classifier.pkl")
        joblib.dump(self.label_encoders, self.models_dir / "label_encoders.pkl")
        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        
        return {'random_forest': rf_metrics, 'logistic_regression': lr_metrics}
    
    def train_regression_model(self, df: pd.DataFrame):
        """Train regression model to predict exact delay in minutes"""
        print("Training Regression Model...")
        
        X, feature_cols = self.prepare_features(df, fit_scaler=True)
        df = self.normalize_columns(df)
        y = df['ARRIVAL_DELAY'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Gradient Boosting Regressor
        print("  - Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        
        gb_metrics = {
            'mse': mean_squared_error(y_test, gb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'r2': r2_score(y_test, gb_pred),
            'mae': np.mean(np.abs(y_test - gb_pred))
        }
        print(f"    Gradient Boosting - RMSE: {gb_metrics['rmse']:.4f}, R²: {gb_metrics['r2']:.4f}")
        
        # Save model
        joblib.dump(gb_model, self.models_dir / "gradient_boosting_regressor.pkl")
        
        return {'gradient_boosting': gb_metrics}
    
    def train_clustering_model(self, df: pd.DataFrame):
        """Train clustering model to identify airport patterns"""
        print("Training Clustering Model...")
        
        df = self.normalize_columns(df)
        
        # Aggregate data by airport
        airport_features = df.groupby('ORIGIN_AIRPORT').agg({
            'DEPARTURE_DELAY': ['mean', 'std'],
            'ARRIVAL_DELAY': ['mean', 'std'],
            'IS_DELAYED': 'mean',
            'DISTANCE': 'mean'
        }).fillna(0)
        
        airport_features.columns = ['_'.join(col).strip() for col in airport_features.columns.values]
        
        # Scale features
        X = self.scaler.fit_transform(airport_features)
        
        # K-Means clustering
        print("  - Training K-Means Clustering...")
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Save model
        joblib.dump(kmeans, self.models_dir / "kmeans_clustering.pkl")
        
        # Create cluster mapping
        cluster_mapping = dict(zip(airport_features.index, clusters))
        
        return {'clusters': cluster_mapping, 'n_clusters': 5}
    
    def get_feature_importance(self, df: pd.DataFrame):
        """Get feature importance from trained models"""
        print("Calculating Feature Importance...")
        
        X, feature_cols = self.prepare_features(df, fit_scaler=True)
        df = self.normalize_columns(df)
        y = df['IS_DELAYED'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.to_dict('records')
