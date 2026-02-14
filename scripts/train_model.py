import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from datetime import datetime

# Create sample flight data for demonstration
def create_sample_data():
    """Generate sample flight data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    airlines = ['AA', 'DL', 'UA', 'SW', 'B6', 'AS']
    airports = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MIA']
    
    data = {
        'AIRLINE': np.random.choice(airlines, n_samples),
        'ORIGIN': np.random.choice(airports, n_samples),
        'DEST': np.random.choice(airports, n_samples),
        'CRS_DEP_TIME': np.random.randint(0, 2400, n_samples),
        'ARR_DELAY': np.random.normal(10, 30, n_samples),
        'FL_DATE': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
    }
    
    df = pd.DataFrame(data)
    df['ARR_DELAY'] = df['ARR_DELAY'].clip(lower=0)
    return df

# Data preprocessing
def preprocess_data(df):
    """Clean and preprocess flight data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop rows with missing essential columns
    df = df.dropna(subset=['ARR_DELAY', 'CRS_DEP_TIME', 'AIRLINE', 'ORIGIN', 'DEST'])
    
    # Ensure CRS_DEP_TIME is numeric
    df['CRS_DEP_TIME'] = pd.to_numeric(df['CRS_DEP_TIME'], errors='coerce')
    df = df.dropna(subset=['CRS_DEP_TIME'])
    
    # Standardize codes to uppercase
    df['AIRLINE'] = df['AIRLINE'].str.upper()
    df['ORIGIN'] = df['ORIGIN'].str.upper()
    df['DEST'] = df['DEST'].str.upper()
    
    # Extract features
    df['dep_hour'] = (df['CRS_DEP_TIME'] // 100).astype(int)
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['dow'] = df['FL_DATE'].dt.dayofweek
    df['month'] = df['FL_DATE'].dt.month
    
    # Create target variable
    df['IsDelayed'] = (df['ARR_DELAY'] > 15).astype(int)
    
    return df

# Train model
def train_model(df):
    """Train Random Forest classifier"""
    # Select features
    features = ['AIRLINE', 'ORIGIN', 'DEST', 'dep_hour', 'dow', 'month']
    X = df[features].copy()
    y = df['IsDelayed']
    
    # Encode categorical variables
    label_encoders = {}
    for col in ['AIRLINE', 'ORIGIN', 'DEST']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return model, label_encoders

# Save model
def save_model(model, label_encoders):
    """Save trained model and encoders"""
    os.makedirs('models', exist_ok=True)
    with open('models/flight_delay_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("Model saved successfully!")

# Main execution
if __name__ == "__main__":
    print("Creating sample flight data...")
    df = create_sample_data()
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Training model...")
    model, label_encoders = train_model(df)
    
    print("Saving model...")
    save_model(model, label_encoders)
