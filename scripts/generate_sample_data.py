import sys
from pathlib import Path

# <CHANGE> Add sys.path to find project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_flight_data(num_records=5000):
    """
    Generate realistic sample flight delay data for testing and demonstration.
    This creates a complete dataset without requiring Kaggle API.
    """
    
    np.random.seed(42)
    
    # Airlines
    airlines = ['AA', 'DL', 'UA', 'SW', 'B6', 'AS', 'NK', 'F9', 'G4', 'SY']
    
    # Major US Airports
    airports = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS', 'MIA',
                'EWR', 'BOS', 'MSP', 'DTW', 'PHL', 'LGA', 'IAD', 'CLT', 'PHX', 'IAH']
    
    # Generate dates (last 2 years)
    start_date = datetime.now() - timedelta(days=730)
    dates = [start_date + timedelta(days=x) for x in range(730)]
    
    data = {
        'FlightDate': np.random.choice(dates, num_records),
        'Airline': np.random.choice(airlines, num_records),
        'Origin': np.random.choice(airports, num_records),
        'Destination': np.random.choice(airports, num_records),
        'ScheduledDeparture': np.random.randint(600, 2300, num_records),
        'ActualDeparture': np.random.randint(600, 2300, num_records),
        'ScheduledArrival': np.random.randint(600, 2300, num_records),
        'ActualArrival': np.random.randint(600, 2300, num_records),
        'DepDelay': np.random.normal(10, 25, num_records),  # Mean 10 min, std 25 min
        'ArrDelay': np.random.normal(8, 30, num_records),   # Mean 8 min, std 30 min
        'Distance': np.random.randint(100, 3000, num_records),
        'AirTime': np.random.randint(30, 600, num_records),
        'Cancelled': np.random.choice([0, 1], num_records, p=[0.98, 0.02]),
        'Diverted': np.random.choice([0, 1], num_records, p=[0.99, 0.01]),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure Origin != Destination
    df = df[df['Origin'] != df['Destination']].reset_index(drop=True)
    
    # Add delay reason (if delayed)
    delay_reasons = ['Weather', 'Mechanical', 'Crew', 'Security', 'Late Aircraft', 'Other']
    df['DelayReason'] = df['ArrDelay'].apply(
        lambda x: np.random.choice(delay_reasons) if x > 15 else 'On Time'
    )
    
    return df.head(num_records)

def save_sample_data():
    """Save sample data to CSV file"""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING SAMPLE FLIGHT DATA")
    print("="*60 + "\n")
    
    print("Generating 5,000 sample flight records...")
    df = generate_sample_flight_data(5000)
    
    # Save as CSV
    output_file = data_dir / "flights.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Sample data generated successfully!")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Records: {len(df):,}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"\nDataset Preview:")
    print(df.head())
    print(f"\nDataset Info:")
    print(f"  - Date Range: {df['FlightDate'].min()} to {df['FlightDate'].max()}")
    print(f"  - Airlines: {df['Airline'].nunique()}")
    print(f"  - Airports: {df['Origin'].nunique() + df['Destination'].nunique()}")
    print(f"  - Avg Departure Delay: {df['DepDelay'].mean():.2f} minutes")
    print(f"  - Avg Arrival Delay: {df['ArrDelay'].mean():.2f} minutes")
    print(f"  - Cancellation Rate: {(df['Cancelled'].sum() / len(df) * 100):.2f}%")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Load data into database:")
    print("   python scripts/etl_pipeline.py")
    print("\n2. Train models:")
    print("   python scripts/train_all_models.py")
    print("\n3. Run Flask API:")
    print("   python app.py")
    print("\n4. Launch Streamlit Dashboard:")
    print("   streamlit run dashboard.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    save_sample_data()
