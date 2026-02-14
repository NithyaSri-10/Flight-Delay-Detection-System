import sys
from pathlib import Path

# <CHANGE> Add sys.path to find project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd

def load_custom_csv():
    """
    Load custom flight data from CSV file.
    Users can place their own CSV files in the 'data' folder.
    """
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("CUSTOM DATA LOADER")
    print("="*60 + "\n")
    
    # List all CSV files in data folder
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in 'data' folder")
        print("\nTo use custom data:")
        print("1. Place your CSV file in the 'data' folder")
        print("2. Required columns:")
        print("   - FlightDate or Date")
        print("   - Airline")
        print("   - Origin")
        print("   - Destination")
        print("   - DepDelay or ArrDelay")
        print("\nOr generate sample data:")
        print("   python scripts/generate_sample_data.py")
        return None
    
    print(f"Found {len(csv_files)} CSV file(s):\n")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file.name}")
    
    if len(csv_files) == 1:
        selected_file = csv_files[0]
        print(f"\nLoading: {selected_file.name}")
    else:
        choice = input(f"\nSelect file (1-{len(csv_files)}): ")
        try:
            selected_file = csv_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection")
            return None
    
    try:
        df = pd.read_csv(selected_file)
        print(f"\n✓ Loaded {len(df):,} records")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"\nPreview:")
        print(df.head())
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

if __name__ == "__main__":
    load_custom_csv()
