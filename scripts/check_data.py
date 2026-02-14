import sys
from pathlib import Path

# <CHANGE> Add sys.path to find project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd

def check_data_folder():
    """Diagnostic script to check data folder and files"""
    print("\n" + "="*60)
    print("DATA FOLDER DIAGNOSTIC")
    print("="*60)
    
    data_dir = Path("data")
    
    # Check if data folder exists
    if not data_dir.exists():
        print(f"❌ Data folder does not exist!")
        print(f"   Creating: {data_dir.absolute()}")
        data_dir.mkdir(exist_ok=True)
        print(f"✓ Data folder created at: {data_dir.absolute()}")
        return
    
    print(f"✓ Data folder found at: {data_dir.absolute()}")
    
    # List all files in data folder
    files = list(data_dir.glob("*"))
    
    if not files:
        print(f"\n❌ Data folder is EMPTY!")
        print(f"\nTo fix this, run:")
        print(f"   python scripts/generate_sample_data.py")
        print(f"\nOr manually download from:")
        print(f"   https://www.kaggle.com/datasets/undersc0re/flight-delay-and-causes")
        print(f"   Then extract CSV files to: {data_dir.absolute()}")
        return
    
    print(f"\n✓ Found {len(files)} file(s) in data folder:\n")
    
    csv_files = []
    for file in sorted(files):
        file_size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   📄 {file.name} ({file_size_mb:.2f} MB)")
        
        if file.suffix.lower() == '.csv':
            csv_files.append(file)
    
    if not csv_files:
        print(f"\n❌ No CSV files found!")
        print(f"   Please extract the downloaded ZIP file to: {data_dir.absolute()}")
        return
    
    print(f"\n✓ Found {len(csv_files)} CSV file(s)")
    
    # Analyze each CSV file
    print(f"\n" + "-"*60)
    print("CSV FILE ANALYSIS")
    print("-"*60)
    
    for csv_file in csv_files:
        print(f"\n📊 {csv_file.name}:")
        try:
            df = pd.read_csv(csv_file, nrows=5)
            print(f"   Rows: {len(df)} (showing first 5)")
            print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
            print(f"   Data types: {dict(df.dtypes)}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. If CSV files are present, run:")
    print(f"   python scripts/train_all_models.py")
    print(f"\n2. If data folder is empty, generate sample data:")
    print(f"   python scripts/generate_sample_data.py")
    print(f"\n3. Or manually download from Kaggle and extract to:")
    print(f"   {data_dir.absolute()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    check_data_folder()
