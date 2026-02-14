"""
Complete setup guide for Flight Delay Prediction System
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_step(step_num, title):
    print(f"\n{'─'*70}")
    print(f"STEP {step_num}: {title}")
    print(f"{'─'*70}")

def main():
    print_header("FLIGHT DELAY PREDICTION SYSTEM - SETUP GUIDE")
    
    print_step(1, "Check Python Installation")
    print(f"Python version: {sys.version}")
    print(f"✓ Python is installed correctly")
    
    print_step(2, "Install Dependencies")
    print("Run the following command:")
    print("\n  pip install -r requirements.txt\n")
    print("This will install all required packages:")
    print("  - pandas, numpy (data processing)")
    print("  - scikit-learn (machine learning)")
    print("  - sqlalchemy (database ORM)")
    print("  - kaggle (dataset download)")
    print("  - streamlit (dashboard)")
    print("  - plotly (visualizations)")
    
    print_step(3, "Check Data Folder")
    print("Run the diagnostic script:")
    print("\n  python scripts/check_data.py\n")
    print("This will show you:")
    print("  - If data folder exists")
    print("  - What CSV files are present")
    print("  - File sizes and structure")
    
    print_step(4, "Download Dataset (if needed)")
    print("If data folder is empty, run:")
    print("\n  python scripts/download_kaggle_data.py\n")
    print("This requires Kaggle API setup:")
    print("  1. Go to: https://www.kaggle.com/settings/account")
    print("  2. Click 'Create New API Token'")
    print("  3. Place kaggle.json in ~/.kaggle/")
    print("  4. Run chmod 600 ~/.kaggle/kaggle.json (Mac/Linux)")
    
    print_step(5, "Train Models")
    print("Once data is ready, train all models:")
    print("\n  python scripts/train_all_models.py\n")
    print("This will:")
    print("  - Load and clean data")
    print("  - Create database tables")
    print("  - Train classification, regression, and clustering models")
    print("  - Save model metrics")
    
    print_step(6, "Run Flask API")
    print("Start the backend API:")
    print("\n  python app.py\n")
    print("API will be available at: http://localhost:5000")
    
    print_step(7, "Launch Dashboard")
    print("In a new terminal, run:")
    print("\n  streamlit run dashboard.py\n")
    print("Dashboard will open at: http://localhost:8501")
    
    print_header("TROUBLESHOOTING")
    
    print("\n❌ 'No module named requests'")
    print("   → Run: pip install -r requirements.txt")
    
    print("\n❌ 'No flight data file found'")
    print("   → Run: python scripts/check_data.py")
    print("   → Then: python scripts/download_kaggle_data.py")
    
    print("\n❌ 'Kaggle API error'")
    print("   → Check kaggle.json is in ~/.kaggle/")
    print("   → Verify API token is valid")
    
    print("\n❌ 'Database error'")
    print("   → Delete flight_delay.db if it exists")
    print("   → Run: python scripts/train_all_models.py")
    
    print_header("QUICK START COMMANDS")
    
    print("\n# Install dependencies")
    print("pip install -r requirements.txt")
    
    print("\n# Check data")
    print("python scripts/check_data.py")
    
    print("\n# Download dataset (if needed)")
    print("python scripts/download_kaggle_data.py")
    
    print("\n# Train models")
    print("python scripts/train_all_models.py")
    
    print("\n# Run API (Terminal 1)")
    print("python app.py")
    
    print("\n# Run Dashboard (Terminal 2)")
    print("streamlit run dashboard.py")
    
    print_header("SETUP COMPLETE!")
    print("\nYour Flight Delay Prediction System is ready to use!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
