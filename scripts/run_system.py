#!/usr/bin/env python
"""
Complete system runner - orchestrates all components
"""
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False

def main():
    """Run complete system"""
    print("\n" + "="*60)
    print("FLIGHT DELAY PREDICTION SYSTEM - COMPLETE SETUP")
    print("="*60)
    
    steps = [
        ([sys.executable, "scripts/download_kaggle_data.py"], 
         "Download Kaggle Dataset"),
        
        ([sys.executable, "scripts/train_all_models.py"], 
         "Train ML Models"),
        
        ([sys.executable, "-m", "pytest", "tests/", "-v"], 
         "Run Tests"),
    ]
    
    completed = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            completed += 1
        else:
            print(f"\nWarning: {description} failed. Continuing...")
    
    print(f"\n{'='*60}")
    print(f"SETUP COMPLETE: {completed}/{len(steps)} steps successful")
    print(f"{'='*60}")
    
    print("\nNext steps:")
    print("1. Start Flask API: python app.py")
    print("2. Start Streamlit Dashboard: streamlit run dashboard.py")
    print("3. Open browser to http://localhost:8501")

if __name__ == "__main__":
    main()
