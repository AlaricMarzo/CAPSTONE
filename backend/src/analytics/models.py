#!/usr/bin/env python3
"""
Analytics Models Runner
Runs Descriptive, Predictive, and Prescriptive analytics models
"""
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    descriptive_dir = script_dir / "Descriptive"
    predictive_dir = script_dir / "Predictive"
    prescriptive_dir = script_dir / "prescriptive"

    print("Running Descriptive Analytics Models...")

    # Run Descriptive analytics
    print("Running Descriptive analytics...")
    try:
        result = subprocess.run([
            sys.executable, "descriptive.py"
        ], cwd=descriptive_dir, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"Descriptive failed: {result.stderr}")
        else:
            print("Descriptive completed successfully")

    except subprocess.TimeoutExpired:
        print("Descriptive timed out")
    except Exception as e:
        print(f"Error running Descriptive: {e}")

    print("Running Predictive Analytics Models...")

    # Run XGBoost model
    print("Running XGBoost model...")
    try:
        result = subprocess.run([
            sys.executable, "xgboost_model.py"
        ], cwd=predictive_dir, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"XGBoost failed: {result.stderr}")
        else:
            print("XGBoost completed successfully")

    except subprocess.TimeoutExpired:
        print("XGBoost timed out")
    except Exception as e:
        print(f"Error running XGBoost: {e}")

    # Run Random Forest model
    print("Running Random Forest model...")
    try:
        result = subprocess.run([
            sys.executable, "random_forest.py", "--topn", "5", "--min_cov", "0.7"
        ], cwd=predictive_dir, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Random Forest failed: {result.stderr}")
        else:
            print("Random Forest completed successfully")

    except subprocess.TimeoutExpired:
        print("Random Forest timed out")
    except Exception as e:
        print(f"Error running Random Forest: {e}")

    print("Predictive analytics models completed.")

    print("Running Prescriptive Analytics Models...")

    # Run Prescriptive analytics
    print("Running Prescriptive analytics...")
    try:
        result = subprocess.run([
            sys.executable, "prescriptive.py"
        ], cwd=prescriptive_dir, capture_output=True, text=True, timeout=1200)

        if result.returncode != 0:
            print(f"Prescriptive failed: {result.stderr}")
        else:
            print("Prescriptive completed successfully")

    except subprocess.TimeoutExpired:
        print("Prescriptive timed out")
    except Exception as e:
        print(f"Error running Prescriptive: {e}")

    print("All analytics models (Descriptive, Predictive, Prescriptive) completed.")
    print('{"success": true, "message": "Analytics pipeline completed successfully"}')

if __name__ == "__main__":
    main()
