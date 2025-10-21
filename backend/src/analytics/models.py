import subprocess
import sys
import os
from pathlib import Path

# Define paths relative to this script's location
BASE_DIR = Path(__file__).parent
DESCRIPTIVE_DIR = BASE_DIR / "descriptive"
PREDICTIVE_DIR = BASE_DIR / "predictive"
PRESCRIPTIVE_DIR = BASE_DIR / "prescriptive"

def get_csv_path():
    """Prompt user for CSV file path"""
    while True:
        csv_path = input("Enter the path to the CSV file to analyze: ").strip()
        if not csv_path:
            print("Path cannot be empty. Please try again.")
            continue
        path = Path(csv_path)
        if not path.exists():
            print(f"File not found: {path}. Please check the path and try again.")
            continue
        if path.suffix.lower() not in ['.csv', '.xlsx', '.xls']:
            print("File must be a CSV, XLSX, or XLS file. Please try again.")
            continue
        return path

def run_script(script_path, cwd=None, extra_args=None):
    """Run a Python script using subprocess."""
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully ran {script_path.name}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to run {script_path.name}: {e}")
        print(e.stderr)
        return False
    return True

def run_descriptive(csv_path):
    """Run descriptive analytics: main.py with provided CSV"""
    main_script = DESCRIPTIVE_DIR / "main.py"
    if main_script.exists():
        print("Running Descriptive Analytics...")
        try:
            # Simulate input for the script (choice 2: manual path)
            process = subprocess.Popen([sys.executable, str(main_script)], cwd=DESCRIPTIVE_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            process.stdin.write("2\n")  # Choose manual path
            process.stdin.write(f"{csv_path}\n")  # Provide path
            process.stdin.flush()
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print("✓ Successfully ran descriptive analytics")
                print(stdout)
                return True
            else:
                print(f"✗ Failed to run descriptive analytics: {stderr}")
                return False
        except Exception as e:
            print(f"✗ Error running descriptive: {e}")
            return False
    else:
        print(f"✗ Descriptive main.py not found at {main_script}")
        return False

def run_predictive(csv_path):
    """Run predictive analytics: all model scripts then compare"""
    models = [
        "arima_ets.py",
        "random_forest.py",
        "xgboost_model.py",
        "extra_trees.py",
        "gradient_boosting.py",
        "lstm.py"
    ]
    success = True
    for model in models:
        script_path = PREDICTIVE_DIR / model
        if script_path.exists():
            print(f"Running Predictive Model: {model}...")
            # Pass the CSV path as an argument to the scripts
            if not run_script(script_path, cwd=PREDICTIVE_DIR, extra_args=[str(csv_path)]):
                success = False
        else:
            print(f"✗ Predictive script {model} not found at {script_path}")
            success = False

    # Run compare_models.py after all models
    compare_script = PREDICTIVE_DIR / "compare_models.py"
    if compare_script.exists():
        print("Running Model Comparison...")
        if not run_script(compare_script, cwd=PREDICTIVE_DIR):
            success = False
    else:
        print(f"✗ Compare script not found at {compare_script}")
        success = False

    return success

def run_prescriptive(csv_path):
    """Run prescriptive analytics: prescriptive.py with provided CSV"""
    script_path = PRESCRIPTIVE_DIR / "prescriptive.py"
    if script_path.exists():
        print("Running Prescriptive Analytics...")
        # Pass the CSV path as an argument to the script
        return run_script(script_path, cwd=PRESCRIPTIVE_DIR, extra_args=[str(csv_path)])
    else:
        print(f"✗ Prescriptive script not found at {script_path}")
        return False

def main():
    print("Starting All Analytics Models...")
    print("=" * 50)

    # Get CSV path from user
    csv_path = get_csv_path()
    print(f"Using CSV file: {csv_path}")
    print()

    # Run descriptive
    desc_success = run_descriptive(csv_path)
    print()

    # Run predictive
    pred_success = run_predictive(csv_path)
    print()

    # Run prescriptive
    presc_success = run_prescriptive(csv_path)
    print()

    if desc_success and pred_success and presc_success:
        print("✓ All analytics models completed successfully!")
        print("Outputs are ready for PowerBI visualization:")
        print("- Descriptive: descriptive_output/")
        print("- Predictive: predictive/out/")
        print("- Prescriptive: prescriptive_output/")
    else:
        print("✗ Some models failed. Check outputs and logs above.")

if __name__ == "__main__":
    main()
