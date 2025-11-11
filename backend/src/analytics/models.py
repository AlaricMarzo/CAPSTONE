import sys
import json
import subprocess
import os
from pathlib import Path

def run_predictive():
    """
    Run predictive analytics by executing the predictive model scripts.
    """
    print("Running predictive analytics...")
    analytics_dir = Path(__file__).parent
    predictive_dir = analytics_dir / "Predictive"

    # List of predictive scripts to run
    scripts = [
        "xgboost_model.py",
    ]

    results = []
    for script in scripts:
        script_path = predictive_dir / script
        if not script_path.exists():
            print(f"Warning: {script} not found, skipping.")
            continue

        print(f"Running {script}...")
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(predictive_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout per script
            )
            if result.returncode == 0:
                print(f"{script} completed successfully.")
                results.append({"script": script, "success": True, "output": result.stdout})
            else:
                print(f"{script} failed with code {result.returncode}: {result.stderr}")
                results.append({"script": script, "success": False, "error": result.stderr})
        except subprocess.TimeoutExpired:
            print(f"{script} timed out.")
            results.append({"script": script, "success": False, "error": "Timeout"})
        except Exception as e:
            print(f"Error running {script}: {str(e)}")
            results.append({"script": script, "success": False, "error": str(e)})

    # Determine overall success: succeed if at least one script succeeded
    overall_success = any(r["success"] for r in results)
    if overall_success:
        message = "Predictive analytics completed successfully"
    else:
        message = "All predictive scripts failed"

    return {
        "success": overall_success,
        "message": message,
        "details": results
    }

def run_analytics(output_path):
    # Placeholder for full analytics pipeline
    # For now, only run predictive
    predictive_result = run_predictive()
    result = {
        "success": predictive_result["success"],
        "message": predictive_result["message"],
        "predictive": predictive_result
    }
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Output path required"}))
        sys.exit(1)

    output_path = sys.argv[1]
    try:
        result = run_analytics(output_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
