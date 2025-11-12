import sys
import json
import subprocess
import os
from pathlib import Path

def run_predictive():
    """
    Run predictive analytics by executing the predictive.py script,
    which handles running all predictive model scripts.
    """
    print("Running predictive analytics...")
    analytics_dir = Path(__file__).parent
    predictive_script = analytics_dir / "Predictive" / "predictive.py"

    if not predictive_script.exists():
        return {
            "success": False,
            "message": "Predictive runner script not found",
            "details": [{"script": "predictive.py", "success": False, "error": "File not found"}]
        }

    print("Running predictive.py...")
    try:
        # Run the predictive.py script
        result = subprocess.run(
            [sys.executable, str(predictive_script)],
            cwd=str(analytics_dir / "Predictive"),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for all models
        )
        if result.returncode == 0:
            print("predictive.py completed successfully.")
            # Parse the JSON output from predictive.py
            try:
                output_data = json.loads(result.stdout.strip())
                return output_data
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "message": "Predictive analytics completed (output parsing failed)",
                    "details": [{"script": "predictive.py", "success": True, "output": result.stdout}]
                }
        else:
            print(f"predictive.py failed with code {result.returncode}: {result.stderr}")
            return {
                "success": False,
                "message": "Predictive analytics failed",
                "details": [{"script": "predictive.py", "success": False, "error": result.stderr}]
            }
    except subprocess.TimeoutExpired:
        print("predictive.py timed out.")
        return {
            "success": False,
            "message": "Predictive analytics timed out",
            "details": [{"script": "predictive.py", "success": False, "error": "Timeout"}]
        }
    except Exception as e:
        print(f"Error running predictive.py: {str(e)}")
        return {
            "success": False,
            "message": f"Error running predictive analytics: {str(e)}",
            "details": [{"script": "predictive.py", "success": False, "error": str(e)}]
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
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        result = run_analytics(output_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
