#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive Analytics Runner: Executes all predictive model scripts in this directory.

This script automatically discovers and runs all Python scripts in the Predictive directory,
excluding __init__.py and itself. Each model script is expected to load data from the database
and generate forecasts, plots, and summaries.

Usage:
    python predictive.py [--timeout SECONDS] [--exclude SCRIPT1 SCRIPT2 ...]

Arguments:
    --timeout: Timeout in seconds per script (default: 600)
    --exclude: List of script names to exclude (e.g., --exclude lstm.py gradient_boosting.py)
"""
import sys
import json
import subprocess
import os
from pathlib import Path
import argparse

def run_all_predictive_models(timeout: int = 600, exclude: list = None):
    """
    Run all predictive model scripts in the current directory.

    Args:
        timeout: Timeout in seconds for each script
        exclude: List of script names to exclude

    Returns:
        Dict with overall success, message, and details for each script
    """
    if exclude is None:
        exclude = []

    # Get the directory of this script
    predictive_dir = Path(__file__).parent

    # Find all .py files except __init__.py and this script
    all_scripts = [
        f for f in predictive_dir.glob("*.py")
        if f.name not in ["__init__.py", "export_dashboard_artifacts","predictive.py"] and f.name not in exclude
    ]

    if not all_scripts:
        return {
            "success": False,
            "message": "No predictive model scripts found to run",
            "details": []
        }

    print(f"Found {len(all_scripts)} predictive model scripts to run:")
    for script in all_scripts:
        print(f"  - {script.name}")
    print()

    results = []
    successful_scripts = 0

    for script_path in all_scripts:
        script_name = script_path.name
        print(f"Running {script_name}...")

        try:
            # Run the script with timeout
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(predictive_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                print(f"✅ {script_name} completed successfully.")
                results.append({
                    "script": script_name,
                    "success": True,
                    "output": result.stdout.strip(),
                    "error": result.stderr.strip() if result.stderr else None
                })
                successful_scripts += 1
            else:
                print(f"❌ {script_name} failed with exit code {result.returncode}")
                results.append({
                    "script": script_name,
                    "success": False,
                    "error": result.stderr.strip() if result.stderr else "Unknown error",
                    "output": result.stdout.strip() if result.stdout else None
                })

        except subprocess.TimeoutExpired:
            print(f"⏰ {script_name} timed out after {timeout} seconds.")
            results.append({
                "script": script_name,
                "success": False,
                "error": f"Timeout after {timeout} seconds"
            })
        except Exception as e:
            print(f"💥 Error running {script_name}: {str(e)}")
            results.append({
                "script": script_name,
                "success": False,
                "error": str(e)
            })

        print()  # Blank line between scripts

    # Overall result
    total_scripts = len(all_scripts)
    overall_success = successful_scripts > 0

    if overall_success:
        message = f"Predictive analytics completed: {successful_scripts}/{total_scripts} scripts succeeded"
    else:
        message = f"All predictive scripts failed ({total_scripts} total)"

    summary = {
        "success": overall_success,
        "message": message,
        "total_scripts": total_scripts,
        "successful_scripts": successful_scripts,
        "failed_scripts": total_scripts - successful_scripts,
        "details": results
    }

    return summary

def main():
    parser = argparse.ArgumentParser(description="Run all predictive analytics models")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds per script (default: 600)"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Script names to exclude (e.g., lstm.py gradient_boosting.py)"
    )

    args = parser.parse_args()

    try:
        result = run_all_predictive_models(timeout=args.timeout, exclude=args.exclude)

        # Print summary
        print("=" * 60)
        print("PREDICTIVE ANALYTICS SUMMARY")
        print("=" * 60)
        print(f"Total scripts: {result['total_scripts']}")
        print(f"Successful: {result['successful_scripts']}")
        print(f"Failed: {result['failed_scripts']}")
        print(f"Overall: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        print()
        print(result['message'])
        print()

        if result['details']:
            print("DETAILS:")
            for detail in result['details']:
                status = "✅" if detail['success'] else "❌"
                print(f"  {status} {detail['script']}")
                if not detail['success'] and detail.get('error'):
                    print(f"      Error: {detail['error'][:100]}{'...' if len(detail['error']) > 100 else ''}")

        # Output JSON for programmatic use
        print("\n" + "=" * 60)
        print(json.dumps(result, indent=2))

    except Exception as e:
        error_result = {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "details": []
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
