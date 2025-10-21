# main.py - Runs all analytics: Descriptive, Predictive, and Prescriptive
import os, sys, tempfile, subprocess
from pathlib import Path
import pandas as pd

# If your servers are headless, keep Agg in the analytics modules themselves:
# import matplotlib
# matplotlib.use("Agg")

from clustering import run_clustering_pipeline
from mba import run_mba_pipeline


def _read_csv_robust(path_or_buf):
    """Try multiple encodings so weird CSVs still load."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path_or_buf, encoding=enc, engine="python")
        except Exception:
            continue
    # last attempt with default
    return pd.read_csv(path_or_buf)


def _browse_or_path_or_url():
    print("=" * 70)
    print("ANALYTICS — DATA IMPORT")
    print("=" * 70)
    print("1. Browse for CSV file")
    print("2. Enter file path manually")
    print("3. Enter URL to CSV file")
    choice = (input("Choose option (1/2/3) and press Enter to browse: ").strip() or "1")

    if choice == "1":
        # GUI browse (falls back to manual if GUI not available)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Select CSV Data File",
                filetypes=[("CSV files", ".csv"), ("All files", ".*")]
            )
            root.destroy()
            return ("file", path) if path else ("file", "")
        except Exception as e:
            print(f"GUI not available ({e}).")
            manual = input("Enter CSV file path manually: ").strip()
            return ("file", manual)

    if choice == "2":
        manual = input("Enter CSV file path: ").strip()
        return ("file", manual)

    if choice == "3":
        url = input("Enter URL to CSV file: ").strip()
        return ("url", url)

    # default again to browse if invalid
    return ("file", "")


def _materialize_from_url(url: str) -> str:
    """Download URL to a temp .csv file and return its path."""
    try:
        import requests
    except ImportError:
        print("✗ The 'requests' package is required for URL mode. Install via: pip install requests")
        sys.exit(1)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # Some endpoints return text/csv; others may gzip—pandas can handle raw text here.
    suffix = ".csv"
    fd, tmp_path = tempfile.mkstemp(prefix="analytics_", suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(resp.content)
    print(f"✓ Downloaded to temp file: {tmp_path}")
    return tmp_path


def _load_dataframe(source_kind: str, source_value: str) -> pd.DataFrame:
    if source_kind == "file":
        if not source_value or not os.path.exists(source_value):
            print("✗ No valid file selected/found.")
            sys.exit(1)
        print(f"Loading CSV from file: {source_value}")
        return _read_csv_robust(source_value)

    if source_kind == "url":
        if not source_value:
            print("✗ No URL provided.")
            sys.exit(1)
        print(f"⇣ Fetching CSV from URL: {source_value}")
        tmp = _materialize_from_url(source_value)
        return _read_csv_robust(tmp)

    print("✗ Unknown source type.")
    sys.exit(1)


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(0)
    return df


def run_predictive(csv_path: str):
    """Run all predictive models and comparison."""
    base_dir = Path(__file__).parent.parent  # backend/src/analytics
    predictive_dir = base_dir / "predictive"
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
        script_path = predictive_dir / model
        if script_path.exists():
            print(f"Running Predictive Model: {model}...")
            result = subprocess.run([sys.executable, str(script_path), csv_path], cwd=predictive_dir, capture_output=False, text=True)
            if result.returncode == 0:
                print(f"✓ Successfully ran {model}")
            else:
                print(f"✗ Failed to run {model} (exit code: {result.returncode})")
                success = False
        else:
            print(f"✗ Predictive script {model} not found")
            success = False

    # Run compare_models.py
    compare_script = predictive_dir / "compare_models.py"
    if compare_script.exists():
        print("Running Model Comparison...")
        result = subprocess.run([sys.executable, str(compare_script)], cwd=predictive_dir, capture_output=False, text=True)
        if result.returncode == 0:
            print("✓ Successfully ran model comparison")
        else:
            print(f"✗ Failed to run model comparison (exit code: {result.returncode})")
            success = False
    else:
        print("✗ Compare script not found")
        success = False

    return success


def run_prescriptive(csv_path: str):
    """Run prescriptive analytics."""
    base_dir = Path(__file__).parent.parent  # backend/src/analytics
    prescriptive_dir = base_dir / "prescriptive"
    script_path = prescriptive_dir / "prescriptive.py"
    if script_path.exists():
        print("Running Prescriptive Analytics...")
        result = subprocess.run([sys.executable, str(script_path), csv_path], cwd=prescriptive_dir, capture_output=False, text=True)
        if result.returncode == 0:
            print("✓ Successfully ran prescriptive analytics")
            return True
        else:
            print(f"✗ Failed to run prescriptive analytics (exit code: {result.returncode})")
            return False
    else:
        print("✗ Prescriptive script not found")
        return False


def main():
    # --- Anchor everything to the analytics directory ---
    script_dir = Path(__file__).parent  # descriptive
    analytics_dir = script_dir.parent  # analytics
    os.chdir(analytics_dir)  # change to analytics root

    # Output directories
    desc_out_dir = analytics_dir / "descriptive_output"
    pred_out_dir = analytics_dir / "predictive" / "out"
    presc_out_dir = analytics_dir / "prescriptive_output"

    desc_out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    presc_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analytics directory: {analytics_dir}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Descriptive output: {desc_out_dir}")
    print(f"Predictive output: {pred_out_dir}")
    print(f"Prescriptive output: {presc_out_dir}")

    # --- Check for command line argument ---
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.exists(csv_path):
            print(f"Loading CSV from argument: {csv_path}")
            df = _read_csv_robust(csv_path)
            csv_path_str = csv_path
        else:
            print(f"✗ File not found: {csv_path}")
            sys.exit(1)
    else:
        # --- Data source selection and load ---
        source_kind, source_value = _browse_or_path_or_url()
        df = _load_dataframe(source_kind, source_value)
        csv_path_str = source_value if source_kind == "file" else _materialize_from_url(source_value)

    df = _basic_clean(df)

    # --- Run Descriptive Analytics ---
    print("\n" + "=" * 70)
    print("DESCRIPTIVE ANALYTICS")
    print("=" * 70)
    print("\n=== CLUSTERING (auto) ===")
    run_clustering_pipeline(df, output_dir=str(desc_out_dir))

    print("\n=== MARKET BASKET ANALYSIS (auto) ===")
    run_mba_pipeline(df, output_dir=str(desc_out_dir))

    print("✓ Descriptive analytics complete.")

    # --- Run Predictive Analytics ---
    print("\n" + "=" * 70)
    print("PREDICTIVE ANALYTICS")
    print("=" * 70)
    pred_success = run_predictive(csv_path_str)

    # --- Run Prescriptive Analytics ---
    print("\n" + "=" * 70)
    print("PRESCRIPTIVE ANALYTICS")
    print("=" * 70)
    presc_success = run_prescriptive(csv_path_str)

    print("\n" + "=" * 70)
    if pred_success and presc_success:
        print("✓ All analytics models completed successfully!")
        print("Outputs are ready for PowerBI visualization:")
        print(f"- Descriptive: {desc_out_dir}/")
        print(f"- Predictive: {pred_out_dir}/")
        print(f"- Prescriptive: {presc_out_dir}/")
    else:
        print("✗ Some models failed. Check outputs and logs above.")


if __name__ == "__main__":
    main()
