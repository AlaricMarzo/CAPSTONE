# descriptive.py
import os, sys, tempfile
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
    print("DESCRIPTIVE ANALYTICS — DATA IMPORT")
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
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
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
        print(f"✓ Loading CSV from file: {source_value}")
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


def main():
    # --- Anchor everything to the script directory to avoid duplicate folders ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # make relative paths in other modules resolve here

    out_dir = os.path.join(script_dir, "descriptive_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Script directory: {script_dir}")
    print(f"Current working directory (anchored): {os.getcwd()}")
    print(f"Output directory: {out_dir}")

    # --- Data source selection and load ---
    source_kind, source_value = _browse_or_path_or_url()
    df = _load_dataframe(source_kind, source_value)
    df = _basic_clean(df)

    # --- Run BOTH pipelines with the same absolute output_dir ---
    print("\n=== CLUSTERING (auto) ===")
    run_clustering_pipeline(df, output_dir=out_dir)

    print("\n=== MARKET BASKET ANALYSIS (auto) ===")
    run_mba_pipeline(df, output_dir=out_dir)

    print("\n✓ All done. Check the 'descriptive_output' folder for CSVs and charts.")


if __name__ == "__main__":
    main()
