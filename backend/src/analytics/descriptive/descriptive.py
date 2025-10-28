# analytics/descriptive/descriptive.py
import os, sys, json, tempfile
from pathlib import Path
import pandas as pd

from kpi import compute_kpis
from mba import run_mba
from clustering import cluster_all

def _read_csv_robust(path_or_buf):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path_or_buf, encoding=enc, engine="python")
        except Exception:
            continue
    return pd.read_csv(path_or_buf)

def _browse_or_path_or_url():
    print("=" * 70)
    print("DESCRIPTIVE ANALYTICS — DATA IMPORT")
    print("=" * 70)
    print("1. Browse for CSV file")
    print("2. Enter file path manually")
    print("3. Enter URL to CSV file")
    choice = (input("Choose option (1/2/3): ").strip() or "1")

    if choice == "1":
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

    return ("file", "")

def _materialize_from_url(url: str) -> str:
    try:
        import requests
    except ImportError:
        print("✗ Install 'requests' for URL mode: pip install requests")
        sys.exit(1)
    resp = requests.get(url, timeout=60); resp.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(prefix="analytics_", suffix=".csv")
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(resp.content)
    print(f"✓ Downloaded to temp file: {tmp_path}")
    return tmp_path

def _load_dataframe(source_kind: str, source_value: str) -> pd.DataFrame:
    if source_kind == "file":
        if not source_value or not os.path.exists(source_value):
            print("✗ No valid file selected/found."); sys.exit(1)
        print(f"✓ Loading CSV from file: {source_value}")
        return _read_csv_robust(source_value)
    if source_kind == "url":
        if not source_value:
            print("✗ No URL provided."); sys.exit(1)
        print(f"⇣ Fetching CSV from URL: {source_value}")
        tmp = _materialize_from_url(source_value)
        return _read_csv_robust(tmp)
    print("✗ Unknown source type."); sys.exit(1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    out_dir = os.path.join(script_dir, "descriptive_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Script directory: {script_dir}")
    print(f"Output directory: {out_dir}")

    source_kind, source_value = _browse_or_path_or_url()
    df = _load_dataframe(source_kind, source_value)

    kpi_out = os.path.join(out_dir, "kpi_output")
    mba_out = os.path.join(out_dir, "mba_output")
    clu_out = os.path.join(out_dir, "clustering_output")
    os.makedirs(kpi_out, exist_ok=True)
    os.makedirs(mba_out, exist_ok=True)
    os.makedirs(clu_out, exist_ok=True)

    print("\n[1/3] Running KPIs ...")
    kpi_res = compute_kpis(df, kpi_out)
    print("✓ KPIs done.")
    print(f"   Avg monthly sales growth: {kpi_res.get('avg_monthly_growth_rate')}")

    print("\n[2/3] Running Market-Basket (MBA) ...")
    mba_res = run_mba(df, mba_out)
    print(f"✓ MBA done. Rules generated: {mba_res.get('rules_count')}")

    print("\n[3/3] Running Clustering (self-setting K) ...")
    clu_res = cluster_all(
        df, clu_out,
        by_tab=True,
        by_category=False,        # flip to True to also generate by-category clusters
        use_sampling_for_silhouette=False,  # FULL DATA
        label_points=False,
        max_point_labels=0,
        make_zoomed_variant=True,
        make_per_cluster_panels=True
    )
    print(f"✓ Clustering done. Global k={clu_res['global']['k']} (silhouette={clu_res['global']['silhouette']:.3f})")

    manifest = {
        "kpi_outputs": [str(p) for p in Path(kpi_out).glob("*.*")],
        "mba_outputs": [str(p) for p in Path(mba_out).glob("*.*")],
        "clustering_outputs": [str(p) for p in Path(clu_out).rglob("*.*")],
    }
    Path(out_dir, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nAll done! Outputs under:", out_dir)

if __name__ == "__main__":
    main()
