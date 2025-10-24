
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# One-Run Predictive Comparator (Console-first)
# Loads all model result CSVs, normalizes columns, merges them.
# Picks best model per SKU using MAPE, then MAE, then RMSE/MSE tie-breakers.
# Prints clear console summaries for VS Code. Optionally saves CSVs if --save.

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------- Defaults ----------------------
DEFAULT_INPUTS = [
    "sarima_ets_hw_results.csv",
    "sarima_ets_hw_90_10_results.csv",
    "sarima_ets_random_results.csv",
    "sarima_ets_holts_random_test.csv",
    "xgb_results.csv",
    "rf_results_summary.csv",
    "lstm_results.csv",
    "gb_results.csv",
    "et_results.csv",
]
DEFAULT_OUT_DIR = "out"

ALIASES = {
    "sku":     ["sku","item_code","item code","item_id","product_code","code","id","barcode","upc","ean"],
    "model":   ["model","algo","algorithm","method"],
    "horizon": ["h","horizon","steps","forecast_steps","n_forecast","periods","forecast_horizon"],
    "mape":    ["mape","mean_absolute_percentage_error","mean_ape"],
    "smape":   ["smape","symmetric_mape","s_mape"],
    "mae":     ["mae","mean_absolute_error"],
    "rmse":    ["rmse","root_mean_squared_error"],
    "mse":     ["mse","mean_squared_error"],
    "rmsle":   ["rmsle"],
    "mase":    ["mase"],
    "wmape":   ["wmape","w_mape","weighted_mape"],
    "mdape":   ["mdape","median_ape","median_absolute_percentage_error"],
    "r2":      ["r2","r_squared","r2_score"],
}

CANON = ["sku","model","horizon","mape","smape","mae","rmse","mse","rmsle","mase","wmape","mdape","r2","source"]

# ---------------------- Helpers ----------------------
def find_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for k in keys:
        if k in cols:
            return cols[k]
    return None

def standardize_df(raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = raw.copy()
    lower = {c: c.lower().strip() for c in df.columns}
    df.rename(columns={c: lower[c] for c in df.columns}, inplace=True)

    # sku
    sku_col = find_col(df, ALIASES["sku"])
    if sku_col is None:
        df["sku"] = "ALL"
    elif sku_col != "sku":
        df.rename(columns={sku_col:"sku"}, inplace=True)

    # model
    model_col = find_col(df, ALIASES["model"])
    if model_col is None:
        df["model"] = re.sub(r"[^a-z0-9_\-]+","", source_name.lower())
    elif model_col != "model":
        df.rename(columns={model_col:"model"}, inplace=True)

    # horizon
    hz_col = find_col(df, ALIASES["horizon"])
    if hz_col is None:
        df["horizon"] = np.nan
    elif hz_col != "horizon":
        df.rename(columns={hz_col:"horizon"}, inplace=True)

    # metrics
    for logical, aliases in ALIASES.items():
        if logical in ["sku","model","horizon"]:
            continue
        found = find_col(df, aliases)
        if found and found != logical and logical not in df.columns:
            df.rename(columns={found: logical}, inplace=True)

    # to numeric
    for m in ["mape","smape","mae","rmse","mse","rmsle","mase","wmape","mdape","r2"]:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    out = pd.DataFrame()
    for c in ["sku","model","horizon"]:
        out[c] = df[c] if c in df.columns else np.nan
    for m in ["mape","smape","mae","rmse","mse","rmsle","mase","wmape","mdape","r2"]:
        out[m] = df[m] if m in df.columns else np.nan
    out["source"] = source_name
    return out

def load_one_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] Failed to read {path.name}: {e}")
            return None
    return standardize_df(df, path.stem)

def discover_csvs(inputs: List[str]) -> List[Path]:
    results = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            results.extend(list(p.rglob("*.csv")))
        else:
            if p.exists():
                results.append(p)
            else:
                here = Path(__file__).resolve().parent
                cand = here / p.name
                if cand.exists():
                    results.append(cand)
                else:
                    cand2 = Path.cwd() / p.name
                    if cand2.exists():
                        results.append(cand2)
                    else:
                        print(f"[WARN] Not found: {inp}")
    uniq, seen = [], set()
    for r in results:
        rp = str(r.resolve())
        if rp not in seen:
            uniq.append(r)
            seen.add(rp)
    return uniq

def load_all_models(paths: List[Path]) -> pd.DataFrame:
    all_dfs: List[pd.DataFrame] = []
    for p in paths:
        df = load_one_csv(p)
        if df is None or df.empty:
            print(f"[WARN] Skipping empty or unreadable: {p.name}")
            continue
        all_dfs.append(df)

    if not all_dfs:
        print("❌ No model results found. Run model scripts first.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Loaded results from {len(all_dfs)} file(s). Total rows: {len(combined)}")
    return combined

def pick_best_model(df: pd.DataFrame) -> pd.DataFrame:
    # Determine metric order based on availability
    order = []
    if "mape" in df.columns and df["mape"].notna().any(): order.append("mape")
    if "mae"  in df.columns and df["mae"].notna().any():   order.append("mae")
    rmse_like = None
    if "rmse" in df.columns and df["rmse"].notna().any():
        rmse_like = "rmse"
    elif "mse" in df.columns and df["mse"].notna().any():
        rmse_like = "mse"
    if rmse_like: order.append(rmse_like)

    if not order:
        # no metrics, return first per sku
        winners = df.sort_values(["sku"]).groupby("sku", as_index=False).first()
        return winners

    # sort by sku + metrics ascending, pick first per sku
    winners = (df.sort_values(["sku"] + order, ascending=[True] + [True]*len(order))
                 .groupby("sku", as_index=False)
                 .first())

    # reorder columns for readability
    front = ["sku","model","horizon"] + order
    keep = [c for c in front if c in winners.columns] + [c for c in winners.columns if c not in front]
    return winners[keep]

def print_summary(combined: pd.DataFrame, best: pd.DataFrame, top: int = 10) -> None:
    models = combined["model"].astype(str).str.upper().unique()
    print("\n🧩 Models detected:", ", ".join(sorted(models)))

    metrics = [m for m in ["mape","mae","rmse","mse","smape","wmape","r2"] if m in combined.columns and combined[m].notna().any()]
    print("📐 Metrics available:", ", ".join(metrics) if metrics else "None")

    if metrics:
        print("\n🌐 Global mean metrics by model:")
        g = combined.groupby("model", as_index=False)[metrics].mean()
        # Sort by primary if present, else fallback
        sort_keys = [k for k in ["mape","mae","rmse","mse"] if k in metrics]
        g = g.sort_values(by=sort_keys if sort_keys else metrics, ascending=True)
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print(g.round(4))

    if not best.empty:
        print("\n🏆 Best model per SKU (sample):")
        show_cols = [c for c in ["sku","model","horizon","mape","mae","rmse","mse"] if c in best.columns]
        with pd.option_context("display.max_rows", 20, "display.width", 140):
            print(best[show_cols].head(top).to_string(index=False))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compile and compare all model outputs, console-first.")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS, help="CSV files or folders to scan recursively.")
    parser.add_argument("--top", type=int, default=10, help="How many rows to print in summaries.")
    parser.add_argument("--save", action="store_true", help="Also save compiled and summary CSVs.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output folder if --save is used.")
    args = parser.parse_args(argv)

    csvs = discover_csvs(args.inputs)
    combined = load_all_models(csvs)
    if combined.empty:
        sys.exit(0)

    best = pick_best_model(combined)
    print_summary(combined, best, top=args.top)

    if args.save:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out / "compiled_all_results.csv", index=False)
        best.to_csv(out / "best_model_per_sku.csv", index=False)
        metrics = [m for m in ["mape","mae","rmse","mse","smape","wmape","r2"] if m in combined.columns]
        if metrics:
            pivot = combined.groupby("model")[metrics].agg(["mean","std","min","max"]).round(4)
            pivot.to_csv(out / "model_metric_summary.csv")
        print(f"\n💾 Saved CSVs in -> {out.resolve()}")

if __name__ == "__main__":
    main()
