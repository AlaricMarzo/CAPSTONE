# backend/analytics/compare_models.py
# Combines ARIMA, ETS, Random Forest, XGBoost, ExtraTrees, Gradient Boosting, and LSTM results.
# Selects the best-performing model per SKU (based on MAPE, then MAE, then MSE).

import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"

# Expected file patterns (each model's results)
MODEL_FILES = {
    "ARIMA_ETS": "_ARIMA_ETS_MASTER.csv",
    "RF": "_RF_MASTER.csv",
    "XGB": "xgb_results.csv",
    "ET": "et_results.csv",
    "GB": "gb_results.csv",
    "LSTM": "lstm_results.csv",
}

def load_model_csv(path: Path, model_name: str) -> pd.DataFrame:
    """Load a model result CSV and standardize its column names."""
    if not path.exists():
        print(f"⚠️ Skipping {model_name} — file not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalize structure
    rename_map = {}
    for c in df.columns:
        if "mae" in c: rename_map[c] = "mae"
        elif "mse" in c: rename_map[c] = "mse"
        elif "mape" in c: rename_map[c] = "mape"
        elif c in ["sku", "item code"]: rename_map[c] = "sku"
        elif "desc" in c: rename_map[c] = "description"

    df = df.rename(columns=rename_map)
    df["model"] = model_name
    df = df[["sku", "description", "model", "mae", "mse", "mape"]].dropna(subset=["sku"])
    return df

def load_all_models() -> pd.DataFrame:
    """Combine all model CSVs into one dataframe."""
    all_dfs = []
    for model_name, filename in MODEL_FILES.items():
        # Search both root and subdirectories
        found = list(OUT_DIR.rglob(filename))
        if found:
            df = load_model_csv(found[0], model_name)
            all_dfs.append(df)
        else:
            print(f"⚠️ No results found for {model_name}")

    if not all_dfs:
        print("❌ No model results found. Run model scripts first.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Loaded results from {len(all_dfs)} model(s). Total rows: {len(combined)}")
    return combined

def pick_best_model(df: pd.DataFrame) -> pd.DataFrame:
    """Pick best model per SKU using lowest MAPE, then MAE, then MSE."""
    df = df.copy()
    df["rank"] = df.groupby("sku")[["mape", "mae", "mse"]].apply(
        lambda x: np.lexsort((x["mse"], x["mae"], x["mape"]))
    ).explode().astype(int)
    df["rank"] = df.groupby("sku")["mape"].rank(method="min", ascending=True)

    winners = df.loc[df.groupby("sku")["mape"].idxmin()].copy()
    winners = winners.sort_values("mape")
    return winners

def main():
    combined = load_all_models()
    if combined.empty:
        return

    best = pick_best_model(combined)
    summary_path = OUT_DIR / "_MASTER_COMPARE_ALL_MODELS.csv"
    best.to_csv(summary_path, index=False)

    print("\n🎉 Model comparison complete!")
    print(f"✅ Saved → {summary_path}")
    print("\n🏆 Best model summary:")
    print(best["model"].value_counts())

    # Optional: export pivot summary for quick analysis
    pivot = combined.pivot_table(
        index="model",
        values=["mae", "mse", "mape"],
        aggfunc=["mean", "std", "min", "max"]
    ).round(2)
    pivot.to_csv(OUT_DIR / "_MODEL_METRIC_SUMMARY.csv")
    print(f"📊 Metric summary saved → {OUT_DIR / '_MODEL_METRIC_SUMMARY.csv'}")

if __name__ == "__main__":
    main()
