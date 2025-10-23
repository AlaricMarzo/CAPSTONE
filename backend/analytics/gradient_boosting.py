import warnings
from pathlib import Path
import re
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

# ---- headless backend to avoid Tkinter errors on Windows ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# -------------------------------------------------------------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_ROOT = HERE / "ml_gradient-boosting"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FORECAST_STEPS = 6
TOP_N          = 5

# -------- column aliases (lowercased) --------
ALIASES = {
    "Date": ["date","transaction date","trans date","receipt date","sales date",
             "order date","invoice date","posting date"],
    "Item Code": ["item code","item_code","itemcode","sku","sku id","sku_id","barcode",
                  "product code","product_code","productcode","upc","ean","code","id",
                  "item id","itemid"],
    "Description": ["description","desc","item name","product","product name","name","item"],
    "Qty": ["qty","quantity","qty sold","sold qty","quantity sold","sales qty","units",
            "units sold","qnt"],
}

# -------- helpers --------
def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", str(s).strip().lower().replace(" ","_")) or "name"

def parse_dates_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return mapping original_name -> standardized: Date, Item Code, Description, Qty.
    """
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename: Dict[str, str] = {}

    def pick(std: str) -> Optional[str]:
        if std.lower() in lower_map:
            return lower_map[std.lower()]
        for a in ALIASES.get(std, []):
            if a in lower_map:
                return lower_map[a]
        if std == "Date":
            for c in df.columns:
                try:
                    s = pd.to_datetime(df[c].head(50), errors="coerce")
                    if s.notna().sum() >= 10:
                        return c
                except Exception:
                    pass
        return None

    got_date = pick("Date")
    got_sku  = pick("Item Code")
    got_desc = pick("Description")
    got_qty  = pick("Qty")

    if not got_date:
        raise ValueError("Could not detect a Date column.")
    if not got_desc:
        raise ValueError("Could not detect a Description column.")
    if not got_qty:
        raise ValueError("Could not detect a Qty/Quantity column.")

    rename[got_date] = "Date"
    rename[got_desc] = "Description"
    rename[got_qty]  = "Qty"
    if got_sku:
        rename[got_sku] = "Item Code"
    return rename

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    for i in range(1, 4):
        df[f"Lag_{i}"] = df["Qty"].shift(i)
    return df.dropna()

# -------- core --------
def run_file(path: Path):
    print(f"\n📄 {path.name}")

    # robust read: CSV/XLS(X)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # autodetect + normalize columns
    rename = detect_columns(df)
    print("Detected columns:", rename)
    df = df.rename(columns=rename)

    # basic cleaning
    df["Date"] = parse_dates_safe(df["Date"])
    df = df.dropna(subset=["Date"])
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    df["Description"] = df["Description"].astype(str).fillna("Unknown Product")
    if "Item Code" not in df.columns:
        df["Item Code"] = df["Description"].astype(str)

    # map SKU -> Description
    sku_desc = (
        df[["Item Code", "Description"]]
        .dropna(subset=["Description"])
        .drop_duplicates(subset=["Item Code"])
        .set_index("Item Code")["Description"]
        .to_dict()
    )

    # monthly aggregate per SKU
    monthly = (
        df.groupby(["Item Code", pd.Grouper(key="Date", freq="MS")])["Qty"]
          .sum().reset_index().sort_values(["Item Code","Date"])
    )
    if monthly.empty:
        print("   (no rows after monthly aggregation)")
        return

    # select top movers
    top_skus = (
        monthly.groupby("Item Code")["Qty"].sum()
        .sort_values(ascending=False).head(TOP_N).index.tolist()
    )

    out_dir = OUT_ROOT / clean_name(path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    for sku in top_skus:
        series = monthly[monthly["Item Code"] == sku].copy()
        if series["Date"].nunique() < 8:
            print(f"   ↪ {sku}: too few months, skipped.")
            continue

        series = add_features(series)
        if len(series) < 8:
            print(f"   ↪ {sku}: too few rows after lags, skipped.")
            continue

        desc  = sku_desc.get(sku, "Unknown Product")
        label = f"{sku} — {desc}"

        X = series[["Month","Quarter","Year","Lag_1","Lag_2","Lag_3"]]
        y = series["Qty"]

        # smart chronological split
        if len(series) < 15:
            test_size = 3
            split = len(series) - test_size
        else:
            split = int(len(series) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        dates_test = series["Date"].iloc[split:]

        # model: Gradient Boosting
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # metrics
        mae  = float(mean_absolute_error(y_test, y_pred))
        mse  = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mp   = float(mape(y_test, y_pred))

        # roll-forward 6-month forecast
        last = series.iloc[-1]
        lag1, lag2, lag3 = last["Qty"], last["Lag_1"], last["Lag_2"]
        fc_dates = pd.date_range(series["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                                 periods=FORECAST_STEPS, freq="MS")
        fc_vals: List[float] = []
        for d in fc_dates:
            row = pd.DataFrame([{
                "Month": d.month, "Quarter": d.quarter, "Year": d.year,
                "Lag_1": lag1, "Lag_2": lag2, "Lag_3": lag3
            }])
            pred = float(model.predict(row)[0])
            fc_vals.append(pred)
            lag1, lag2, lag3 = pred, lag1, lag2  # shift lags

        # plot (with bottom metrics footer)
        plt.figure(figsize=(10,5))
        plt.plot(series["Date"], series["Qty"], label="Historical", linewidth=2)
        if len(y_pred) > 0:
            plt.plot(dates_test, y_pred, "r--", label="Test Pred", linewidth=1.8)
        plt.plot(fc_dates, fc_vals, "g-", label="Gradient Boosting Forecast (t+1..t+6)", linewidth=2)
        plt.title(f"Gradient Boosting Forecast — {label}")
        plt.xlabel("Month"); plt.ylabel("Qty")
        plt.legend(); plt.grid(alpha=0.25)

        # ---- bottom metric text block (consistent with other models) ----
        footer = f"Gradient Boosting → MAE {mae:.2f}, MSE {mse:.2f}, RMSE {rmse:.2f}, MAPE {mp:.2f}%"
        plt.gcf().text(0.5, -0.06, footer, ha="center", va="top", fontsize=9, linespacing=1.4)

        plt.tight_layout()
        out_img = out_dir / f"gb_{clean_name(sku)}.png"
        plt.savefig(out_img, dpi=150, bbox_inches="tight")
        plt.close()

        rows.append({
            "dataset": path.name,
            "sku": sku,
            "description": desc,
            "gb_rmse": rmse,
            "gb_mae": mae,
            "gb_mse": mse,
            "gb_mape": mp,
            **{f"gb_t+{i}": float(round(v, 2)) for i, v in enumerate(fc_vals, 1)}
        })

        print(f"   ✅ {label}: RMSE={rmse:.2f}  MAE={mae:.2f}  MSE={mse:.2f}  MAPE={mp:.2f}%  → {out_img}")

    # save results
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "gb_results.csv", index=False)
        print(f"✅ Saved → {out_dir/'gb_results.csv'}")
    else:
        print("No results produced for this dataset.")

def main():
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        print(f"❌ No files found in {DATA_DIR}")
        return
    for f in files:
        run_file(f)

if __name__ == "__main__":
    main()
