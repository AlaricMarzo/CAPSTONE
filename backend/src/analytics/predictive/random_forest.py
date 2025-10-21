import warnings
from pathlib import Path
import re
import sys
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
HERE      = Path(__file__).resolve().parent
DATA_PATH = HERE / "cleaned" / "complete_data.csv"   # change if needed
OUT_DIR   = HERE / "out" / "rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_STEPS = 6
TOP_N          = 5   # top products by total Qty to run

# Accepted aliases for auto-detection (lowercased)
ALIASES = {
    "Date": [
        "date", "transaction date", "trans date", "receipt date", "sales date",
        "order date", "invoice date", "posting date"
    ],
    "Item Code": [
        "item code", "item_code", "itemcode", "sku", "sku id", "sku_id", "barcode",
        "product code", "product_code", "productcode", "upc", "ean", "code", "id", "item id", "itemid"
    ],
    "Description": ["description", "desc", "item name", "product", "product name", "name", "item"],
    "Qty": ["qty", "quantity", "qty sold", "sold qty", "quantity sold", "sales qty", "units", "units sold", "qnt"],
}

# -------------- helpers --------------
def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", s.strip().lower().replace(" ", "_")) or "name"

def parse_dates_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    """
    Return mapping: original_name -> standardized_name: 'Date','Item Code','Description','Qty'
    Falls back to using Description as key if Item Code is absent.
    """
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}

    def pick(std: str) -> Optional[str]:
        # exact
        for k, orig in lower_map.items():
            if k == std.lower():
                return orig
        # aliases
        for alias in ALIASES.get(std, []):
            if alias in lower_map:
                return lower_map[alias]
        # for Date: first datetime-like column if nothing matched
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
        raise ValueError("Could not detect a Date column. Add/rename a date-like column.")
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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    return df

def add_lags(df: pd.DataFrame, n_lags: int = 3, target_col: str = "Qty") -> pd.DataFrame:
    for i in range(1, n_lags+1):
        df[f"Lag_{i}"] = df[target_col].shift(i)
    return df

def monthly_aggregate(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    monthly = (
        df.groupby([key_col, pd.Grouper(key="Date", freq="MS")])["Qty"]
          .sum()
          .reset_index()
          .sort_values(["{0}".format(key_col), "Date"])
    )
    return monthly

# -------------- main --------------
def main():
    if len(sys.argv) > 1:
        # Use provided CSV path
        csv_path = Path(sys.argv[1])
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}")
            return
        data_path = csv_path
        print(f"Using provided file: {csv_path}")
    else:
        print("❌ No file provided. Please provide CSV path as argument.")
        return

    # robust CSV read
    if data_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)

    # detect & normalize columns
    rename = detect_columns(df)
    print("Detected columns mapping:", rename)
    df = df.rename(columns=rename)

    # basic cleaning
    df["Date"] = parse_dates_safe(df["Date"])
    df = df.dropna(subset=["Date"])
    df["Description"] = df["Description"].astype(str).fillna("Unknown Product")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)

    # if no Item Code, synthesize one from Description
    if "Item Code" not in df.columns:
        df["Item Code"] = df["Description"].astype(str)

    key_col = "Item Code"  # keep codes as keys; descriptions for labels
    sku_desc = (
        df[[key_col, "Description"]]
        .dropna(subset=["Description"])
        .drop_duplicates(subset=[key_col])
        .set_index(key_col)["Description"]
        .to_dict()
    )

    monthly = monthly_aggregate(df, key_col)
    if monthly.empty:
        print("No monthly data after cleaning; stopping.")
        return

    # choose top-N by total Qty
    top_keys = (
        monthly.groupby(key_col)["Qty"].sum()
        .sort_values(ascending=False).head(TOP_N).index.tolist()
    )
    print(f"✅ Top products: {top_keys}")

    summary_rows: List[List] = []

    for sku in top_keys:
        series = monthly[monthly[key_col] == sku].copy()
        if series["Date"].nunique() < 8:  # need some history
            print(f"↪ {sku}: too few months ({series['Date'].nunique()}); skipping.")
            continue

        desc = sku_desc.get(sku, "Unknown Product")
        label = f"{sku} — {desc}"

        # features
        series = engineer_features(series)
        series = add_lags(series, n_lags=3, target_col="Qty")
        series = series.dropna().reset_index(drop=True)

        if len(series) < 8:
            print(f"↪ {sku}: too few rows after lags; skipping.")
            continue

        X = series[["Month", "Quarter", "Year", "Lag_1", "Lag_2", "Lag_3"]]
        y = series["Qty"]

        # Train / Test split (last 3 months as test if enough data)
        test_size = 3 if len(series) >= 12 else max(1, len(series)//5)
        split_idx = len(series) - test_size
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        dates_all = series["Date"]
        dates_test = dates_all.iloc[split_idx:]

        # model
        model = RandomForestRegressor(
            n_estimators=400, random_state=42, max_depth=None, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        mp  = float(mape(y_test, y_pred))

        # iterative forecast next 6 months
        last = series.iloc[-1].copy()
        fc_vals = []
        fc_dates = pd.date_range(series["Date"].iloc[-1] + pd.offsets.MonthBegin(1),
                                 periods=FORECAST_STEPS, freq="MS")
        lag1, lag2, lag3 = last["Qty"], last.get("Lag_1", 0), last.get("Lag_2", 0)

        for d in fc_dates:
            row = pd.DataFrame([{
                "Month": d.month, "Quarter": d.quarter, "Year": d.year,
                "Lag_1": lag1, "Lag_2": lag2, "Lag_3": lag3
            }])
            pred = float(model.predict(row)[0])
            fc_vals.append(pred)
            # update lags for next step
            lag1, lag2, lag3 = pred, lag1, lag2

        # ----- plot -----
        plt.figure(figsize=(10,5))
        plt.plot(series["Date"], y, label="Historical", linewidth=2)
        plt.plot(dates_test, y_pred, "r--", label="Test Pred", linewidth=1.8)
        plt.plot(fc_dates, fc_vals, "g-", label="RF Forecast (t+1..t+6)", linewidth=2)
        plt.title(f"Random Forest Forecast — {label}")
        plt.xlabel("Month"); plt.ylabel("Qty")
        plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
        out_img = OUT_DIR / f"rf_{clean_name(sku)}.png"
        plt.savefig(out_img, dpi=150); plt.close()

        print(f"📊 {sku}: MAE={mae:.2f}  MSE={mse:.2f}  MAPE={mp:.2f}%  → {out_img}")

        summary_rows.append([
            sku, desc, len(series), mae, mse, mp,
            *[round(v,2) for v in fc_vals]
        ])

    # save summary
    if summary_rows:
        cols = ["SKU","Description","HistoryPts","MAE","MSE","MAPE%"] + [f"t+{i+1}" for i in range(FORECAST_STEPS)]
        pd.DataFrame(summary_rows, columns=cols).to_csv(OUT_DIR / "rf_results_summary.csv", index=False)
        print(f"\n✅ Saved summary → {OUT_DIR / 'rf_results_summary.csv'}")
    else:
        print("No products produced results. Check column detection or data length.")

if __name__ == "__main__":
    main()
