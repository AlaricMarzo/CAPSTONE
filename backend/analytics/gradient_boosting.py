# backend/analytics/xgboost_model.py
# XGBoost forecasting per SKU (80/20 split) — same schema as Random Forest

import warnings
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_STEPS = 6
TOP_N = 5

def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", str(s).strip().lower().replace(" ","_")) or "name"

def parse_dates_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred)/denom))*100

def add_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    for i in range(1, 4):
        df[f"Lag_{i}"] = df["Qty"].shift(i)
    return df.dropna()

def run_file(path: Path):
    print(f"\n📄 {path.name}")
    df = pd.read_csv(path)
    df["Date"] = parse_dates_safe(df["Date"])
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    df["Item Code"] = df["Item Code"].astype(str)
    df["Description"] = df["Description"].astype(str)

    monthly = df.groupby(["Item Code", pd.Grouper(key="Date", freq="MS")])["Qty"].sum().reset_index()
    top_skus = monthly.groupby("Item Code")["Qty"].sum().sort_values(ascending=False).head(TOP_N).index
    out_dir = OUT_DIR / clean_name(path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for sku in top_skus:
        series = monthly[monthly["Item Code"] == sku].copy()
        if len(series) < 8: continue
        series = add_features(series)
        if len(series) < 8: continue

        X = series[["Month","Quarter","Year","Lag_1","Lag_2","Lag_3"]]
        y = series["Qty"]
        split = int(len(X)*0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        dates_test = series["Date"].iloc[split:]

        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mp  = mape(y_test, y_pred)

        # Forecast
        last = series.iloc[-1]
        lag1, lag2, lag3 = last["Qty"], last["Lag_1"], last["Lag_2"]
        fc_dates = pd.date_range(series["Date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS")
        fc_vals = []
        for d in fc_dates:
            row = pd.DataFrame([[d.month, d.quarter, d.year, lag1, lag2, lag3]],
                               columns=["Month","Quarter","Year","Lag_1","Lag_2","Lag_3"])
            pred = model.predict(row)[0]
            fc_vals.append(pred)
            lag1, lag2, lag3 = pred, lag1, lag2

        plt.figure(figsize=(10,5))
        plt.plot(series["Date"], series["Qty"], label="Historical")
        plt.plot(dates_test, y_pred, "r--", label="Test Pred")
        plt.plot(fc_dates, fc_vals, "g-", label="XGBoost Forecast")
        plt.title(f"XGBoost Forecast — {sku}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"xgb_{clean_name(sku)}.png", dpi=150)
        plt.close()

        row = {"sku": sku, "mae": mae, "mse": mse, "mape": mp}
        for i,v in enumerate(fc_vals,1):
            row[f"xgb_t+{i}"] = v
        results.append(row)

        print(f"   ✅ {sku}: MAE={mae:.2f}, MAPE={mp:.2f}%")

    pd.DataFrame(results).to_csv(out_dir/"gb_results.csv", index=False)
    print(f"✅ Saved → {out_dir/'et_results.csv'}")

def main():
    for f in DATA_DIR.glob("*.csv"):
        run_file(f)

if __name__ == "__main__":
    main()
