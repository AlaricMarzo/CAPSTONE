#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
import re
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"                # put your CSV/XLSX here
OUT_ROOT = HERE / "ts_sarima_ets_holts(80,20)"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEASONAL_PERIODS  = 12     # monthly seasonality
FORECAST_STEPS    = 6      # 6-month horizon
TOP_N_PER_FILE    = 5      # how many products to chart
MIN_SERIES_LENGTH = 12     # skip very short series (set 6 if needed)
TRAIN_FRACTION    = 0.80   # 80/20 split

# SARIMA orders (thesis-style)
SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 1, SEASONAL_PERIODS)

# ---------------- Helpers ----------------
def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", str(s).strip().lower().replace(" ", "_")) or "name"

def parse_dates_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower_map = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "Date": ["date","transaction date","receipt date","sales date","order date","invoice date","trans date"],
        "Description": ["description","item name","product","product name","name","item","desc"],
        "Qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty","sold qty","sale qty","qnt"],
    }
    def pick(std: str) -> Optional[str]:
        for k, orig in lower_map.items():
            if k == std.lower(): return orig
        for alias in aliases.get(std, []):
            if alias in lower_map: return lower_map[alias]
        if std == "Date":
            for c in df.columns:
                try:
                    s = pd.to_datetime(df[c].head(50), errors="coerce")
                    if s.notna().sum() >= 10: return c
                except Exception:
                    pass
        return None
    got_date = pick("Date"); got_desc = pick("Description"); got_qty = pick("Qty")
    if not got_date: raise ValueError("Could not detect a Date column.")
    if not got_desc: raise ValueError("Could not detect a Description column.")
    if not got_qty:  raise ValueError("Could not detect a Qty/Quantity column.")
    return {got_date: "Date", got_desc: "Description", got_qty: "Qty"}

def mape_safe(y_true, y_pred) -> float:
    """MAPE that ignores zero actuals to avoid exploding percentages."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def ensure_fc_index(hist: pd.Series, fc: pd.Series) -> pd.Series:
    if not hasattr(fc.index, "freq") or fc.index.freq is None:
        fc.index = pd.date_range(hist.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=len(fc), freq="MS")
    return fc

# ---------------- Data prep ----------------
def load_monthly(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    df = df.rename(columns=detect_columns(df))
    df["Date"] = parse_dates_safe(df["Date"])
    df = df.dropna(subset=["Date"])

    # Normalize product names to avoid duplicates by trivial differences
    df["Description"] = (df["Description"].astype(str)
                         .str.strip()
                         .str.replace(r"\s+", " ", regex=True))
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)

    monthly = (
        df.set_index("Date")
          .groupby("Description")["Qty"]
          .resample("MS")
          .sum()
          .reset_index()
    )
    return monthly

def series_for(monthly: pd.DataFrame, desc: str) -> pd.Series:
    return (monthly[monthly["Description"] == desc]
            .set_index("Date")["Qty"]
            .asfreq("MS", fill_value=0)
            .astype(float))

def top_products(monthly: pd.DataFrame, n: int, mode: str = "sales") -> list:
    """
    mode:
      - 'sales'     -> top by total quantity (BEST-SELLING)
      - 'frequency' -> most months with sales > 0
      - 'recent6'   -> top by sum in last 6 months (recency)
    """
    if mode == "frequency":
        ranker = (monthly.groupby("Description")["Qty"]
                  .apply(lambda s: (s > 0).sum()))
    elif mode == "recent6":
        m = monthly.set_index("Date")
        def last6_sum(g):
            g = g.asfreq("MS", fill_value=0)
            return g.iloc[-6:].sum()
        ranker = m.groupby("Description")["Qty"].apply(last6_sum)
    else:  # 'sales'
        ranker = monthly.groupby("Description")["Qty"].sum()

    top = ranker.sort_values(ascending=False).head(n).index.tolist()
    print(f"   ▶ Top-5 by {mode}: {top}")
    return top

# ---------------- Models ----------------
def fit_sarima(train: pd.Series):
    return SARIMAX(
        train,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

def fit_ets(train: pd.Series):
    """ETS seasonal-only baseline."""
    use_season = len(train) >= 2 * SEASONAL_PERIODS
    model = ExponentialSmoothing(
        train,
        trend=None,
        seasonal="add" if use_season else None,
        seasonal_periods=SEASONAL_PERIODS if use_season else None,
        initialization_method="estimated",
    )
    return model.fit()

def fit_holt_winters(train: pd.Series):
    """Holt–Winters additive (level + trend + season)."""
    use_season = len(train) >= 2 * SEASONAL_PERIODS
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add" if use_season else None,
        seasonal_periods=SEASONAL_PERIODS if use_season else None,
        initialization_method="estimated",
    )
    return model.fit()

# ---------------- Split ----------------
def split_train_test(y: pd.Series, frac: float = TRAIN_FRACTION):
    split_idx = max(1, int(len(y) * frac))
    return y.iloc[:split_idx], y.iloc[split_idx:]

# ---------------- Plot ----------------
def build_report(
    y: pd.Series,
    sa_fc: pd.Series, sa_metrics: dict,
    ets_fc: pd.Series, ets_metrics: dict,
    hw_fc: pd.Series, hw_metrics: dict,
    title: str, out_png: Path
):
    train, test = split_train_test(y, TRAIN_FRACTION)
    sa_fc  = ensure_fc_index(y, sa_fc)
    ets_fc = ensure_fc_index(y, ets_fc)
    hw_fc  = ensure_fc_index(y, hw_fc)

    plt.figure(figsize=(10.5, 6.8))
    plt.plot(train.index, train.values, label="Training Data", linewidth=1.4)
    plt.plot(test.index,  test.values,  color="black", label="Test Data", linewidth=1.6)

    plt.plot(sa_fc.index,  sa_fc.values,  color="red",        linewidth=2, label="SARIMA Forecast")
    plt.plot(ets_fc.index, ets_fc.values, color="deepskyblue",linewidth=2, label="ETS Forecast")
    plt.plot(hw_fc.index,  hw_fc.values,  color="green",      linewidth=2, label="Holt-Winters Forecast")

    plt.title(title, fontsize=13, pad=10)
    plt.xlabel("Month"); plt.ylabel("Quantity")
    plt.grid(alpha=0.3); plt.legend()

    def fmt(m):
        if any(pd.isna(list(m.values()))):
            return "MAE n/a, MSE n/a, RMSE n/a, MAPE n/a"
        return f"MAE {m['MAE']:.2f}, MSE {m['MSE']:.2f}, RMSE {m['RMSE']:.2f}, MAPE {m['MAPE']:.2f}%"

    text = (
        f"SARIMA (1,1,1)(0,1,1,12) → {fmt(sa_metrics)}\n"
        f"ETS (seasonal-only) → {fmt(ets_metrics)}\n"
        f"Holt-Winters (additive) → {fmt(hw_metrics)}"
    )
    plt.gcf().text(0.5, -0.06, text, ha="center", va="top", fontsize=9, linespacing=1.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

# ---------------- Runner ----------------
def run_on_file(file_path: Path) -> List[dict]:
    print(f"\n📄 {file_path.name}")
    monthly = load_monthly(file_path)
    if monthly.empty:
        print("   (empty after cleaning)"); return []

    print("   Total distinct products:", monthly["Description"].nunique())
    products = top_products(monthly, TOP_N_PER_FILE, mode="sales")  # BEST-SELLING

    rows: List[dict] = []

    for desc in products:
        y = series_for(monthly, desc)
        if len(y) < MIN_SERIES_LENGTH:
            print(f"   ↪ {desc}: too few months ({len(y)}), skipped."); continue

        train, test = split_train_test(y, TRAIN_FRACTION)

        def metrics(y_true, y_pred):
            if len(y_true) == 0:
                return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
            return {
                "MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "MAPE": mape_safe(y_true, y_pred)
            }

        # SARIMA
        try:
            sa_fit = fit_sarima(train)
            sa_val = sa_fit.forecast(len(test))
            sa_metrics = metrics(test, sa_val)
            sa_fc = fit_sarima(y).forecast(FORECAST_STEPS)
        except Exception as e:
            print(f"   ⚠️ SARIMA failed for '{desc}': {e}")
            sa_fc = pd.Series([np.nan]*FORECAST_STEPS,
                              index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),
                                                  periods=FORECAST_STEPS, freq="MS"))
            sa_metrics = {"MAE":np.nan,"MSE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

        # ETS
        try:
            ets_fit = fit_ets(train)
            ets_val = ets_fit.forecast(len(test))
            ets_metrics = metrics(test, ets_val)
            ets_fc = fit_ets(y).forecast(FORECAST_STEPS)
        except Exception as e:
            print(f"   ⚠️ ETS failed for '{desc}': {e}")
            ets_fc = pd.Series([np.nan]*FORECAST_STEPS,
                               index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),
                                                   periods=FORECAST_STEPS, freq="MS"))
            ets_metrics = {"MAE":np.nan,"MSE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

        # Holt-Winters
        try:
            hw_fit = fit_holt_winters(train)
            hw_val = hw_fit.forecast(len(test))
            hw_metrics = metrics(test, hw_val)
            hw_fc = fit_holt_winters(y).forecast(FORECAST_STEPS)
        except Exception as e:
            print(f"   ⚠️ Holt-Winters failed for '{desc}': {e}")
            hw_fc = pd.Series([np.nan]*FORECAST_STEPS,
                               index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),
                                                   periods=FORECAST_STEPS, freq="MS"))
            hw_metrics = {"MAE":np.nan,"MSE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

        # flat filename, no folders
        dataset_slug  = clean_name(file_path.stem)
        product_slug  = clean_name(desc)
        out_png = OUT_ROOT / f"{dataset_slug}__{product_slug}__sarima-ets-hw_80-20.png"

        title = f"SARIMA vs. ETS vs. Holt-Winters — 80% Training, 20% Testing — {desc}"
        build_report(y, sa_fc, sa_metrics, ets_fc, ets_metrics, hw_fc, hw_metrics, title, out_png)
        print(f"   ✅ Saved → {out_png}")

        # ----- consolidated row for CSV -----
        row = {
            "dataset": file_path.name,
            "product": desc,
            # SARIMA metrics
            "sarima_mae": sa_metrics["MAE"], "sarima_mse": sa_metrics["MSE"],
            "sarima_rmse": sa_metrics["RMSE"], "sarima_mape": sa_metrics["MAPE"],
            # ETS metrics
            "ets_mae": ets_metrics["MAE"], "ets_mse": ets_metrics["MSE"],
            "ets_rmse": ets_metrics["RMSE"], "ets_mape": ets_metrics["MAPE"],
            # Holt-Winters metrics
            "hw_mae": hw_metrics["MAE"], "hw_mse": hw_metrics["MSE"],
            "hw_rmse": hw_metrics["RMSE"], "hw_mape": hw_metrics["MAPE"],
        }
        # add 6-step forecasts per model
        for i in range(FORECAST_STEPS):
            t = i + 1
            row[f"sarima_t+{t}"] = float(sa_fc.iloc[i]) if not pd.isna(sa_fc.iloc[i]) else np.nan
            row[f"ets_t+{t}"]    = float(ets_fc.iloc[i]) if not pd.isna(ets_fc.iloc[i]) else np.nan
            row[f"hw_t+{t}"]     = float(hw_fc.iloc[i])  if not pd.isna(hw_fc.iloc[i])  else np.nan

        rows.append(row)

    return rows

# ---------------- Entrypoint ----------------
def main():
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        print(f"❌ No files found in {DATA_DIR}. Put .csv/.xlsx there."); return
    print(f"Found {len(files)} dataset(s) in {DATA_DIR}")

    all_rows: List[dict] = []
    for f in files:
        all_rows.extend(run_on_file(f))

    # write ONE consolidated CSV in OUT_ROOT
    if all_rows:
        out_csv = OUT_ROOT / "sarima_ets_hw_results.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\n📑 Metrics CSV saved → {out_csv}")

    print("\n🎉 Done. PNGs and CSV are directly in:", OUT_ROOT)

if __name__ == "__main__":
    main()
