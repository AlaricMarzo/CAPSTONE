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
DATA_DIR = HERE / "cleaned"             # put your CSV/XLSX here
OUT_ROOT = HERE / "ts_sarima_ets(2,1,2)" # flat outputs (PNGs + one CSV)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEASONAL_PERIODS  = 12     # monthly seasonality (m)
FORECAST_STEPS    = 6      # 6 months ahead
TOP_N_PER_FILE    = 5      # how many frequent products to chart
MIN_SERIES_LENGTH = 12     # skip very short series
MIN_SARIMA_TRAIN  = 24     # SARIMA needs ~2 seasons
RANDOM_SEED       = 2025   # reproducible random test months

# ---------------- Helpers ----------------
def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", s.strip().lower().replace(" ", "_")) or "name"

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
    """NaN-safe MAPE that ignores zero true values."""
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

    df["Description"] = df["Description"].astype(str).fillna("Unknown Product")
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

def top_products_by_frequency(monthly: pd.DataFrame, n: int) -> List[str]:
    """'Frequently purchased' = most months with sales > 0."""
    freq = (monthly.groupby("Description")["Qty"]
            .apply(lambda s: (s > 0).sum()))
    return freq.sort_values(ascending=False).head(n).index.tolist()

# ---------------- Models ----------------
def fit_sarima_212(train: pd.Series, m: int = SEASONAL_PERIODS):
    """SARIMA with fixed order (2,1,2) x (1,1,1,12)."""
    return SARIMAX(
        train,
        order=(2,1,2),
        seasonal_order=(1,1,1,m),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

def fit_ets(train: pd.Series):
    """
    ETS (Error, Trend, Seasonal) using Holt-Winters' ExponentialSmoothing.
    Additive trend + additive seasonality when enough history exists.
    """
    use_season = len(train) >= 2 * SEASONAL_PERIODS
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=("add" if use_season else None),
        seasonal_periods=(SEASONAL_PERIODS if use_season else None),
        initialization_method="estimated",
    )
    return model.fit()

# ---------------- Random 80/20 evaluation (time-valid) ----------------
def random_one_step_eval(y: pd.Series, test_frac: float = 0.2, rng_seed: int = 2025):
    rng = np.random.default_rng(rng_seed)
    dates = y.index

    earliest = 6  # keep first 6 months for initial training
    candidate_idx = np.arange(earliest, len(dates))
    if len(candidate_idx) == 0:
        return None, pd.DatetimeIndex([])

    n_test = max(1, int(len(dates) * test_frac))
    test_positions = rng.choice(candidate_idx, size=min(n_test, len(candidate_idx)), replace=False)
    test_idx = dates[np.sort(test_positions)]

    def eval_model(fit_fn, min_train_points: int):
        preds, actuals = [], []
        for t in test_idx:
            cutoff_pos = np.where(dates == t)[0][0]
            train = y.iloc[:cutoff_pos]           # strictly before t
            if len(train) < min_train_points:
                continue
            try:
                model = fit_fn(train)
                pred = float(model.forecast(1).iloc[0])
                preds.append(pred)
                actuals.append(float(y.loc[t]))
            except Exception:
                pass

        if len(preds) == 0:
            metrics = {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
        else:
            metrics = {
                "MAE": mean_absolute_error(actuals, preds),
                "MSE": mean_squared_error(actuals, preds),
                "RMSE": rmse(actuals, preds),
                "MAPE": mape_safe(actuals, preds),
            }

        # final 6-month forecast trained on full history
        try:
            final = fit_fn(y)
            fc = final.forecast(FORECAST_STEPS)
        except Exception:
            fc = pd.Series([np.nan]*FORECAST_STEPS,
                           index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),
                                               periods=FORECAST_STEPS, freq="MS"))
        return metrics, fc

    # SARIMA (require some history)
    if len(y) >= MIN_SARIMA_TRAIN:
        sa_metrics, sa_fc = eval_model(lambda s: fit_sarima_212(s), min_train_points=MIN_SARIMA_TRAIN//2)
    else:
        sa_metrics = {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
        sa_fc = pd.Series([np.nan]*FORECAST_STEPS,
                          index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),
                                              periods=FORECAST_STEPS, freq="MS"))

    # ETS
    ets_metrics, ets_fc = eval_model(fit_ets, min_train_points=12)

    results = {"SARIMA": (sa_metrics, sa_fc),
               "ETS":    (ets_metrics, ets_fc)}
    return results, test_idx

# ---------------- Plot ----------------
def build_random_report(y: pd.Series,
                        results: dict,
                        test_idx: pd.DatetimeIndex,
                        title: str,
                        out_png: Path):
    sa_metrics, sa_fc = results["SARIMA"]
    ets_metrics, ets_fc = results["ETS"]

    sa_fc = ensure_fc_index(y, sa_fc)
    ets_fc = ensure_fc_index(y, ets_fc)

    # training vs random test
    train = y[~y.index.isin(test_idx)]
    test  = y[y.index.isin(test_idx)]

    plt.figure(figsize=(10.7, 7))
    plt.plot(train.index, train.values, label="Training Data", linewidth=1.2)
    plt.plot(test.index, test.values, color="black", label="Test Data", linewidth=1.5)

    if not np.all(np.isnan(sa_fc.values)):
        plt.plot(sa_fc.index, sa_fc.values, color="red", linewidth=2,
                 label="SARIMA (2,1,2)×(1,1,1,12) Forecast")
    plt.plot(ets_fc.index, ets_fc.values, color="deepskyblue", linewidth=2,
             label="ETS Forecast (trend+seasonal)")

    plt.title(title, fontsize=13, pad=10)
    plt.xlabel("Month"); plt.ylabel("Quantity")
    plt.grid(alpha=0.3)
    plt.legend()

    # Bottom metrics (same styling convention)
    def fmt(m):
        if any(pd.isna(list(m.values()))):
            return "MAE: n/a   MSE: n/a   RMSE: n/a   MAPE: n/a"
        return f"MAE: {m['MAE']:.2f}   MSE: {m['MSE']:.2f}   RMSE: {m['RMSE']:.2f}   MAPE: {m['MAPE']:.2f}%"

    text = (f"SARIMA (2,1,2)×(1,1,1,12)  →  {fmt(sa_metrics)}\n"
            f"ETS (additive trend/season) →  {fmt(ets_metrics)}")
    plt.gcf().text(0.5, -0.06, text, ha="center", va="top", fontsize=9, linespacing=1.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

# ---------------- Per-file driver ----------------
def run_on_file(file_path: Path) -> List[dict]:
    print(f"\n📄 {file_path.name}")
    monthly = load_monthly(file_path)
    if monthly.empty:
        print("   (empty after cleaning)")
        return []

    products = top_products_by_frequency(monthly, TOP_N_PER_FILE)
    dataset_slug = clean_name(file_path.stem)

    rows: List[dict] = []

    for desc in products:
        y = series_for(monthly, desc)
        if len(y) < MIN_SERIES_LENGTH:
            print(f"   ↪ {desc}: too few months ({len(y)}), skipped.")
            continue

        results, test_idx = random_one_step_eval(y, test_frac=0.2, rng_seed=RANDOM_SEED)
        if results is None:
            print(f"   ↪ {desc}: not enough data to random-split, skipped.")
            continue

        sa_metrics, sa_fc = results["SARIMA"]
        ets_metrics, ets_fc = results["ETS"]

        # flat filename, no folders
        product_slug = clean_name(desc)
        out_png = OUT_ROOT / f"{dataset_slug}__{product_slug}__random_sarima-ets.png"

        build_random_report(
            y, results, test_idx,
            title=f"SARIMA vs. ETS — Random Train/Test — {desc}",
            out_png=out_png
        )
        print(f"   ✅ Saved → {out_png}")

        # consolidated CSV row
        row = {
            "dataset": file_path.name,
            "product": desc,
            "random_test_points": int(len(test_idx)),
            # SARIMA metrics
            "sarima_mae": sa_metrics["MAE"], "sarima_mse": sa_metrics["MSE"],
            "sarima_rmse": sa_metrics["RMSE"], "sarima_mape": sa_metrics["MAPE"],
            # ETS metrics
            "ets_mae": ets_metrics["MAE"], "ets_mse": ets_metrics["MSE"],
            "ets_rmse": ets_metrics["RMSE"], "ets_mape": ets_metrics["MAPE"],
        }
        # Add 6-step forecasts per model
        sa_fc = ensure_fc_index(y, sa_fc)
        ets_fc = ensure_fc_index(y, ets_fc)
        for i in range(FORECAST_STEPS):
            t = i + 1
            row[f"sarima_t+{t}"] = float(sa_fc.iloc[i]) if not pd.isna(sa_fc.iloc[i]) else np.nan
            row[f"ets_t+{t}"]    = float(ets_fc.iloc[i]) if not pd.isna(ets_fc.iloc[i]) else np.nan

        rows.append(row)

    return rows

# ---------------- Entrypoint ----------------
def main():
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        print(f"❌ No files found in {DATA_DIR}. Put .csv/.xlsx there.")
        return
    print(f"Found {len(files)} dataset(s) in {DATA_DIR}")

    all_rows: List[dict] = []
    for f in files:
        if "error" in f.stem.lower() or f.stem.lower().endswith("_errors"):
            print(f"   ↪ Skipping helper file: {f.name}")
            continue
        all_rows.extend(run_on_file(f))

    # single consolidated CSV in OUT_ROOT
    if all_rows:
        out_csv = OUT_ROOT / "sarima_ets_random_results.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\n📑 Metrics CSV saved → {out_csv}")

    print("\n🎉 Done. PNGs and CSV are directly in:", OUT_ROOT)

if __name__ == "__main__":
    main()
