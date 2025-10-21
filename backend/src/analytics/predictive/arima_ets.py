import warnings
from pathlib import Path
import re
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- CONFIG ----------------
HERE      = Path(__file__).resolve().parent
DATA_DIR  = HERE / "cleaned"
OUT_ROOT  = HERE / "out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Accept common column aliases
ALIASES = {
    "Date": ["date", "transaction date", "receipt date", "sales date", "order date"],
    "Item Code": ["item code","sku","barcode","product code","code","upc","ean"],
    "Description": ["description","item name","product","product name","name"],
    "Qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty"],
}

SEASONAL_PERIODS = 12
FORECAST_STEPS   = 6
TOP_N_PER_FILE   = 5

# -------------- Helpers --------------
def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", s.strip().lower().replace(" ", "_")) or "name"

def parse_dates_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

# --- REPLACE your detect_columns with this ---
def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    """
    Return a {original_name: standardized_name} map.
    Standardized names: "Date", "Item Code", "Description", "Qty".
    If no Item Code is present, we fallback to Description as the key.
    """
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}

    # Expanded alias lists (case-insensitive)
    aliases = {
        "Date": ["date","transaction date","receipt date","sales date","order date","invoice date","trans date"],
        "Item Code": [
            "item code","item_code","itemcode","code","sku","sku id","sku_id","barcode",
            "product code","product_code","productcode","upc","ean","itemid","item id","id"
        ],
        "Description": ["description","item name","product","product name","name","item","desc"],
        "Qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty","sold qty","sale qty","qnt"],
    }

    def pick(std: str) -> Optional[str]:
        # exact
        for k, orig in lower_map.items():
            if k == std.lower():
                return orig
        # aliases
        for alias in aliases.get(std, []):
            if alias in lower_map:
                return lower_map[alias]
        # fallback for Date: first datetime-like looking column
        if std == "Date":
            for c in df.columns:
                try:
                    s = pd.to_datetime(df[c].head(50), errors="coerce")
                    if s.notna().sum() >= 10:
                        return c
                except Exception:
                    pass
        return None

    # Try to detect all four first
    got_date = pick("Date")
    got_sku  = pick("Item Code")
    got_desc = pick("Description")
    got_qty  = pick("Qty")

    if not got_date:
        raise ValueError("Could not detect a Date column (tried common aliases and datetime-like fallback).")
    if not got_desc:
        raise ValueError("Could not detect a Description column. Please add/rename a product description column.")
    if not got_qty:
        raise ValueError("Could not detect a Qty/Quantity column. Please ensure a numeric quantity column exists.")

    # If no SKU column, we'll synthesize one from Description later in load_monthly
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

def ensure_fc_index(hist: pd.Series, fc: pd.Series) -> pd.Series:
    if not hasattr(fc.index, "freq") or fc.index.freq is None:
        fc.index = pd.date_range(hist.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=len(fc), freq="MS")
    return fc

# -------------- Data prep --------------
# --- REPLACE your load_monthly with this ---
def load_monthly(path: Path) -> Tuple[pd.DataFrame, Dict[str,str]]:
    # Robust CSV/XLSX loader
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # low_memory=False to avoid DtypeWarning and mixed-type chunking
        df = pd.read_csv(path, low_memory=False)

    # Normalize cols (with fallback to Description as key if SKU missing)
    rename = detect_columns(df)
    df = df.rename(columns=rename)

    # Basic cleaning
    df["Date"] = parse_dates_safe(df["Date"])
    df = df.dropna(subset=["Date"])

    # If we didn't detect Item Code, synthesize it from Description
    if "Item Code" not in df.columns:
        df["Item Code"] = df["Description"].astype(str)

    # Ensure types
    df["Item Code"]   = df["Item Code"].astype(str)
    df["Description"] = df["Description"].astype(str).fillna("Unknown Product")
    df["Qty"]         = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)

    # Map code -> description
    sku_desc = (
        df[["Item Code","Description"]]
        .dropna(subset=["Description"])
        .drop_duplicates(subset=["Item Code"])
        .set_index("Item Code")["Description"]
        .to_dict()
    )

    # Monthly aggregation per SKU
    monthly = (
        df.set_index("Date")
          .groupby("Item Code")["Qty"]
          .resample("MS")
          .sum()
          .reset_index()
    )
    return monthly, sku_desc


def series_for(monthly: pd.DataFrame, sku: str) -> pd.Series:
    return (monthly[monthly["Item Code"] == sku]
            .set_index("Date")["Qty"]
            .asfreq("MS", fill_value=0)
            .astype(float))

def top_skus(monthly: pd.DataFrame, n: int) -> list:
    return list(monthly.groupby("Item Code")["Qty"].sum().sort_values(ascending=False).head(n).index)

# -------------- Models --------------
def fit_arima(train: pd.Series):
    return ARIMA(train, order=(1,1,1),
                 enforce_stationarity=False,
                 enforce_invertibility=False).fit()

def fit_ets(train: pd.Series):
    use_season = len(train) >= 2 * SEASONAL_PERIODS
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add" if use_season else None,
        seasonal_periods=SEASONAL_PERIODS if use_season else None,
        initialization_method="estimated"
    )
    return model.fit()

# -------------- Report builder --------------
def build_report(y: pd.Series,
                 ar_fc: pd.Series, ar_mae: float, ar_mse: float, ar_mape: float,
                 ets_fc: pd.Series, et_mae: float, et_mse: float, et_mape: float,
                 title: str, out_png: Path, out_pdf: Path):

    # Make chart + table on one canvas
    plt.figure(figsize=(10.7, 7.5))  # A4-ish landscape feel
    gs = plt.GridSpec(3, 1, height_ratios=[2.1, 0.1, 0.9])

    # -------- Chart (top) --------
    ax = plt.subplot(gs[0, 0])
    ax.plot(y.index, y.values, label="Historical", linewidth=2)
    ar_fc = ensure_fc_index(y, ar_fc)
    ets_fc = ensure_fc_index(y, ets_fc)
    ax.plot(ar_fc.index, ar_fc.values, label="ARIMA Forecast", linewidth=2)
    ax.plot(ets_fc.index, ets_fc.values, label="ETS Forecast", linewidth=2)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Month"); ax.set_ylabel("Qty")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    # -------- Spacer (middle) --------
    plt.subplot(gs[1, 0]).axis("off")

    # -------- Table (bottom) --------
    ax2 = plt.subplot(gs[2, 0])
    ax2.axis("off")

    # Build table data
    months = [d.strftime("%Y-%m") for d in ar_fc.index]
    header = ["Model","MAE","MSE","MAPE %"] + [f"t+{i+1}" for i in range(len(months))]
    row_arima = ["ARIMA",
                 f"{ar_mae:.2f}" if ar_mae is not None else "N/A",
                 f"{ar_mse:.2f}" if ar_mse is not None else "N/A",
                 f"{ar_mape:.2f}" if ar_mape is not None else "N/A"] + [f"{v:.0f}" for v in ar_fc.values]
    row_ets   = ["ETS",
                 f"{et_mae:.2f}" if et_mae is not None else "N/A",
                 f"{et_mse:.2f}" if et_mse is not None else "N/A",
                 f"{et_mape:.2f}" if et_mape is not None else "N/A"] + [f"{v:.0f}" for v in ets_fc.values]
    table_data = [header, row_arima, row_ets]

    tbl = ax2.table(cellText=table_data,
                    loc="center",
                    cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)

    # style header
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6eefc")
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.savefig(out_pdf)  # vector for thesis
    plt.close()

# -------------- Main per-file pipeline --------------
def run_on_file(file_path: Path):
    print(f"\n📄 {file_path.name}")
    monthly, sku_desc = load_monthly(file_path)
    if monthly.empty:
        print("   (empty after cleaning)")
        return

    skus = top_skus(monthly, TOP_N_PER_FILE)
    base_out = OUT_ROOT / clean_name(file_path.stem)

    for sku in skus:
        y = series_for(monthly, sku)
        if len(y) < 6:
            print(f"   ↪ {sku}: too few points ({len(y)}), skipped.")
            continue

        desc = sku_desc.get(sku, "Unknown Product")
        label = f"{sku} — {desc}"

        # holdout split
        test_steps = 3 if len(y) > 9 else max(1, len(y)//4)
        train, test = y.iloc[:-test_steps], y.iloc[-test_steps:]

        # ARIMA
        try:
            ar_fit = fit_arima(train)
            ar_val = ar_fit.forecast(steps=len(test)) if len(test) else pd.Series(dtype=float)
            ar_mae = mean_absolute_error(test, ar_val) if len(test) else None
            ar_mse = mean_squared_error(test, ar_val) if len(test) else None
            ar_mp  = mape(test, ar_val) if len(test) else None

            ar_full = fit_arima(y)
            ar_fc   = ar_full.forecast(FORECAST_STEPS)
        except Exception as e:
            print(f"   ⚠️ ARIMA failed for {sku}: {e}")
            ar_fc = pd.Series([np.nan]*FORECAST_STEPS, index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS"))
            ar_mae = ar_mse = ar_mp = None

        # ETS
        try:
            ets_fit = fit_ets(train)
            ets_val = ets_fit.forecast(len(test)) if len(test) else pd.Series(dtype=float)
            et_mae = mean_absolute_error(test, ets_val) if len(test) else None
            et_mse = mean_squared_error(test, ets_val) if len(test) else None
            et_mp  = mape(test, ets_val) if len(test) else None

            ets_full = fit_ets(y)
            ets_fc   = ets_full.forecast(FORECAST_STEPS)
        except Exception as e:
            print(f"   ⚠️ ETS failed for {sku}: {e}")
            ets_fc = pd.Series([np.nan]*FORECAST_STEPS, index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS"))
            et_mae = et_mse = et_mp = None

        out_dir = base_out / clean_name(sku)
        out_png = out_dir / f"report_{clean_name(sku)}.png"
        out_pdf = out_dir / f"report_{clean_name(sku)}.pdf"

        build_report(
            y, ar_fc, ar_mae, ar_mse, ar_mp,
               ets_fc, et_mae, et_mse, et_mp,
            title=f"ARIMA vs ETS Forecast — {label}",
            out_png=out_png, out_pdf=out_pdf
        )
        print(f"   ✅ Saved → {out_png}")

# -------------- Entrypoint --------------
def main():
    files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not files:
        print(f"❌ No files found in {DATA_DIR}. Put .csv/.xlsx there.")
        return
    print(f"Found {len(files)} dataset(s) in {DATA_DIR}")
    for f in files:
        # skip helpers like *_errors.csv
        if "error" in f.stem.lower() or f.stem.lower().endswith("_errors"):
            print(f"   ↪ Skipping helper file: {f.name}")
            continue
        run_on_file(f)
    print("\n🎉 Done. See /analytics/out/<dataset>/<sku>/report_<sku>.png")

if __name__ == "__main__":
    main()
