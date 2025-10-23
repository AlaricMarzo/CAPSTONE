import warnings
from pathlib import Path
import re
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

# --- headless backend (avoids Tkinter errors on Windows batch runs) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_DIR  = HERE / "ml_lstm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_STEPS = 6
TOP_N          = 5
N_STEPS        = 3  

# Column aliases (lowercased)
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

# ---------- helpers ----------
def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", str(s).strip().lower().replace(" ", "_")) or "name"

def parse_dates_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Return mapping original_name -> standardized: Date, Item Code, Description, Qty."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}

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
        raise ValueError("Could not detect a Date column (tried common aliases + datetime-like fallback).")
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

def make_sequences(values: np.ndarray, n_steps: int = N_STEPS) -> Tuple[np.ndarray, np.ndarray]:
    """Create X (n_steps window) and y (next value)."""
    X, y = [], []
    for i in range(len(values) - n_steps):
        X.append(values[i:i+n_steps])
        y.append(values[i+n_steps])
    return np.array(X), np.array(y)

# ---------- core ----------
def run_file(path: Path):
    print(f"\n📄 {path.name}")

    # robust read
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # autodetect + normalize columns
    rename = detect_columns(df)
    print("Detected columns:", rename)
    df = df.rename(columns=rename)

    df["Date"] = parse_dates_safe(df["Date"])
    df = df.dropna(subset=["Date"])
    df["Description"] = df["Description"].astype(str).fillna("Unknown Product")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    if "Item Code" not in df.columns:
        df["Item Code"] = df["Description"].astype(str)  # fallback key

    # description per SKU
    sku_desc = (
        df[["Item Code", "Description"]]
        .dropna(subset=["Description"])
        .drop_duplicates(subset=["Item Code"])
        .set_index("Item Code")["Description"]
        .to_dict()
    )

    # monthly per SKU
    monthly = (df.groupby(["Item Code", pd.Grouper(key="Date", freq="MS")])["Qty"]
                 .sum().reset_index())
    if monthly.empty:
        print("   (no rows after monthly aggregation)")
        return

    # top movers
    top_skus = (monthly.groupby("Item Code")["Qty"].sum()
                .sort_values(ascending=False).head(TOP_N).index.tolist())

    out_dir = OUT_DIR / clean_name(path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    for sku in top_skus:
        ser = monthly[monthly["Item Code"] == sku].copy().sort_values("Date")
        if ser["Date"].nunique() < 10:
            print(f"   ↪ {sku}: too few months, skipped.")
            continue

        desc = sku_desc.get(sku, "Unknown Product")
        label = f"{sku} — {desc}"

        y_vals = ser["Qty"].values.astype(float)

        # scale target
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y_vals.reshape(-1, 1)).flatten()

        # sequences (aligned dates for targets start at index N_STEPS)
        X_seq, y_seq = make_sequences(y_scaled, N_STEPS)
        target_dates = ser["Date"].iloc[N_STEPS:].reset_index(drop=True)

        if len(X_seq) < 8:
            print(f"   ↪ {sku}: too few rows after sequencing, skipped.")
            continue

        # smart time split: short → last 3 months, else 80/20
        if len(X_seq) < 15:
            test_size = 3
            split = len(X_seq) - test_size
        else:
            split = int(len(X_seq) * 0.8)

        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        # dates for the test targets
        dates_test = target_dates.iloc[split:].to_numpy()

        # reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

        # model
        model = Sequential([
            LSTM(64, activation="tanh", input_shape=(N_STEPS, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        # early stopping to avoid overfit
        es = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, callbacks=[es])

        # predictions and inverse scale
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

        mae = float(mean_absolute_error(y_test_inv, y_pred))
        mse = float(mean_squared_error(y_test_inv, y_pred))
        rmse = float(np.sqrt(mse))
        mp  = float(mape(y_test_inv, y_pred))

        # ----- roll-forward 6-month forecast -----
        last_seq = y_scaled[-N_STEPS:].reshape(1, N_STEPS, 1)
        fc_vals_scaled = []
        for _ in range(FORECAST_STEPS):
            next_pred = model.predict(last_seq, verbose=0)
            fc_vals_scaled.append(next_pred[0, 0])
            # append and shift window
            last_seq = np.append(last_seq[:, 1:, :], next_pred.reshape(1,1,1), axis=1)

        fc_vals = scaler.inverse_transform(np.array(fc_vals_scaled).reshape(-1,1)).flatten()
        last_date = pd.to_datetime(ser["Date"].iloc[-1])
        fc_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS")

        # ----- plot with bottom metrics footer (consistent style) -----
        plt.figure(figsize=(10,5))
        plt.plot(ser["Date"], ser["Qty"], label="Historical", linewidth=2)
        if len(y_pred) > 0:
            plt.plot(dates_test, y_pred.flatten(), "r--", label="Test Pred", linewidth=1.8)
        plt.plot(fc_dates, fc_vals, "g-", label="LSTM Forecast (t+1..t+6)", linewidth=2)
        plt.title(f"LSTM Forecast — {label}")
        plt.xlabel("Month"); plt.ylabel("Qty")
        plt.legend(); plt.grid(alpha=0.25)

        # bottom metric text block (same placement/format as other models)
        footer = f"LSTM → MAE {mae:.2f}, MSE {mse:.2f}, RMSE {rmse:.2f}, MAPE {mp:.2f}%"
        plt.gcf().text(0.5, -0.06, footer, ha="center", va="top", fontsize=9, linespacing=1.4)

        plt.tight_layout()
        out_img = out_dir / f"lstm_{clean_name(sku)}.png"
        plt.savefig(out_img, dpi=150, bbox_inches="tight")
        plt.close()

        rows.append({
            "dataset": path.name,
            "sku": sku,
            "description": desc,
            "lstm_rmse": rmse,
            "lstm_mae": mae,
            "lstm_mse": mse,
            "lstm_mape": mp,
            **{f"lstm_t+{i}": float(round(v,2)) for i, v in enumerate(fc_vals, 1)}
        })

        print(f"   ✅ {label}: RMSE={rmse:.2f}  MAE={mae:.2f}  MSE={mse:.2f}  MAPE={mp:.2f}%  → {out_img}")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "lstm_results.csv", index=False)
        print(f"✅ Saved → {out_dir/'lstm_results.csv'}")
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
