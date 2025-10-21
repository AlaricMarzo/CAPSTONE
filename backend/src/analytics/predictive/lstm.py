# backend/analytics/lstm_model.py
import warnings
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. LSTM model requires TensorFlow. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_STEPS = 6
TOP_N = 5

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

def clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "", str(s).strip().lower().replace(" ", "_")) or "name"

def parse_dates_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return mapping original_name -> standardized: Date, Item Code, Description, Qty."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}

    def pick(std: str) -> str | None:
        # exact
        if std.lower() in lower_map:
            return lower_map[std.lower()]
        # aliases
        for a in ALIASES.get(std, []):
            if a in lower_map:
                return lower_map[a]
        # fallback for Date: look for datetime-like col
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

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred)/denom))*100

def make_sequences(series, n_steps=3):
    X, y = [], []
    for i in range(len(series)-n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

def run_file(path: Path):
    if not TENSORFLOW_AVAILABLE:
        print("⚠️ Skipping LSTM analysis due to missing TensorFlow dependency.")
        return

    print(f"\n📄 {path.name}")
    df = pd.read_csv(path)

    # Detect and rename columns
    rename_map = detect_columns(df)
    df = df.rename(columns=rename_map)

    df["Date"] = parse_dates_safe(df["Date"])
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    if "Item Code" in df.columns:
        df["Item Code"] = df["Item Code"].astype(str)
    df["Description"] = df["Description"].astype(str)

    monthly = df.groupby(["Item Code", pd.Grouper(key="Date", freq="MS")])["Qty"].sum().reset_index()
    top_skus = monthly.groupby("Item Code")["Qty"].sum().sort_values(ascending=False).head(TOP_N).index
    out_dir = OUT_DIR / clean_name(path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for sku in top_skus:
        y = monthly[monthly["Item Code"] == sku]["Qty"].values
        if len(y) < 10: continue

        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1,1)).flatten()
        X_seq, y_seq = make_sequences(y_scaled, 3)

        split = int(len(X_seq)*0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential([
            LSTM(50, activation="relu", input_shape=(3,1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=100, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mp  = mape(y_test_inv, y_pred_inv)

        # forecast next 6 months
        fc_vals = []
        last_seq = y_scaled[-3:].reshape(1,3,1)
        for _ in range(FORECAST_STEPS):
            next_pred = model.predict(last_seq)
            fc_vals.append(next_pred[0,0])
            last_seq = np.append(last_seq[:,1:,:], [[next_pred[0]]], axis=1)
        fc_inv = scaler.inverse_transform(np.array(fc_vals).reshape(-1,1)).flatten()

        plt.figure(figsize=(10,5))
        plt.plot(y, label="Historical")
        plt.plot(range(split, split+len(y_pred_inv)), y_pred_inv, "r--", label="Test Pred")
        plt.plot(range(len(y), len(y)+len(fc_inv)), fc_inv, "g-", label="LSTM Forecast")
        plt.title(f"LSTM — {sku}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"lstm_{clean_name(sku)}.png", dpi=150)
        plt.close()

        row = {"sku": sku, "mae": mae, "mse": mse, "mape": mp}
        for i,v in enumerate(fc_inv,1):
            row[f"lstm_t+{i}"] = v
        results.append(row)

        print(f"   ✅ {sku}: MAE={mae:.2f}, MAPE={mp:.2f}%")

    pd.DataFrame(results).to_csv(out_dir/"lstm_results.csv", index=False)
    print(f"✅ Saved → {out_dir/'lstm_results.csv'}")

def main():
    if len(sys.argv) > 1:
        # Use provided CSV path
        csv_path = Path(sys.argv[1])
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}")
            return
        files = [csv_path]
        print(f"Using provided file: {csv_path}")
    else:
        print("❌ No file provided. Please provide CSV path as argument.")
        return

    for f in files:
        run_file(f)

if __name__ == "__main__":
    main()
