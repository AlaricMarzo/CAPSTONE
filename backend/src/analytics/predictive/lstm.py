# backend/analytics/lstm_model.py
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_STEPS = 6
TOP_N = 5

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
    print(f"\n📄 {path.name}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    df["Item Code"] = df["Item Code"].astype(str)
    df["Description"] = df["Description"].astype(str)

    monthly = df.groupby(["Item Code", pd.Grouper(key="Date", freq="MS")])["Qty"].sum().reset_index()
    top_skus = monthly.groupby("Item Code")["Qty"].sum().sort_values(ascending=False).head(TOP_N).index
    out_dir = OUT_DIR / "lstm"
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
        plt.savefig(out_dir / f"lstm_{sku}.png", dpi=150)
        plt.close()

        results.append({"sku": sku, "mae": mae, "mse": mse, "mape": mp})
        print(f"✅ {sku}: MAE={mae:.2f}, MAPE={mp:.2f}%")

    pd.DataFrame(results).to_csv(out_dir/"lstm_results.csv", index=False)
    print(f"Saved → {out_dir/'lstm_results.csv'}")

def main():
    for f in DATA_DIR.glob("*.csv"):
        run_file(f)

if __name__ == "__main__":
    main()
