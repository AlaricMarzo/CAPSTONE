import warnings, argparse, re, sys, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE=Path(__file__).resolve().parent
DATA_DIR=HERE/"cleaned"
OUT_ROOT=HERE/"ml_lstm"; OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------- knobs ----------------------------
FORECAST_STEPS=6
SEASON_M=12
TOP_N=10
INTERMITTENT_ZERO_SHARE=0.40

ALIASES = {
    "Date": ["date","transaction date","trans date","receipt date","sales date","order date","invoice date","posting date"],
    "Description": ["description","desc","item name","product","product name","name","item"],
    "Qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty","sold qty","sale qty","qnt","qnty"],
    "Item Code": ["item code","item_code","itemcode","sku","sku id","sku_id","barcode","product code","product_code","productcode","upc","ean","code","id","item id","itemid"],
}
def clean_name(s:str)->str: return re.sub(r"[^a-z0-9_\-]+","_", str(s).strip().lower().replace(" ","_")) or "name"
def parse_dates_safe(s: pd.Series) -> pd.Series:
    try:  return pd.to_datetime(s, errors="coerce", dayfirst=False)
    except: return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    lower = {c.lower().strip(): c for c in df.columns}
    def pick(std:str):
        if std.lower() in lower: return lower[std.lower()]
        for a in ALIASES.get(std, []):
            if a in lower: return lower[a]
        if std=="Date":
            for c in df.columns:
                try:
                    s = pd.to_datetime(df[c].head(50), errors="coerce")
                    if s.notna().sum()>=10: return c
                except: pass
        return None
    d=pick("Date"); q=pick("Qty"); desc=pick("Description"); sku=pick("Item Code")
    if not d or not q or not desc: raise ValueError("Could not detect Date/Qty/Description.")
    ren = {d:"Date", q:"Qty", desc:"Description"}
    if sku: ren[sku]="Item Code"
    return ren

def monthly(df: pd.DataFrame, key:str) -> pd.DataFrame:
    return (df.groupby([key, pd.Grouper(key="Date", freq="MS")])["Qty"].sum()
            .reset_index().sort_values([key,"Date"]))

def choose_top_best(mon: pd.DataFrame, key:str, topn:int, min_cov:float) -> List[str]:
    stats = (mon.groupby(key)["Qty"].agg(total="sum", nz=lambda s:int((s>0).sum()), n="size").reset_index())
    stats["coverage"]=stats["nz"]/stats["n"]
    pool = stats[stats["coverage"]>=min_cov] if (stats["coverage"]>=min_cov).any() else stats
    return pool.sort_values(["total","coverage"], ascending=[False,False])[key].head(topn).tolist()

def winsorize(y: pd.Series, q=0.995) -> pd.Series:
    if len(y)<8: return y
    return y.clip(upper=float(y.quantile(q)))

# ---------------- metrics ----------------
def mape(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float); den=np.where(yt==0,1,yt)
    return float(np.mean(np.abs((yt-yp)/den))*100)

def wape(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float); den=np.sum(np.abs(yt))
    return float(np.sum(np.abs(yt-yp))/den*100) if den>0 else np.nan

def mpe(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float); mask=yt!=0
    return float(np.mean(((yp[mask]-yt[mask])/yt[mask]))*100) if mask.any() else np.nan

def mase(y_true, y_pred, insample, m=SEASON_M):
    ins=np.asarray(insample,float)
    if len(ins)<=m: denom=np.mean(np.abs(np.diff(ins))) if len(ins)>1 else 1.0
    else: denom=np.mean(np.abs(ins[m:]-ins[:-m]))
    if denom==0: denom=1.0
    return float(np.mean(np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float)))/denom)

# ---------------- baselines ----------------
def seasonal_naive(y, h, m=SEASON_M):
    y=np.asarray(y,float)
    if len(y)<m: last=y[-1] if len(y) else 0.0; return np.repeat(max(last,0.0), h)
    pat=y[-m:]; reps=int(np.ceil(h/m)); fc=np.tile(pat,reps)[:h]; return np.maximum(fc,0.0)

def is_intermittent(y: pd.Series, thr=INTERMITTENT_ZERO_SHARE)->bool: return (y==0).mean() >= thr

def croston_sba(y: pd.Series, alpha=0.1, steps=FORECAST_STEPS)->pd.Series:
    yv=y.values.astype(float); z_hat=None; p_hat=None; k=0
    for val in yv:
        k+=1
        if val>0:
            z_hat = val if z_hat is None else z_hat + alpha*(val-z_hat)
            p_hat = k   if p_hat is None else p_hat + alpha*(k-p_hat)
            k=0
    fc = np.zeros(steps) if z_hat is None else np.array([(z_hat/max(p_hat,1e-8))*(1-alpha/2.0)]*steps)
    idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(fc, index=idx)

# ---------------- LSTM helpers ----------------
def make_sequences(values: np.ndarray, n_steps:int)->Tuple[np.ndarray,np.ndarray]:
    X,y=[],[]
    for i in range(len(values)-n_steps):
        X.append(values[i:i+n_steps]); y.append(values[i+n_steps])
    return np.array(X), np.array(y)

def build_lstm(n_steps:int):
    m=Sequential([LSTM(64, activation="tanh", input_shape=(n_steps,1)), Dense(1)])
    m.compile(optimizer="adam", loss="mse"); return m

# ---------------- plot (legend outside, connected lines, no Params col) ----------------
def plot_with_bands(title, y_idx, y_vals, train_idx, test_idx, pred_series, metrics_row, future_fc, out_png):
    fig=plt.figure(figsize=(14.5,7))
    gs=fig.add_gridspec(2,1,height_ratios=[3.1,1.0],hspace=0.15)
    ax=fig.add_subplot(gs[0])

    # Historical
    series_y = pd.Series(y_vals, index=pd.to_datetime(y_idx))
    ax.plot(series_y.index, series_y.values, label="Historical (Actual)", lw=2, zorder=2)

    # Regions
    if len(train_idx):
        ax.axvspan(train_idx[0], train_idx[-1], color="tab:blue", alpha=0.06, label="PAST (Train)", zorder=0)
        last_train_dt = pd.to_datetime(train_idx[-1])
        if last_train_dt not in series_y.index:
            prev = series_y.index[series_y.index <= last_train_dt]
            last_train_dt = prev[-1] if len(prev) else series_y.index[0]
        last_train_val = float(series_y.loc[last_train_dt])
    else:
        last_train_dt = series_y.index[0]; last_train_val = float(series_y.iloc[0])

    if len(test_idx):
        ax.axvspan(test_idx[0], test_idx[-1], color="tab:orange", alpha=0.08, label="PRESENT (Holdout)", zorder=0)
        split_dt = pd.to_datetime(test_idx[0])
        ax.axvline(split_dt, ls="-.", lw=1.5, alpha=0.7, zorder=1)
        ymax = ax.get_ylim()[1]
        ax.text(split_dt, ymax*0.95, "80/20 split", rotation=90, va="top", ha="right", fontsize=9, alpha=0.8)

    # Connected holdout line
    if pred_series is not None and not pred_series.empty:
        ser = pred_series.sort_index()
        ser_conn = pd.concat([pd.Series([last_train_val], index=[last_train_dt]), ser])
        ax.plot(ser_conn.index, ser_conn.values, lw=2, ls="--", label="Chosen (holdout)", zorder=3)

    # Holdout + Future continuous line
    if pred_series is not None and not pred_series.empty and future_fc is not None and not future_fc.empty:
        full_future = pd.concat([pred_series.sort_index(), future_fc.sort_index()])
        full_future = full_future[~full_future.index.duplicated(keep="last")]
        full_future = pd.concat([pd.Series([last_train_val], index=[last_train_dt]), full_future])
        ax.axvspan(future_fc.index[0], future_fc.index[-1], color="tab:green", alpha=0.07, label="FUTURE (Forecast)", zorder=0)
        ax.plot(full_future.index, full_future.values, lw=2.2, label="Forecast (holdout + future)", zorder=4)

    # Cosmetics & legend outside
    ax.set_title(title)
    ax.set_xlabel("Month"); ax.set_ylabel("Qty"); ax.grid(ls=":", alpha=0.6); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels(): t.set_rotation(45); t.set_ha("right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0, frameon=True, framealpha=0.9, ncol=1)
    fig.subplots_adjust(right=0.82)

    # Compact metrics (no Params)
    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    header=["Model","MAE","RMSE","MAPE%","MASE","WAPE%","MPE%"]
    if metrics_row is not None:
        n, mae, rmse, mape_v, mase_v, wape_v, mpe_v = metrics_row
        rows=[[n, f"{mae:,.2f}", f"{rmse:,.2f}", f"{mape_v:,.2f}", f"{mase_v:.3f}", f"{wape_v:,.2f}", f"{mpe_v:,.2f}"]]
        tbl=ax2.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.10)

    fig.tight_layout(); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------- split & selection ----------------
def split_80_20(y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    h = max(1, math.ceil(len(y) * 0.20))
    return y.iloc[:-h], y.iloc[-h:]

def metric_row(name, y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_v = mape(y_true, y_pred)
    mase_v = mase(y_true, y_pred, insample, SEASON_M)
    wape_v = wape(y_true, y_pred)
    mpe_v = mpe(y_true, y_pred)
    return (name, mae, rmse, mape_v, mase_v, wape_v, mpe_v)

def choose_best_by_metric(cands: Dict[str, np.ndarray], y_true: np.ndarray, insample: np.ndarray) -> str:
    # Primary: lowest MASE; tie-break: WAPE, then RMSE
    ranking=[]
    for name, pred in cands.items():
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mase_v = mase(y_true, pred, insample, SEASON_M)
        wape_v = wape(y_true, pred)
        ranking.append((name, mase_v, wape_v, rmse))
    ranking.sort(key=lambda x: (np.nan_to_num(x[1], nan=1e9), np.nan_to_num(x[2], nan=1e9), np.nan_to_num(x[3], nan=1e9)))
    return ranking[0][0]

# ---------------- holdout using LSTM (with optional blend) ----------------
def holdout_predict(mode:str, y: pd.Series, n_steps:int, refit_every:int):
    train, test = split_80_20(y)

    # scale on train only
    scaler = MinMaxScaler()
    y_tr = scaler.fit_transform(train.values.reshape(-1,1)).flatten()
    Xtr, ytr = make_sequences(y_tr, n_steps)

    if len(Xtr) < 8:  # not enough history → SN
        y_sn = seasonal_naive(train.values, len(test), SEASON_M)
        chosen = "Seasonal-Naive"
        chosen_ser = pd.Series(y_sn, index=test.index)
        row = metric_row("Seasonal-Naive", test.values, y_sn, train.values)
        return chosen, chosen_ser, row, train, test, {"alpha":None,"bias":None,"scaler":scaler,"model":None}

    # train LSTM on train sequences
    model = build_lstm(n_steps)
    es=EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
    model.fit(Xtr.reshape(Xtr.shape[0], n_steps,1), ytr, epochs=120, batch_size=16, verbose=0, callbacks=[es])

    # recursive prediction across holdout with quick refits
    preds_scaled=[]; last_seq = y_tr[-n_steps:].reshape(1,n_steps,1)
    for i in range(len(test)):
        pred = float(model.predict(last_seq, verbose=0)[0,0])
        preds_scaled.append(pred)
        last_seq = np.append(last_seq[:,1:,:], np.array(pred).reshape(1,1,1), axis=1)
        if (i+1) % max(1, refit_every) == 0:
            tmp = np.concatenate([y_tr, np.array(preds_scaled)], axis=0)
            Xrt, yrt = make_sequences(tmp, n_steps)
            model.fit(Xrt.reshape(Xrt.shape[0], n_steps,1), yrt, epochs=40, batch_size=16, verbose=0, callbacks=[es])

    yhat = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()

    # SN candidate
    y_sn = seasonal_naive(train.values, len(test), SEASON_M)

    # Blend (only in auto mode)
    y_blend=None; alpha=None; bias=None
    if mode=="auto":
        alphas=np.linspace(0,0.6,7); best_a=0; best_mae=1e18
        for a in alphas:
            yb=a*y_sn+(1-a)*yhat
            mae=mean_absolute_error(test.values, yb)
            if mae<best_mae: best_mae, best_a=mae, a
        yb=best_a*y_sn+(1-best_a)*yhat
        bc=np.clip((test.values.sum()/yb.sum()) if yb.sum()>0 else 1.0, 0.7, 1.3)
        yb*=bc
        y_blend=yb; alpha=best_a; bias=bc

    # Candidate set per mode
    if mode=="lstm_only":
        cands={"LSTM": yhat}
    else:
        cands={"LSTM": yhat, "Seasonal-Naive": y_sn}
        if y_blend is not None: cands["Blend"]=y_blend

    chosen_name = choose_best_by_metric(cands, test.values, train.values)
    chosen_pred = cands[chosen_name]
    chosen_ser = pd.Series(chosen_pred, index=test.index)
    row = metric_row(chosen_name, test.values, chosen_pred, train.values)

    extra = {"alpha":alpha, "bias":bias, "scaler":scaler, "model":model}
    return chosen_name, chosen_ser, row, train, test, extra

# ---------------- future forecast consistent with chosen logic ----------------
def future_forecast(y: pd.Series, mode:str, extra: Dict[str,object], n_steps:int) -> pd.Series:
    scaler = MinMaxScaler().fit(y.values.reshape(-1,1))
    ys = scaler.transform(y.values.reshape(-1,1)).flatten()
    X, yv = make_sequences(ys, n_steps)

    if len(X)<8:
        fc = seasonal_naive(y.values, FORECAST_STEPS, SEASON_M)
        idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),periods=FORECAST_STEPS,freq="MS")
        return pd.Series(fc, index=idx)

    model = extra.get("model", None)
    if model is None:
        model = build_lstm(n_steps)
        es=EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
        model.fit(X.reshape(X.shape[0], n_steps,1), yv, epochs=120, batch_size=16, verbose=0, callbacks=[es])

    last = ys[-n_steps:].reshape(1,n_steps,1)
    fc_scaled=[]
    for _ in range(FORECAST_STEPS):
        p=float(model.predict(last, verbose=0)[0,0]); fc_scaled.append(p)
        last=np.append(last[:,1:,:], np.array(p).reshape(1,1,1), axis=1)
    fc_ml = scaler.inverse_transform(np.array(fc_scaled).reshape(-1,1)).flatten()

    if mode=="auto" and extra.get("alpha") is not None and extra.get("bias") is not None:
        fc_sn=seasonal_naive(y.values, FORECAST_STEPS, SEASON_M)
        fc= extra["alpha"]*fc_sn + (1-extra["alpha"])*fc_ml
        fc*= extra["bias"]
    else:
        fc = fc_ml

    idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),periods=FORECAST_STEPS,freq="MS")
    return pd.Series(fc, index=idx)

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--topn", type=int, default=TOP_N)
    ap.add_argument("--min_cov", type=float, default=0.7)
    ap.add_argument("--n_steps", type=int, default=3)
    ap.add_argument("--wf_refit_every", type=int, default=3)
    ap.add_argument("--mode", type=str, choices=["auto","lstm_only"], default="auto")
    args=ap.parse_args()

    in_path=Path(args.input) if args.input else (DATA_DIR/"ANC - 4 YEARS (1).csv")
    if not in_path.exists(): sys.exit(f"Input not found: {in_path}")
    df = pd.read_csv(in_path, low_memory=False) if in_path.suffix.lower()==".csv" else pd.read_excel(in_path)

    rename=detect_columns(df); df=df.rename(columns=rename)
    df["Date"]=parse_dates_safe(df["Date"]); df=df.dropna(subset=["Date"])
    df["Qty"]=pd.to_numeric(df["Qty"], errors="coerce").fillna(0.0).astype(float)
    df["Description"]=df["Description"].astype(str)
    if "Item Code" not in df.columns: df["Item Code"]=df["Description"].astype(str)

    key="Item Code"
    mon=monthly(df,key)

    # Export Top-10 for reporting
    stats = (mon.groupby(key)["Qty"].agg(total="sum", months="size", nz=lambda s:int((s>0).sum()))
             .reset_index().rename(columns={"nz":"nonzero_months"}))
    stats["coverage"] = stats["nonzero_months"] / stats["months"]
    out_dir_root = OUT_ROOT/clean_name(in_path.stem); out_dir_root.mkdir(parents=True, exist_ok=True)
    stats.sort_values(["total","coverage"], ascending=[False,False]).head(args.topn)\
         .to_csv(out_dir_root/"top10_bestsellers.csv", index=False)

    # Choose top for modeling
    top=choose_top_best(mon,key,args.topn,args.min_cov)

    rows=[]
    for sku in top:
        s=mon[mon[key]==sku].copy().sort_values("Date")
        span=pd.date_range(s["Date"].min(), s["Date"].max(), freq="MS")   # all years available
        y=(s.set_index("Date")["Qty"].reindex(span).fillna(0.0).astype(float))
        y=winsorize(y,0.995)
        desc=df[df[key]==sku]["Description"].dropna().iloc[0] if (df[key]==sku).any() else sku

        chosen_name, holdout_ser, row, train, test, extra = holdout_predict(args.mode, y, args.n_steps, args.wf_refit_every)
        future = future_forecast(y, args.mode, extra, args.n_steps)

        # Save per-SKU files
        y_name=clean_name(sku)
        future.to_csv(out_dir_root/f"{y_name}_forecast.csv", header=["forecast"])

        actual = pd.Series(y.values, index=y.index, name="Actual")
        comp = pd.concat([actual, holdout_ser.rename("Holdout_Pred"), future.rename("Forecast")], axis=1)
        comp.index.name = "Month"
        comp.to_csv(out_dir_root / f"{y_name}_comparison.csv")

        # Plot (only chosen model)
        png=out_dir_root/f"lstm_{y_name}.png"
        plot_with_bands(
            title=f"{chosen_name} — 80/20 holdout — {sku} — {desc}",
            y_idx=y.index, y_vals=y.values, train_idx=train.index, test_idx=test.index,
            pred_series=holdout_ser, metrics_row=row, future_fc=future, out_png=str(png)
        )

        # Walk-forward MASE (1-step; compact WF engine)
        preds=[]; trues=[]
        for i in range(max(24, SEASON_M+args.n_steps), len(y)):
            hist=y.iloc[:i]
            sc=MinMaxScaler().fit(hist.values.reshape(-1,1))
            ys=sc.transform(hist.values.reshape(-1,1)).flatten()
            X,yv=make_sequences(ys, args.n_steps)
            if len(X)<8: continue
            m=build_lstm(args.n_steps)
            m.fit(X.reshape(X.shape[0],args.n_steps,1), yv, epochs=40, batch_size=16, verbose=0)
            last=ys[-args.n_steps:].reshape(1,args.n_steps,1)
            p=float(m.predict(last, verbose=0)[0,0])
            preds.append(float(sc.inverse_transform([[p]])[0,0])); trues.append(float(y.iloc[i]))
        wf_mase = mase(np.array(trues), np.array(preds), y.values[:-len(trues)] if trues else y.values, SEASON_M) if preds else np.nan

        rows.append({
            "sku":sku,"description":desc,
            "mode":args.mode,"chosen":chosen_name,
            "MAE":row[1],"RMSE":row[2],"MAPE%":row[3],"MASE":row[4],"WAPE%":row[5],"MPE%":row[6],
            "MASE_WF":wf_mase,
            "plot":png.name
        })

    if rows:
        pd.DataFrame(rows).to_csv(out_dir_root/"lstm_summary.csv", index=False)
        print(f"✅ Saved → {out_dir_root/'lstm_summary.csv'}")

if __name__=="__main__":
    main()
