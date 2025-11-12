import warnings, argparse, re, sys, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE=Path(__file__).resolve().parent
DATA_DIR=HERE/"cleaned"
OUT_ROOT=HERE/"ml_gradient_boosting"; OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------- knobs ----------------------------
FORECAST_STEPS=6
SEASON_M=12
TOP_N=10                         # default to top 10
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

# Best-sellers by total qty, respecting min coverage
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

# ---------------- GB model ----------------
def fit_model_gb(Xtr, ytr_log):
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(Xtr, ytr_log); return model

MODEL_NAME = "GradientBoosting"

# TS features
def add_ts_features(s: pd.Series) -> pd.DataFrame:
    df=s.to_frame("y").copy(); idx=pd.to_datetime(df.index); month=idx.month
    df["m_sin"]=np.sin(2*np.pi*month/12.0); df["m_cos"]=np.cos(2*np.pi*month/12.0)
    for L in range(1, SEASON_M+1): df[f"lag{L}"]=df["y"].shift(L)
    for W in [3,6,12]:
        df[f"rmean{W}"]=df["y"].rolling(W).mean(); df[f"rstd{W}"]=df["y"].rolling(W).std(); df[f"rmed{W}"]=df["y"].rolling(W).median()
    df["diff1"]=df["y"].diff(1); df["pct1"]=df["y"].pct_change(1).replace([np.inf,-np.inf],0).fillna(0)
    return df.dropna()

def split_80_20(y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    h = max(1, math.ceil(len(y) * 0.20))
    return y.iloc[:-h], y.iloc[-h:]

# pick best by MASE (tie WAPE, then RMSE)
def choose_best_by_metric(cands: Dict[str, np.ndarray], y_true: np.ndarray, insample: np.ndarray) -> str:
    ranking=[]
    for name, pred in cands.items():
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mase_v = mase(y_true, pred, insample, SEASON_M)
        wape_v = wape(y_true, pred)
        ranking.append((name, mase_v, wape_v, rmse))
    ranking.sort(key=lambda x: (np.nan_to_num(x[1], nan=1e9),
                                np.nan_to_num(x[2], nan=1e9),
                                np.nan_to_num(x[3], nan=1e9)))
    return ranking[0][0]

# holdout evaluation (returns CHOSEN only)
def train_eval_holdout(y: pd.Series):
    train, test = split_80_20(y)
    feats_tr, feats_te = add_ts_features(train), add_ts_features(y).iloc[-len(test):]

    # if features empty → SN only
    if feats_tr.empty or feats_te.empty:
        y_sn = seasonal_naive(train.values, len(test), SEASON_M)
        chosen_name = "Seasonal-Naive"
        chosen_ser = pd.Series(y_sn, index=pd.date_range(test.index[0], periods=len(test), freq="MS"))
        row = ("Seasonal-Naive",
               mean_absolute_error(test,y_sn),
               np.sqrt(mean_squared_error(test,y_sn)),
               mape(test,y_sn),
               mase(test,y_sn, train.values, SEASON_M),
               wape(test,y_sn),
               mpe(test,y_sn))
        return train, test, chosen_name, chosen_ser, row, 0.0, 1.0, None

    # train GB
    y_tr_raw=feats_tr["y"].values
    cap=np.quantile(y_tr_raw,0.995)
    ytr_log=np.log1p(np.minimum(y_tr_raw,cap))
    mdl = fit_model_gb(feats_tr.drop(columns=["y"]).values, ytr_log)

    # predictions on holdout
    yhat = np.expm1(mdl.predict(feats_te.drop(columns=["y"]).values)).clip(0)
    y_sn = seasonal_naive(train.values, len(test), m=SEASON_M)

    # Blend alpha on MAE
    alphas=np.linspace(0,0.6,7); best_a=0; best_mae=1e18
    for a in alphas:
        yb=a*y_sn+(1-a)*yhat; mae=mean_absolute_error(test.values, yb)
        if mae<best_mae: best_mae, best_a = mae, a
    yb=best_a*y_sn+(1-best_a)*yhat

    # bias correction
    bc=np.clip((test.values.sum()/yb.sum()) if yb.sum()>0 else 1.0, 0.7, 1.3)
    yb*=bc

    # choose best
    cands={"GradientBoosting":yhat, "Seasonal-Naive":y_sn, "Blend": yb}
    chosen_name = choose_best_by_metric(cands, test.values, train.values)
    chosen_pred = cands[chosen_name]
    chosen_ser = pd.Series(chosen_pred, index=feats_te.index)

    row = (chosen_name,
           mean_absolute_error(test, chosen_pred),
           np.sqrt(mean_squared_error(test, chosen_pred)),
           mape(test, chosen_pred),
           mase(test, chosen_pred, train.values, SEASON_M),
           wape(test, chosen_pred),
           mpe(test, chosen_pred))

    return train, test, chosen_name, chosen_ser, row, best_a, bc, mdl

# one-step WF helper
def fit_fn_gb_1step(hist: pd.Series):
    feats=add_ts_features(hist)
    if feats.empty: return float(hist.iloc[-1])
    y_tr_raw=feats["y"].values; cap=np.quantile(y_tr_raw,0.995); ytr=np.log1p(np.minimum(y_tr_raw,cap))
    mdl=fit_model_gb(feats.drop(columns=["y"]).values, ytr)
    last=feats.index[-1].to_period("M").to_timestamp()
    nxt=(last.to_period("M")+1).to_timestamp()
    tmp=hist.copy(); tmp.loc[nxt]=hist.iloc[-1]
    feats_next=add_ts_features(tmp)
    if feats_next.empty: return float(hist.iloc[-1])
    x=feats_next.drop(columns=["y"]).iloc[[-1]].values
    return float(np.expm1(mdl.predict(x))[0])

# --------------- plotting (bands, connected, legend outside, compact metrics) ---------------
def plot_with_bands(title, y_idx, y_vals, train_idx, test_idx, chosen_ser, metrics_row, future_fc, out_png):
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

    # Connected holdout line (chosen model only)
    if chosen_ser is not None and not chosen_ser.empty:
        ser = chosen_ser.sort_index()
        ser_conn = pd.concat([pd.Series([last_train_val], index=[last_train_dt]), ser])
        ax.plot(ser_conn.index, ser_conn.values, lw=2, ls="--", label="Chosen (holdout)", zorder=3)

    # Holdout + Future continuous line
    if chosen_ser is not None and not chosen_ser.empty and future_fc is not None and not future_fc.empty:
        full_future = pd.concat([chosen_ser.sort_index(), future_fc.sort_index()])
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

    # Compact metrics (no Params, no MSE)
    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    header=["Model","MAE","RMSE","MAPE%","MASE","WAPE%","MPE%"]
    if metrics_row is not None:
        n, mae, rmse, mape_v, mase_v, wape_v, mpe_v = metrics_row
        rows=[[n, f"{mae:,.2f}", f"{rmse:,.2f}", f"{mape_v:,.2f}", f"{mase_v:.3f}", f"{wape_v:,.2f}", f"{mpe_v:,.2f}"]]
        tbl=ax2.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.10)

    fig.tight_layout(); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig)

# --------------- intermittent path ---------------
def intermittent_path(y: pd.Series, out_dir: Path, sku: str, desc: str):
    train, test = split_80_20(y)
    cro = croston_sba(train, steps=len(test)); cro.index=test.index
    sn = seasonal_naive(train.values, len(test), SEASON_M)

    # pick best (Croston vs SN)
    cands={"Croston-SBA":cro.values, "Seasonal-Naive":sn}
    best = choose_best_by_metric(cands, test.values, train.values)
    chosen = cro if best=="Croston-SBA" else pd.Series(sn, index=test.index)

    # future = Croston
    future = croston_sba(y, steps=FORECAST_STEPS)

    # metrics row for chosen
    pred = chosen.values
    row = (best,
           mean_absolute_error(test, pred),
           np.sqrt(mean_squared_error(test, pred)),
           mape(test, pred),
           mase(test, pred, train.values, SEASON_M),
           wape(test, pred),
           mpe(test, pred))

    # save outputs
    fname=clean_name(sku)
    future.to_csv(out_dir/f"{fname}_forecast.csv", header=["forecast"])
    comp = pd.concat([
        pd.Series(y.values, index=y.index, name="Actual"),
        chosen.rename("Holdout_Pred"),
        future.rename("Forecast")
    ], axis=1)
    comp.index.name="Month"; comp.to_csv(out_dir/f"{fname}_comparison.csv")

    png=out_dir/f"gb_{fname}.png"
    plot_with_bands(f"Intermittent → {best} — {sku} — {desc}",
                    y.index, y.values, train.index, test.index,
                    chosen, row, future, str(png))
    return row, png.name, best

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--topn", type=int, default=TOP_N)
    ap.add_argument("--min_cov", type=float, default=0.7)
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

    out_dir=OUT_ROOT/clean_name(in_path.stem); out_dir.mkdir(parents=True, exist_ok=True)

    # export top-10 list for reporting
    stats = (mon.groupby(key)["Qty"].agg(total="sum", months="size", nz=lambda s:int((s>0).sum()))
             .reset_index().rename(columns={"nz":"nonzero_months"}))
    stats["coverage"] = stats["nonzero_months"] / stats["months"]
    stats.sort_values(["total","coverage"], ascending=[False,False]).head(args.topn)\
         .to_csv(out_dir/"top10_bestsellers.csv", index=False)

    # choose top-10 for modeling
    top=choose_top_best(mon,key,args.topn,args.min_cov)

    rows=[]
    for sku in top:
        s=mon[mon[key]==sku].copy().sort_values("Date")
        span=pd.date_range(s["Date"].min(), s["Date"].max(), freq="MS")  # all available years
        y=(s.set_index("Date")["Qty"].reindex(span).fillna(0.0).astype(float))
        y=winsorize(y,0.995)
        desc=df[df[key]==sku]["Description"].dropna().iloc[0] if (df[key]==sku).any() else sku

        # very short → SN only
        if len(y)<18:
            train, test = split_80_20(y)
            y_sn = seasonal_naive(train.values, len(test), SEASON_M)
            chosen_ser = pd.Series(y_sn, index=test.index)
            row = ("Seasonal-Naive",
                   mean_absolute_error(test,y_sn),
                   np.sqrt(mean_squared_error(test,y_sn)),
                   mape(test,y_sn),
                   mase(test,y_sn, train.values, SEASON_M),
                   wape(test,y_sn),
                   mpe(test,y_sn))
            future = pd.Series(seasonal_naive(y.values,FORECAST_STEPS,SEASON_M),
                               index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS"))
            fname=clean_name(sku)
            future.to_csv(out_dir/f"{fname}_forecast.csv", header=["forecast"])
            comp = pd.concat([
                pd.Series(y.values, index=y.index, name="Actual"),
                chosen_ser.rename("Holdout_Pred"),
                future.rename("Forecast")
            ], axis=1)
            comp.index.name="Month"; comp.to_csv(out_dir/f"{fname}_comparison.csv")
            png=out_dir/f"gb_{fname}.png"
            plot_with_bands(f"SN (short fallback) — {sku} — {desc}", y.index, y.values, train.index, test.index,
                            chosen_ser, row, future, str(png))
            rows.append({"sku":sku,"description":desc,"chosen":"SN","MASE_WF":np.nan,"MASE_holdout":row[4],"plot":png.name})
            continue

        # intermittent → Croston/SN path
        if is_intermittent(y):
            row, png_name, best = intermittent_path(y, out_dir, sku, desc)
            rows.append({"sku":sku,"description":desc,"chosen":best,"MASE_WF":np.nan,"MASE_holdout":row[4],"plot":png_name})
            continue

        # regular series → GB vs SN vs Blend (80/20)
        train,test,chosen_name,holdout_ser,row,alpha,bias,mdl = train_eval_holdout(y)

        # walk-forward 1-step MASE
        preds=[]; trues=[]
        for i in range(max(24,SEASON_M+1), len(y)):
            hist=y.iloc[:i]
            try:
                preds.append(fit_fn_gb_1step(hist)); trues.append(float(y.iloc[i]))
            except: pass
        wf_mase = mase(np.array(trues), np.array(preds),
                       y.values[:-len(trues)] if trues else y.values, SEASON_M) if trues else np.nan

        # forward 6m forecast using chosen blending recipe
        feats=add_ts_features(y)
        if feats.empty:
            fc = seasonal_naive(y.values, FORECAST_STEPS, SEASON_M)
        else:
            X=feats.drop(columns=["y"]).values
            ytr=np.log1p(np.minimum(feats["y"].values, np.quantile(feats["y"].values,0.995)))
            mdl_full=fit_model_gb(X,ytr)
            hist=y.copy(); preds_ml=[]
            for _ in range(FORECAST_STEPS):
                tmp=add_ts_features(hist)
                if tmp.empty: yhat=hist.iloc[-1]
                else:
                    x=tmp.drop(columns=["y"]).iloc[[-1]].values
                    yhat=max(float(np.expm1(mdl_full.predict(x))[0]),0.0)
                nxt=(hist.index[-1].to_period('M')+1).to_timestamp()
                hist.loc[nxt]=yhat; preds_ml.append(yhat)
            fc_ml=np.array(preds_ml); fc_sn=seasonal_naive(y.values,FORECAST_STEPS,SEASON_M)
            # if Blend won, use its alpha/bias; else if GB won, just fc_ml; if SN won, just fc_sn
            if chosen_name=="Blend":
                fc = alpha*fc_sn + (1-alpha)*fc_ml; fc *= bias
            elif chosen_name=="GradientBoosting":
                fc = fc_ml
            else:
                fc = fc_sn

        future_idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),periods=FORECAST_STEPS,freq="MS")
        future=pd.Series(fc, index=future_idx)

        # save files
        fname=clean_name(sku)
        future.to_csv(out_dir/f"{fname}_forecast.csv", header=["forecast"])
        comp = pd.concat([
            pd.Series(y.values, index=y.index, name="Actual"),
            holdout_ser.rename("Holdout_Pred"),
            future.rename("Forecast")
        ], axis=1)
        comp.index.name="Month"; comp.to_csv(out_dir/f"{fname}_comparison.csv")

        # plot (chosen only) with bands
        png=out_dir/f"gb_{fname}.png"
        plot_with_bands(f"{chosen_name} — 80/20 holdout — {sku} — {desc}",
                        y.index, y.values, train.index, test.index,
                        holdout_ser, row, future, str(png))

        rows.append({"sku":sku,"description":desc,"chosen":chosen_name,
                     "MAE":row[1],"RMSE":row[2],"MAPE%":row[3],"MASE":row[4],"WAPE%":row[5],"MPE%":row[6],
                     "MASE_WF":wf_mase,"alpha":alpha,"bias":bias,"plot":png.name})

    if rows:
        pd.DataFrame(rows).to_csv(OUT_ROOT/clean_name(in_path.stem)/"gb_summary.csv", index=False)
        print(f"✅ Saved → {OUT_ROOT/clean_name(in_path.stem)/'gb_summary.csv'}")

if __name__=="__main__":
    main()
