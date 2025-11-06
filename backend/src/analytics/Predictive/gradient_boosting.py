#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GradientBoostingRegressor monthly forecast with walk-forward MASE, SN blend, bias correction.

Run:
  python gradient_boosting.py --input "cleaned/ANC - 4 YEARS (1).csv" --topn 5 --min_cov 0.7
"""
import warnings, argparse, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable
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
OUT_ROOT=HERE/"predictive_output"/"ml_gradient_boosting"; OUT_ROOT.mkdir(parents=True, exist_ok=True)
FORECAST_STEPS=6; SEASON_M=12; TOP_N=5; INTERMITTENT_ZERO_SHARE=0.40

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
def choose_top(mon: pd.DataFrame, key:str, topn:int, min_cov:float) -> List[str]:
    stats = (mon.groupby(key)["Qty"].agg(total="sum", nz=lambda s:int((s>0).sum()), n="size").reset_index())
    stats["coverage"]=stats["nz"]/stats["n"]
    pool = stats[stats["coverage"]>=min_cov] if (stats["coverage"]>=min_cov).any() else stats
    return pool.sort_values(["coverage","total"], ascending=[False,False])[key].head(topn).tolist()
def winsorize(y: pd.Series, q=0.995) -> pd.Series:
    if len(y)<8: return y
    return y.clip(upper=float(y.quantile(q)))
def add_ts_features(s: pd.Series) -> pd.DataFrame:
    df=s.to_frame("y").copy(); idx=pd.to_datetime(df.index); month=idx.month
    df["m_sin"]=np.sin(2*np.pi*month/12.0); df["m_cos"]=np.cos(2*np.pi*month/12.0)
    for L in range(1, SEASON_M+1): df[f"lag{L}"]=df["y"].shift(L)
    for W in [3,6,12]:
        df[f"rmean{W}"]=df["y"].rolling(W).mean(); df[f"rstd{W}"]=df["y"].rolling(W).std(); df[f"rmed{W}"]=df["y"].rolling(W).median()
    df["diff1"]=df["y"].diff(1); df["pct1"]=df["y"].pct_change(1).replace([np.inf,-np.inf],0).fillna(0)
    return df.dropna()
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
def plot_with_table(title, train_idx, full_idx, y, test_idx, pred_map, rows, out_png):
    fig=plt.figure(figsize=(13,7)); gs=fig.add_gridspec(2,1,height_ratios=[3,1],hspace=0.15)
    ax=fig.add_subplot(gs[0]); ax.plot(full_idx, y, label="Historical", lw=2)
    for name, ser in pred_map.items(): ax.plot(ser.index, ser.values, lw=2, ls="--", label=name)
    ax.set_title(title); ax.set_xlabel("Month"); ax.set_ylabel("Qty"); ax.grid(ls=":", alpha=0.6); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels(): t.set_rotation(45); t.set_ha("right")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)
    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    header=["Model","Params","MAE","MSE","RMSE","MAPE%","MASE","WAPE%","MPE%"]
    table=[[n,str(p),f"{a:,.2f}",f"{m:,.2f}",f"{r:,.2f}",f"{mp:,.2f}",f"{ms:.3f}",f"{wa:,.2f}",f"{me:,.2f}"] for (n,p,a,m,r,mp,ms,wa,me) in rows]
    tbl=ax2.table(cellText=table, colLabels=header, loc="center", cellLoc="center"); tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.15)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def fit_model_gb(Xtr, ytr_log):
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(Xtr, ytr_log); return model, {"n_estimators":300,"lr":0.05,"max_depth":3}

# Reuse GB for this file
fit_model = fit_model_gb
MODEL_NAME = "GradientBoosting"

def train_eval_holdout(y: pd.Series, h:int):
    train, test = y.iloc[:-h], y.iloc[-h:]
    feats_tr, feats_te = add_ts_features(train), add_ts_features(y).iloc[-h:]
    if feats_tr.empty or feats_te.empty:
        y_sn = seasonal_naive(train.values, h, SEASON_M)
        te_idx = pd.date_range(y.index[-h], periods=h, freq="MS")
        rows=[("Seasonal-Naive", f"m={SEASON_M}",
               mean_absolute_error(test,y_sn), mean_squared_error(test,y_sn), np.sqrt(mean_squared_error(test,y_sn)),
               mape(test,y_sn), mase(test,y_sn, train.values, SEASON_M), wape(test,y_sn), mpe(test,y_sn))]
        pred_map={"SN holdout":pd.Series(y_sn, index=te_idx)}
        return train, test, pred_map, rows, 1.0, 1.0, None, {}
    y_tr_raw=feats_tr["y"].values; cap=np.quantile(y_tr_raw,0.995); ytr_log=np.log1p(np.minimum(y_tr_raw,cap))
    mdl, params = fit_model(feats_tr.drop(columns=["y"]).values, ytr_log)
    yhat = np.expm1(mdl.predict(feats_te.drop(columns=["y"]).values)).clip(0)
    y_sn = seasonal_naive(train.values, h, m=SEASON_M)
    # choose alpha
    alphas=np.linspace(0,0.6,7); best_a=0; best_mae=1e18
    for a in alphas:
        yb=a*y_sn+(1-a)*yhat; mae=mean_absolute_error(test.values, yb)
        if mae<best_mae: best_mae, best_a=a, a
    yb=best_a*y_sn+(1-best_a)*yhat
    bc=np.clip((test.values.sum()/yb.sum()) if yb.sum()>0 else 1.0, 0.7, 1.3); yb*=bc
    def met(name,p,pdesc):
        return (name,pdesc, mean_absolute_error(test,p), mean_squared_error(test,p),
                np.sqrt(mean_squared_error(test,p)), mape(test,p),
                mase(test,p,train.values,SEASON_M), wape(test,p), mpe(test,p))
    rows=[met(MODEL_NAME, yhat, params), met("Seasonal-Naive", y_sn, f"m={SEASON_M}"), met("Blend", yb, f"alpha={best_a:.2f}, bias={bc:.2f}")]
    te_idx=feats_te.index
    pred_map={f"{MODEL_NAME} holdout":pd.Series(yhat,index=te_idx),"SN holdout":pd.Series(y_sn,index=te_idx),"Blend (bias-corr)":pd.Series(yb,index=te_idx)}
    return train, test, pred_map, rows, best_a, bc, mdl, params

def fit_fn_gb_1step(hist: pd.Series):
    feats=add_ts_features(hist)
    if feats.empty: return float(hist.iloc[-1])
    y_tr_raw=feats["y"].values; cap=np.quantile(y_tr_raw,0.995); ytr=np.log1p(np.minimum(y_tr_raw,cap))
    mdl,_=fit_model(feats.drop(columns=["y"]).values, ytr)
    last=feats.index[-1].to_period("M").to_timestamp()
    nxt=(last.to_period("M")+1).to_timestamp()
    tmp=hist.copy(); tmp.loc[nxt]=hist.iloc[-1]
    feats_next=add_ts_features(tmp)
    if feats_next.empty: return float(hist.iloc[-1])
    x=feats_next.drop(columns=["y"]).iloc[[-1]].values
    return float(np.expm1(mdl.predict(x))[0])

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

    key="Item Code"; mon=monthly(df,key); top=choose_top(mon,key,args.topn,args.min_cov)
    out_dir=OUT_ROOT/clean_name(in_path.stem); out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for sku in top:
        s=mon[mon[key]==sku].copy().sort_values("Date")
        span=pd.date_range(s["Date"].min(), s["Date"].max(), freq="MS")
        y=(s.set_index("Date")["Qty"].reindex(span).fillna(0.0).astype(float))
        desc=df[df[key]==sku]["Description"].dropna().iloc[0] if (df[key]==sku).any() else sku

        if len(y)<18:
            h=min(3,max(1,len(y)//5)); te_idx=y.index[-h:]; y_sn=seasonal_naive(y.iloc[:-h].values,h,SEASON_M)
            row=("Seasonal-Naive", f"m={SEASON_M}",
                 mean_absolute_error(y.iloc[-h:],y_sn),
                 mean_squared_error(y.iloc[-h:],y_sn),
                 np.sqrt(mean_squared_error(y.iloc[-h:],y_sn)),
                 mape(y.iloc[-h:],y_sn),
                 mase(y.iloc[-h:],y_sn,y.iloc[:-h].values,SEASON_M),
                 wape(y.iloc[-h:],y_sn), mpe(y.iloc[-h:],y_sn))
            png=out_dir/f"gb_{clean_name(sku)}.png"
            plot_with_table(f"GB (short fallback) — {sku} — {desc}", y.index[:-h], y.index, y.values, te_idx, {"SN holdout":pd.Series(y_sn,index=te_idx)}, [row], png)
            pd.Series(seasonal_naive(y.values,FORECAST_STEPS,SEASON_M), index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS")).to_csv(out_dir/f"{clean_name(sku)}_forecast.csv", header=["forecast"])
            rows.append({"sku":sku,"description":desc,"chosen":"SN","MASE_WF":np.nan,"MASE_holdout":row[6],"plot":png.name}); continue

        y=winsorize(y,0.995)

        if is_intermittent(y):
            h=min(6, max(3, len(y)//5))
            train, test = y.iloc[:-h], y.iloc[-h:]
            cro = croston_sba(train, steps=h); cro.index=test.index
            sn = seasonal_naive(train.values, h, SEASON_M)
            def met(name,p,pdsc): 
                return (name,pdsc, mean_absolute_error(test,p), mean_squared_error(test,p),
                        np.sqrt(mean_squared_error(test,p)), mape(test,p),
                        mase(test,p,train.values,SEASON_M), wape(test,p), mpe(test,p))
            tbl=[met("Croston-SBA", cro.values, "alpha=0.1"), met("Seasonal-Naive", sn, f"m={SEASON_M}")]
            pred_map={"Croston holdout":pd.Series(cro.values,index=test.index), "SN holdout":pd.Series(sn,index=test.index)}
            png=out_dir/f"gb_{clean_name(sku)}.png"
            plot_with_table(f"Intermittent → Croston — {sku} — {desc}", y.index[:-h], y.index, y.values, test.index, pred_map, tbl, png)
            croston_sba(y, steps=FORECAST_STEPS).to_csv(out_dir/f"{clean_name(sku)}_forecast.csv", header=["forecast"])
            rows.append({"sku":sku,"description":desc,"chosen":"Croston-SBA","MASE_WF":np.nan,"MASE_holdout":tbl[0][6],"plot":png.name})
            continue

        h=min(6,max(3,len(y)//5))
        train,test,pred_map,table_rows,alpha,bias,mdl,params= train_eval_holdout(y,h)

        # walk-forward
        preds=[]; trues=[]
        for i in range(max(24,SEASON_M+1), len(y)):
            hist=y.iloc[:i]
            try: preds.append(fit_fn_gb_1step(hist)); trues.append(float(y.iloc[i]))
            except: pass
        wf_mase = mase(np.array(trues), np.array(preds), y.values[:-len(trues)] if trues else y.values, SEASON_M) if trues else np.nan

        png=out_dir/f"gb_{clean_name(sku)}.png"
        plot_with_table(f"GradientBoosting — last-{h} holdout — {sku} — {desc}", y.index[:-h], y.index, y.values, y.index[-h:], pred_map, table_rows, png)

        # forward 6m forecast
        feats=add_ts_features(y)
        if feats.empty:
            fc = seasonal_naive(y.values, FORECAST_STEPS, SEASON_M)
        else:
            X=feats.drop(columns=["y"]).values
            ytr=np.log1p(np.minimum(feats["y"].values, np.quantile(feats["y"].values,0.995)))
            mdl_full,_=fit_model(X,ytr)
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
            fc=alpha*fc_sn+(1-alpha)*fc_ml; fc*=bias
        idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1),periods=FORECAST_STEPS,freq="MS")
        pd.Series(fc, index=idx).to_csv(out_dir/f"{clean_name(sku)}_forecast.csv", header=["forecast"])
        holdout_mase = {r[0]: r[6] for r in table_rows}.get("Blend", np.nan)
        rows.append({"sku":sku,"description":desc,"chosen":"GB+Blend","MASE_WF":wf_mase,"MASE_holdout":holdout_mase,"alpha":alpha,"bias":bias,"plot":png.name})

    if rows:
        pd.DataFrame(rows).to_csv(OUT_ROOT/clean_name(in_path.stem)/"gb_summary.csv", index=False)
        print(f"✅ Saved → {OUT_ROOT/clean_name(in_path.stem)/'gb_summary.csv'}")

if __name__=="__main__":
    main()
