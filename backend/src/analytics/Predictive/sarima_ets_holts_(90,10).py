#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
90/10 holdout with calibrated forecasts
Same toolkit as 80/20: ETS/SARIMA/SARIMAX/Holt-Winters + Croston, recent-window, winsor, log1p, calibration.
"""
import argparse, warnings, re, sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import matplotlib as mpl
mpl.rcParams.update({"figure.dpi":200,"axes.titlesize":16,"axes.labelsize":12,
                     "legend.fontsize":10,"xtick.labelsize":10,"ytick.labelsize":10})

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_ROOT = HERE / "ts_sarima-ets-holts(90,10)"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEASONAL_PERIODS=12; DEFAULT_STEPS=6; DEFAULT_TOPN=5

# --- copy helpers from 80/20 (short) ---
def parse_dates_safe(s):
    try: return pd.to_datetime(s, errors="coerce")
    except: return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df):
    lower={c.lower().strip():c for c in df.columns}
    aliases={"date":["date","transaction date","sales date","receipt date","order date","invoice date","trans date","posting date"],
             "description":["description","item name","product","product name","name","item","desc"],
             "qty":["qty","quantity","qty sold","units","units sold","quantity sold","sales qty","sold qty","sale qty","qnt"]}
    def pick(k):
        if k in lower: return lower[k]
        for a in aliases[k]:
            if a in lower: return lower[a]
        if k=="date":
            for c in df.columns:
                try:
                    if pd.to_datetime(df[c], errors="coerce").notna().sum()>=10: return c
                except: pass
        return None
    d=pick("date"); desc=pick("description"); q=pick("qty")
    if not(d and desc and q): raise ValueError("Cannot detect Date/Description/Qty.")
    return {d:"Date", desc:"Description", q:"Qty"}

def load_monthly(p: Path):
    df = pd.read_excel(p) if p.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(p, low_memory=False)
    df = df.rename(columns=detect_columns(df)); df["Date"]=parse_dates_safe(df["Date"]); df=df.dropna(subset=["Date"])
    df["Description"]=df["Description"].astype(str).str.strip().str.replace(r"\s+"," ", regex=True)
    df["Qty"]=pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(float)
    return (df.set_index("Date").groupby("Description")["Qty"].resample("MS").sum().reset_index())

def series_for(monthly, desc): 
    return (monthly[monthly["Description"]==desc].set_index("Date")["Qty"].asfreq("MS", fill_value=0.0).astype(float))

def top_by_frequency(monthly, n): 
    f=monthly.groupby("Description")["Qty"].apply(lambda s:(s>0).sum()); return f.sort_values(ascending=False).head(n).index.tolist()

def winsorize_series(y,q): 
    if len(y)<6: return y
    return y.clip(upper=y.quantile(q))

def mape_safe(yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float); m=yt!=0
    return float(np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100) if m.any() else np.nan
def wape(yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float); den=np.sum(np.abs(yt))
    return float(np.sum(np.abs(yt-yp))/den*100) if den>0 else np.nan
def mpe(yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float); m=yt!=0
    return float(np.mean(((yp[m]-yt[m])/yt[m]))*100) if m.any() else np.nan
def rmse(yt, yp): 
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(yt, yp)))
def mase(y_true, y_pred, insample, m=12):
    ins=np.asarray(insample,float)
    den=np.mean(np.abs(ins[m:]-ins[:-m])) if len(ins)>m else (np.mean(np.abs(np.diff(ins))) if len(ins)>1 else 1.0)
    den=1.0 if den==0 else den
    return float(np.mean(np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float)))/den)
def is_intermittent(y, thr=0.40): return (y==0).mean()>=thr
def clip_nonneg(s: pd.Series)->pd.Series: return s.clip(lower=0)

def croston_sba(y, alpha=0.1, steps=DEFAULT_STEPS):
    yv=y.values.astype(float); z=None; p=None; k=0
    for val in yv:
        k+=1
        if val>0:
            z = val if z is None else z + alpha*(val-z)
            p = k   if p is None else p + alpha*(k-p)
            k=0
    fc=np.zeros(steps) if z is None else np.array([ (z/max(p,1e-8))*(1-alpha/2.0) ]*steps)
    idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(fc, index=idx)

def seasonal_naive_forecast(y, m, steps):
    last_idx = y.index[-1]
    idx = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    fc = [y.iloc[-m + h] for h in range(steps)]
    return clip_nonneg(pd.Series(fc, index=idx))

def month_dummies(idx): d=pd.get_dummies(idx.month); d.index=idx; d.columns=[f"m{c:02d}" for c in d.columns]; return d
def future_month_dummies(last_idx, steps): 
    idx=pd.date_range(last_idx+pd.offsets.MonthBegin(1), periods=steps, freq="MS"); return month_dummies(idx)

def fit_ets_grid(train, m, use_log=True):
    cfg=[("add","add",False,None),("add","add",True,None),("add","mul",False,None),("add","mul",True,None),
         (None,"add",False,None),(None,"mul",False,None),("add","add",False,"log"),("add","mul",False,"log")]
    best, best_aic, params=None, float("inf"), None
    y=np.log1p(train) if use_log else train
    for trend,seas,damped,boxcox in cfg:
        seasonal=seas is not None and len(train)>=24
        try:
            mdl=ExponentialSmoothing(y, trend=trend, seasonal=(seas if seasonal else None),
                                     seasonal_periods=(m if seasonal else None),
                                     damped_trend=(damped if trend else False),
                                     initialization_method="estimated", use_boxcox=boxcox)
            fit=mdl.fit(optimized=True, remove_bias=True); aic=getattr(fit,"aic",np.inf)
            if aic<best_aic: best, best_aic, params=fit, aic, (trend,seas,damped,"log1p" if use_log else None)
        except: pass
    return best, params

def ets_forecast(fit, steps, last_idx, use_log=True):
    fc=fit.forecast(steps); fc=np.expm1(fc) if use_log else fc
    fc.index=pd.date_range(last_idx+pd.offsets.MonthBegin(1), periods=steps, freq="MS"); return clip_nonneg(fc)

def fit_sarima_grid(train, m, use_log=True):
    pdq=[(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1),(2,1,2)]
    PDQ=[(0,1,0,m),(0,1,1,m),(1,1,0,m),(1,1,1,m)]
    best, best_aic, params=None, float("inf"), None
    y=np.log1p(train) if use_log else train
    for o in pdq:
        for s in PDQ:
            try:
                fit= SARIMAX(y, order=o, seasonal_order=s, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if fit.aic<best_aic: best, best_aic, params=fit, fit.aic, (o,s,"log1p" if use_log else None)
            except: pass
    return best, params

def sarima_forecast(fit, steps, last_idx, use_log=True):
    fc=fit.forecast(steps); fc=np.expm1(fc) if use_log else fc
    fc.index=pd.date_range(last_idx+pd.offsets.MonthBegin(1), periods=steps, freq="MS"); return clip_nonneg(fc)

def fit_sarimax_month_dummies(train, m, use_log=True):
    ex=month_dummies(train.index); pdq=[(0,1,1),(1,1,1),(1,0,1)]; PDQ=[(0,1,1,m),(1,1,1,m),(0,1,0,m)]
    best, best_aic, params=None, float("inf"), None
    y=np.log1p(train) if use_log else train
    for o in pdq:
        for s in PDQ:
            try:
                fit= SARIMAX(y, exog=ex, order=o, seasonal_order=s, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if fit.aic<best_aic: best, best_aic, params=fit, fit.aic, (o,s,"month_dummies","log1p" if use_log else None)
            except: pass
    return best, params

def sarimax_forecast(fit, last_idx, steps, use_log=True):
    exf=future_month_dummies(last_idx, steps); fc=fit.forecast(steps, exog=exf); fc=np.expm1(fc) if use_log else fc
    fc.index=exf.index; return clip_nonneg(fc)

def calibrate(y_true, y_pred, lo=0.3, hi=3.0):
    yp=np.maximum(np.asarray(y_pred,float), 1e-8); r=np.asarray(y_true,float)/yp
    r=np.clip(r, lo, hi); return float(np.median(r))

def plot_pretty(desc, train, test, preds_map, metrics, out_png):
    fig=plt.figure(figsize=(13,7)); gs=fig.add_gridspec(2,1,height_ratios=[3,1],hspace=0.15)
    ax=fig.add_subplot(gs[0])
    ax.plot(train.index, train.values, label="Training Data", lw=2)
    ax.plot(test.index,  test.values,  label="Test Data", color="black", lw=2)
    pal={"ETS":"tab:orange","SARIMA":"tab:green","SARIMAX":"tab:red","Holt-Winters":"tab:purple","Croston-SBA":"tab:cyan"}
    for n,s in preds_map.items(): ax.plot(s.index, s.values, lw=2, ls="--", color=pal.get(n,None), label=f"{n} Forecast")
    ax.grid(True, ls=":", alpha=0.6); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels(): t.set_rotation(45); t.set_ha("right")
    ax.set_title(f"SARIMA/ETS/SARIMAX/Holt-Winters — 90% Train, 10% Test — {desc}")
    ax.set_xlabel("Month"); ax.set_ylabel("Qty"); ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)
    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    header=["Model","Params","MAE","MSE","RMSE","MAPE%","MASE","WAPE%","MPE%"]
    table=[[n,str(p),f"{a:,.2f}",f"{m:,.2f}",f"{r:,.2f}",f"{mp:,.2f}",f"{ms:,.3f}",f"{wa:,.2f}",f"{me:,.2f}"] for (n,p,a,m,r,mp,ms,wa,me) in metrics]
    tbl=ax2.table(cellText=table, colLabels=header, loc="center", cellLoc="center"); tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.15)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def sanitize(x:str)->str: return re.sub(r"[^a-z0-9]+","_", x.lower()).strip("_")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--top", type=int, default=DEFAULT_TOPN)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--recent", type=int, default=24)
    ap.add_argument("--winsor", type=float, default=0.98)
    ap.add_argument("--no-log", dest="use_log", action="store_false"); ap.set_defaults(use_log=True)
    ap.add_argument("--no-calibrate", dest="do_cal", action="store_false"); ap.set_defaults(do_cal=True)
    args=ap.parse_args()

    files=[Path(args.file)] if args.file else (sorted(list(DATA_DIR.glob("*.csv")))+sorted(list(DATA_DIR.glob("*.xlsx"))))
    if not files: print(f"No files in {DATA_DIR}"); sys.exit(0)

    for f in files:
        try: monthly=load_monthly(f)
        except Exception as e: print(f"[skip] {f.name}: {e}"); continue

        for desc in top_by_frequency(monthly, args.top):
            y=series_for(monthly, desc)
            if len(y)<12: continue
            y=winsorize_series(y, args.winsor)

            n=len(y); k=max(1,int(round(n*0.10)))
            train_full, test = y.iloc[:-k], y.iloc[-k:]
            train = train_full.iloc[-args.recent:] if len(train_full)>args.recent else train_full

            preds_map={}; metrics=[]

            if is_intermittent(y):
                cro=croston_sba(train, steps=len(test)); cro.index=test.index
                preds_map["Croston-SBA"]=cro
                metrics.append(("Croston-SBA","alpha=0.1",
                    mean_absolute_error(test,cro), mean_squared_error(test,cro), rmse(test,cro),
                    mape_safe(test.values,cro.values), mase(test.values,cro.values,train.values,SEASONAL_PERIODS),
                    wape(test.values,cro.values), mpe(test.values,cro.values)))
                fc6=croston_sba(y, steps=args.steps)
                if args.do_cal: fc6 = fc6 * calibrate(test.values, cro.values)
                fc6.to_csv(OUT_ROOT/f"{sanitize(desc)}_forecast.csv", header=["forecast"])
            else:
                efit, ecfg = fit_ets_grid(train, SEASONAL_PERIODS, args.use_log)
                if efit is not None:
                    ep=ets_forecast(efit, len(test), train.index[-1], args.use_log); ep.index=test.index
                    preds_map["ETS"]=ep
                    metrics.append(("ETS", str(ecfg),
                        mean_absolute_error(test,ep), mean_squared_error(test,ep), rmse(test,ep),
                        mape_safe(test.values,ep.values), mase(test.values,ep.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,ep.values), mpe(test.values,ep.values)))
                sfit, scfg = fit_sarima_grid(train, SEASONAL_PERIODS, args.use_log)
                if sfit is not None:
                    sp=sarima_forecast(sfit, len(test), train.index[-1], args.use_log); sp.index=test.index
                    preds_map["SARIMA"]=sp
                    metrics.append(("SARIMA", str(scfg),
                        mean_absolute_error(test,sp), mean_squared_error(test,sp), rmse(test,sp),
                        mape_safe(test.values,sp.values), mase(test.values,sp.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,sp.values), mpe(test.values,sp.values)))
                try:
                    seasonal=len(train)>=24
                    hw=ExponentialSmoothing(np.log1p(train) if args.use_log else train,
                                            trend="add", seasonal=("add" if seasonal else None),
                                            seasonal_periods=(SEASONAL_PERIODS if seasonal else None),
                                            damped_trend=True, initialization_method="estimated").fit(optimized=True, remove_bias=True)
                    hp=hw.forecast(len(test)); hp=np.expm1(hp) if args.use_log else hp; hp.index=test.index
                    preds_map["Holt-Winters"]=clip_nonneg(hp)
                    metrics.append(("Holt-Winters","add+add+damped",
                        mean_absolute_error(test,hp), mean_squared_error(test,hp), rmse(test,hp),
                        mape_safe(test.values,hp.values), mase(test.values,hp.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,hp.values), mpe(test.values,hp.values)))
                except: pass
                xfit, xcfg = fit_sarimax_month_dummies(train, SEASONAL_PERIODS, args.use_log)
                if xfit is not None:
                    xp=sarimax_forecast(xfit, train.index[-1], len(test), args.use_log); xp.index=test.index
                    preds_map["SARIMAX"]=xp
                    metrics.append(("SARIMAX", str(xcfg),
                        mean_absolute_error(test,xp), mean_squared_error(test,xp), rmse(test,xp),
                        mape_safe(test.values,xp.values), mase(test.values,xp.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,xp.values), mpe(test.values,xp.values)))

                if metrics:
                    best = sorted(metrics, key=lambda r: np.nan_to_num(r[6], nan=1e9))[0][0]
                    if   best=="ETS" and efit is not None: fc6=ets_forecast(efit, args.steps, y.index[-1], args.use_log)
                    elif best=="SARIMA" and sfit is not None: fc6=sarima_forecast(sfit, args.steps, y.index[-1], args.use_log)
                    elif best=="SARIMAX" and xfit is not None: fc6=sarimax_forecast(xfit, y.index[-1], args.steps, args.use_log)
                    elif best=="Holt-Winters":
                        tmp=hw.forecast(args.steps); tmp=np.expm1(tmp) if args.use_log else tmp
                        idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=args.steps, freq="MS")
                        fc6=clip_nonneg(pd.Series(tmp.values, index=idx))
                    else: fc6 = seasonal_naive_forecast(y, SEASONAL_PERIODS, args.steps)
                    if args.do_cal and best in preds_map:
                        fc6 = fc6 * calibrate(test.values, preds_map[best].values)
                    fc6.to_csv(OUT_ROOT/f"{sanitize(desc)}_forecast.csv", header=["forecast"])

            plot_pretty(desc, train, test, preds_map, metrics, OUT_ROOT/f"{sanitize(desc)}_90_10.png")
    print(f"Done. See: {OUT_ROOT}")

if __name__ == "__main__":
    main()
