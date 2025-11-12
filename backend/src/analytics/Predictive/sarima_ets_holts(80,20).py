import argparse, warnings, re, sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=ConvergenceWarning)
mpl.rcParams.update({
    "figure.dpi": 220,
    "axes.titlesize": 16, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10
})

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_ROOT = HERE / "ts_sarima-ets-holts(80,20)_v3"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEASONAL_PERIODS = 12
DEFAULT_STEPS = 6
DEFAULT_TOPN = 10
INTERMITTENT_ZERO_SHARE = 0.40

# ----------------- helpers -----------------
def parse_dates_safe(s):
    try: return pd.to_datetime(s, errors="coerce")
    except: return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df):
    lower = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "date": ["date","transaction date","sales date","receipt date","order date","invoice date","trans date","posting date"],
        "description": ["description","item name","product","product name","name","item","desc"],
        "qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty","sold qty","sale qty","qnt","qnty"]
    }
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

def load_monthly(p: Path) -> pd.DataFrame:
    df = pd.read_excel(p) if p.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(p, low_memory=False)
    df = df.rename(columns=detect_columns(df))
    df["Date"] = parse_dates_safe(df["Date"]); df = df.dropna(subset=["Date"])
    df["Description"] = df["Description"].astype(str).str.strip().str.replace(r"\s+"," ", regex=True)
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(float)
    return (df.set_index("Date").groupby("Description")["Qty"].resample("MS").sum().reset_index())

def series_for(monthly, desc):
    return (monthly[monthly["Description"]==desc]
            .set_index("Date")["Qty"].asfreq("MS", fill_value=0.0).astype(float))

def top10_bestsellers(monthly: pd.DataFrame, n: int) -> pd.DataFrame:
    g = (monthly.groupby("Description")["Qty"]
         .agg(total_qty="sum", nz=lambda s:int((s>0).sum()), n="size")
         .reset_index())
    g["coverage"] = g["nz"]/g["n"]
    return g.sort_values(["total_qty","coverage"], ascending=[False, False]).head(n).reset_index(drop=True)

# ---------- robust transforms ----------
def stl_dampen(y: pd.Series, m:int=SEASONAL_PERIODS, k:float=3.5) -> pd.Series:
    """Dampen spikes via STL residuals using MAD threshold."""
    if len(y) < max(2*m, 24):  # too short for STL
        return y
    res = STL(y, period=m, robust=True).fit()
    resid = y - (res.trend + res.seasonal)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-8
    hi = np.median(resid) + k*1.4826*mad
    y_adj = y.copy()
    # cap positive spikes relative to (trend+seasonal)
    cap = (res.trend + res.seasonal) + hi
    y_adj = np.minimum(y_adj, cap)
    return y_adj.clip(lower=0)

def winsorize_series(y, q):
    if len(y)<6: return y
    return y.clip(upper=y.quantile(q))

def choose_log1p(y: pd.Series) -> bool:
    if y.max() <= 0: return False
    # use simple skew heuristic
    sk = pd.Series(y.values + 1e-8).skew()
    return bool(sk > 1.0)

# ---------- metrics ----------
def mape_safe(yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float); den=np.where(yt==0,1,yt)
    return float(np.mean(np.abs((yt-yp)/den))*100)

def wape(yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float); den=np.sum(np.abs(yt))
    return float(np.sum(np.abs(yt-yp))/den*100) if den>0 else np.nan

def mpe(yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float); m=yt!=0
    return float(np.mean(((yp[m]-yt[m])/yt[m]))*100) if m.any() else np.nan

def rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))
def mase(y_true, y_pred, insample, m=SEASONAL_PERIODS):
    ins=np.asarray(insample,float)
    if len(ins)<=m: den=np.mean(np.abs(np.diff(ins))) if len(ins)>1 else 1.0
    else: den=np.mean(np.abs(ins[m:]-ins[:-m]))
    if den==0: den=1.0
    return float(np.mean(np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float)))/den)

def is_intermittent(y, thr=INTERMITTENT_ZERO_SHARE): return (y==0).mean()>=thr
def clip_nonneg(s: pd.Series)->pd.Series: return s.clip(lower=0)

# ---------- baselines ----------
def seasonal_naive_forecast(y, m, steps):
    last_idx = y.index[-1]
    idx = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    if len(y)<m:
        last=float(y.iloc[-1]) if len(y) else 0.0
        return clip_nonneg(pd.Series(np.repeat(last, steps), index=idx))
    pat = y.iloc[-m:].values
    reps = int(np.ceil(steps/m))
    fc = np.tile(pat, reps)[:steps]
    return clip_nonneg(pd.Series(fc, index=idx))

# Croston-SBA
def croston_sba(y, alpha=0.1, steps=DEFAULT_STEPS):
    yv=y.values.astype(float); z=None; p=None; k=0
    for val in yv:
        k+=1
        if val>0:
            z = val if z is None else z + alpha*(val-z)
            p = k   if p is None else p + alpha*(k-p)
            k=0
    fc = np.zeros(steps) if z is None else np.array([ (z/max(p,1e-8))*(1-alpha/2.0) ]*steps)
    idx = pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(fc, index=idx)

# ---------- classical models ----------
def fit_ets_grid(train, m, use_log=True):
    cfg=[("add","add",False),("add","add",True),("add","mul",False),("add","mul",True),
         (None,"add",False),(None,"mul",False)]
    best_fit, best_aic, best_params = None, np.inf, None
    y = np.log1p(train) if use_log else train
    for trend, seas, damped in cfg:
        seasonal = seas is not None and len(train)>=24
        try:
            mdl = ExponentialSmoothing(
                y,
                trend=trend,
                seasonal=(seas if seasonal else None),
                seasonal_periods=(m if seasonal else None),
                damped_trend=(damped if trend else False),
                initialization_method="estimated"
            )
            fit = mdl.fit(optimized=True, remove_bias=True)
            aic = getattr(fit, "aic", np.inf)
            if aic < best_aic:
                best_fit, best_aic, best_params = fit, aic, (trend,seas,damped)
        except: pass
    return best_fit, best_params

def ets_forecast(fit, steps, last_idx, use_log=True):
    fc=fit.forecast(steps); fc=np.expm1(fc) if use_log else fc
    fc.index=pd.date_range(last_idx+pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return clip_nonneg(fc)

def fit_sarima_grid(train, m, use_log=True):
    pdq=[(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1),(2,1,2)]
    PDQ=[(0,1,0,m),(0,1,1,m),(1,1,0,m),(1,1,1,m),(2,1,1,m)]
    best_fit, best_aic, best_params = None, np.inf, None
    y = np.log1p(train) if use_log else train
    for o in pdq:
        for s in PDQ:
            try:
                fit = SARIMAX(y, order=o, seasonal_order=s, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if fit.aic < best_aic:
                    best_fit, best_aic, best_params = fit, fit.aic, (o,s)
            except: pass
    return best_fit, best_params

def sarima_forecast(fit, steps, last_idx, use_log=True):
    fc=fit.forecast(steps); fc=np.expm1(fc) if use_log else fc
    fc.index=pd.date_range(last_idx+pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return clip_nonneg(fc)

def month_dummies(idx):
    d=pd.get_dummies(idx.month); d.index=idx; d.columns=[f"m{c:02d}" for c in d.columns]; return d

def future_month_dummies(last_idx, steps):
    idx=pd.date_range(last_idx+pd.offsets.MonthBegin(1), periods=steps, freq="MS"); return month_dummies(idx)

def fit_sarimax_month_dummies(train, m, use_log=True):
    ex=month_dummies(train.index)
    pdq=[(0,1,1),(1,1,1),(1,0,1)]
    PDQ=[(0,1,1,m),(1,1,1,m),(0,1,0,m)]
    best_fit, best_aic, best_params = None, np.inf, None
    y = np.log1p(train) if use_log else train
    for o in pdq:
        for s in PDQ:
            try:
                fit = SARIMAX(y, exog=ex, order=o, seasonal_order=s, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if fit.aic < best_aic:
                    best_fit, best_aic, best_params = fit, fit.aic, (o,s,"month_dummies")
            except: pass
    return best_fit, best_params

def sarimax_forecast(fit, last_idx, steps, use_log=True):
    exf=future_month_dummies(last_idx, steps)
    fc=fit.forecast(steps, exog=exf); fc=np.expm1(fc) if use_log else fc
    fc.index=exf.index; return clip_nonneg(fc)

# ---------- calibration ----------
def calibrate(y_true, y_pred, lo=0.7, hi=1.3):
    yp=np.maximum(np.asarray(y_pred,float), 1e-8)
    r=np.asarray(y_true,float)/yp; r=np.clip(r, lo, hi)
    return float(np.median(r))

# ---------- CV ----------
def rolling_origin_cv(y: pd.Series, models: List[str], use_log: bool, k:int=3, min_train:int=24
                     ) -> Dict[str, Tuple[float,float,float]]:
    """
    Returns per-model (mean_MASE, mean_WAPE, mean_RMSE) over k expanding splits
    on the most recent portion of series.
    """
    results = {m: [] for m in models}
    n = len(y)
    if n < (min_train + k):  # not enough length
        return {m: (np.inf, np.inf, np.inf) for m in models}

    # define split points near the end
    # ensure each test block is ~max(3, n//(8)) months
    test_len = max(3, n // 8)
    start = n - (k * test_len)
    if start < min_train:  # adjust if too short
        start = min_train

    for i in range(k):
        split = start + i*test_len
        train = y.iloc[:split]
        test  = y.iloc[split: split+test_len]
        if len(test) < 3 or len(train) < min_train: 
            continue

        # Intermittent: only compare Croston vs SN here
        if is_intermittent(y):
            cro = croston_sba(train, steps=len(test)); cro.index=test.index
            sn  = seasonal_naive_forecast(train, SEASONAL_PERIODS, len(test)); sn.index=test.index
            for name, pred in [("Croston-SBA", cro), ("Seasonal-Naive", sn)]:
                mae = mean_absolute_error(test, pred)
                mr  = rmse(test, pred)
                ms  = mase(test, pred, train.values, SEASONAL_PERIODS)
                wa  = wape(test, pred)
                results[name].append((ms, wa, mr))
            continue

        # Classical models:
        # decide transform by skew on this train slice
        use_log_train = use_log and choose_log1p(train)
        # ETS
        efit, _ = fit_ets_grid(train, SEASONAL_PERIODS, use_log_train)
        if efit is not None:
            ep = ets_forecast(efit, len(test), train.index[-1], use_log_train); ep.index=test.index
            results["ETS"].append((mase(test, ep, train.values, SEASONAL_PERIODS), wape(test, ep), rmse(test, ep)))
        # SARIMA
        sfit, _ = fit_sarima_grid(train, SEASONAL_PERIODS, use_log_train)
        if sfit is not None:
            sp = sarima_forecast(sfit, len(test), train.index[-1], use_log_train); sp.index=test.index
            results["SARIMA"].append((mase(test, sp, train.values, SEASONAL_PERIODS), wape(test, sp), rmse(test, sp)))
        # HW
        try:
            seasonal = len(train) >= 24
            hw = ExponentialSmoothing(np.log1p(train) if use_log_train else train,
                                      trend="add",
                                      seasonal=("add" if seasonal else None),
                                      seasonal_periods=(SEASONAL_PERIODS if seasonal else None),
                                      damped_trend=True,
                                      initialization_method="estimated").fit(optimized=True, remove_bias=True)
            hp = hw.forecast(len(test)); hp = np.expm1(hp) if use_log_train else hp; hp.index=test.index
            hp = clip_nonneg(hp)
            results["Holt-Winters"].append((mase(test, hp, train.values, SEASONAL_PERIODS), wape(test, hp), rmse(test, hp)))
        except: pass
        # SARIMAX
        xfit, _ = fit_sarimax_month_dummies(train, SEASONAL_PERIODS, use_log_train)
        if xfit is not None:
            xp = sarimax_forecast(xfit, train.index[-1], len(test), use_log_train); xp.index=test.index
            results["SARIMAX"].append((mase(test, xp, train.values, SEASONAL_PERIODS), wape(test, xp), rmse(test, xp)))
        # SN baseline for reference
        sn = seasonal_naive_forecast(train, SEASONAL_PERIODS, len(test)); sn.index=test.index
        results["Seasonal-Naive"].append((mase(test, sn, train.values, SEASONAL_PERIODS), wape(test, sn), rmse(test, sn)))

    # aggregate means
    out={}
    for m, arr in results.items():
        if arr:
            a=np.array(arr, float)
            out[m]=(float(np.nanmean(a[:,0])), float(np.nanmean(a[:,1])), float(np.nanmean(a[:,2])))
        else:
            out[m]=(np.inf, np.inf, np.inf)
    return out

# ---------- plotting ----------
def plot_bands_connected(desc, y, train, test, holdout_map, selected_line, future, metrics_rows, out_png):
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs  = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0])
    ax  = fig.add_subplot(gs[0])

    ax.plot(y.index, y.values, label="Historical (Actual)", lw=2, color="black")

    if len(train):
        ax.axvspan(train.index[0], train.index[-1], color="tab:blue", alpha=0.06, label="PAST (Train)")
    if len(test):
        ax.axvspan(test.index[0], test.index[-1], color="tab:orange", alpha=0.08, label="PRESENT (Holdout)")
    if future is not None and not future.empty:
        ax.axvspan(future.index[0], future.index[-1], color="tab:green", alpha=0.07, label="FUTURE (Forecast)")

    for name, ser in (holdout_map or {}).items():
        ax.plot(ser.index, ser.values, lw=2, ls="--", label=f"{name} (holdout)")

    if selected_line is not None and not selected_line.empty:
        ax.plot(selected_line.index, selected_line.values, lw=2.6, label="Selected Forecast (connected)")

    ax.set_title(f"ETS / SARIMA / SARIMAX / Holt-Winters — 80/20 + CV — {desc}")
    ax.set_xlabel("Month"); ax.set_ylabel("Qty")
    ax.grid(ls=":", alpha=0.6); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels(): t.set_rotation(45); t.set_ha("right")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)

    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    header=["Model","MAE","RMSE","MAPE%","MASE","WAPE%","MPE%"]
    table=[[n,f"{a:,.2f}",f"{r:,.2f}",f"{mp:,.2f}",f"{ms:.3f}",f"{wa:,.2f}",f"{me:,.2f}"]
           for (n,a,r,mp,ms,wa,me) in metrics_rows]
    if table:
        tbl=ax2.table(cellText=table, colLabels=header, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.04)

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def sanitize(x:str)->str: return re.sub(r"[^a-z0-9]+","_", x.lower()).strip("_")

# -------------- main --------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--top", type=int, default=DEFAULT_TOPN)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--winsor", type=float, default=0.98)
    ap.add_argument("--recent", type=int, default=36, help="cap training to last N months for fit speed and stability")
    ap.add_argument("--cv_splits", type=int, default=3, help="rolling-origin CV splits")
    ap.add_argument("--no_log", dest="use_log", action="store_false"); ap.set_defaults(use_log=True)
    args=ap.parse_args()

    files=[Path(args.file)] if args.file else (sorted(list(DATA_DIR.glob("*.csv")))+sorted(list(DATA_DIR.glob("*.xlsx"))))
    if not files: print(f"No files in {DATA_DIR}"); sys.exit(0)

    for f in files:
        try: monthly=load_monthly(f)
        except Exception as e: print(f"[skip] {f.name}: {e}"); continue

        # Export Top-10 bestsellers (coverage-aware)
        top_tbl = top10_bestsellers(monthly, args.top)
        out_dir = OUT_ROOT / sanitize(f.stem); out_dir.mkdir(parents=True, exist_ok=True)
        top_tbl.to_csv(out_dir/"top10_bestsellers.csv", index=False)

        rows=[]
        for desc in top_tbl["Description"]:
            y_raw = series_for(monthly, desc)
            if len(y_raw)<12: continue

            # robust pre-processing
            y = stl_dampen(y_raw, m=SEASONAL_PERIODS)
            y = winsorize_series(y, args.winsor)
            use_log = args.use_log and choose_log1p(y)

            # 80/20 split
            n=len(y); h=max(1, int(round(n*0.20)))
            train_full, test = y.iloc[:-h], y.iloc[-h:]
            train = train_full.iloc[-args.recent:] if len(train_full)>args.recent else train_full

            holdout_map = {}
            metrics = []

            # Intermittent path
            if is_intermittent(y):
                cro=croston_sba(train, steps=len(test)); cro.index=test.index
                sn = seasonal_naive_forecast(train, SEASONAL_PERIODS, len(test)); sn.index=test.index

                def met(name, p):
                    arr=p.values if isinstance(p,pd.Series) else p
                    return (name,
                            mean_absolute_error(test,arr),
                            rmse(test,arr),
                            mape_safe(test.values,arr),
                            mase(test.values,arr,train.values,SEASONAL_PERIODS),
                            wape(test.values,arr),
                            mpe(test.values,arr))
                metrics.append(met("Croston-SBA", cro))
                metrics.append(met("Seasonal-Naive", sn))
                holdout_map["Croston-SBA"]=cro
                holdout_map["Seasonal-Naive"]=sn

                # model selection with CV (Croston vs SN)
                cv_stats = rolling_origin_cv(y, ["Croston-SBA","Seasonal-Naive"], use_log, k=args.cv_splits, min_train=24)
                key = lambda kv: (kv[1][0], kv[1][1], kv[1][2])  # mean MASE, WAPE, RMSE
                best_name = sorted(cv_stats.items(), key=key)[0][0]

                future = croston_sba(y, steps=args.steps)  # stable for intermittent
                best_hold = cro if best_name=="Croston-SBA" else sn
                selected = pd.concat([best_hold, future]).sort_index()

            else:
                # Rolling-origin CV across classical models + SN
                cv_stats = rolling_origin_cv(y, ["ETS","SARIMA","Holt-Winters","SARIMAX","Seasonal-Naive"], use_log, k=args.cv_splits, min_train=24)
                # Fit models on train and produce holdout preds
                # ETS
                efit, _ = fit_ets_grid(train, SEASONAL_PERIODS, use_log)
                if efit is not None:
                    ep=ets_forecast(efit, len(test), train.index[-1], use_log); ep.index=test.index
                    holdout_map["ETS"]=ep
                    metrics.append(("ETS",
                        mean_absolute_error(test,ep), rmse(test,ep),
                        mape_safe(test.values,ep.values),
                        mase(test.values,ep.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,ep.values), mpe(test.values,ep.values)))
                # SARIMA
                sfit, _ = fit_sarima_grid(train, SEASONAL_PERIODS, use_log)
                if sfit is not None:
                    sp=sarima_forecast(sfit, len(test), train.index[-1], use_log); sp.index=test.index
                    holdout_map["SARIMA"]=sp
                    metrics.append(("SARIMA",
                        mean_absolute_error(test,sp), rmse(test,sp),
                        mape_safe(test.values,sp.values),
                        mase(test.values,sp.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,sp.values), mpe(test.values,sp.values)))
                # Holt-Winters
                try:
                    seasonal=len(train)>=24
                    hw=ExponentialSmoothing(np.log1p(train) if use_log else train,
                                            trend="add", seasonal=("add" if seasonal else None),
                                            seasonal_periods=(SEASONAL_PERIODS if seasonal else None),
                                            damped_trend=True, initialization_method="estimated").fit(optimized=True, remove_bias=True)
                    hp=hw.forecast(len(test)); hp=np.expm1(hp) if use_log else hp; hp.index=test.index
                    hp=clip_nonneg(hp)
                    holdout_map["Holt-Winters"]=hp
                    metrics.append(("Holt-Winters",
                        mean_absolute_error(test,hp), rmse(test,hp),
                        mape_safe(test.values,hp.values),
                        mase(test.values,hp.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,hp.values), mpe(test.values,hp.values)))
                except: pass
                # SARIMAX
                xfit, _ = fit_sarimax_month_dummies(train, SEASONAL_PERIODS, use_log)
                if xfit is not None:
                    xp=sarimax_forecast(xfit, train.index[-1], len(test), use_log); xp.index=test.index
                    holdout_map["SARIMAX"]=xp
                    metrics.append(("SARIMAX",
                        mean_absolute_error(test,xp), rmse(test,xp),
                        mape_safe(test.values,xp.values),
                        mase(test.values,xp.values,train.values,SEASONAL_PERIODS),
                        wape(test.values,xp.values), mpe(test.values,xp.values)))
                # SN baseline
                sn=seasonal_naive_forecast(train, SEASONAL_PERIODS, len(test)); sn.index=test.index
                holdout_map["Seasonal-Naive"]=sn
                metrics.append(("Seasonal-Naive",
                    mean_absolute_error(test,sn), rmse(test,sn),
                    mape_safe(test.values,sn.values),
                    mase(test.values,sn.values,train.values,SEASONAL_PERIODS),
                    wape(test.values,sn.values), mpe(test.values,sn.values)))

                # Choose CV winner first (more stable than single split)
                key_cv = lambda kv: (kv[1][0], kv[1][1], kv[1][2])
                best_cv_name = sorted(cv_stats.items(), key=key_cv)[0][0]

                # Blend = best classical (by CV) + SN → alpha via MAE on holdout → bias-corr
                classical_candidates = {k:v for k,v in holdout_map.items() if k in ["ETS","SARIMA","SARIMAX","Holt-Winters"]}
                if best_cv_name not in classical_candidates and classical_candidates:
                    # fallback to best classical by holdout MASE
                    def get_row(name):
                        for r in metrics:
                            if r[0]==name: return r
                        return None
                    best_classical = sorted(
                        [(n, get_row(n)) for n in classical_candidates.keys() if get_row(n) is not None],
                        key=lambda t: (np.nan_to_num(t[1][4],nan=1e9), np.nan_to_num(t[1][5],nan=1e9), np.nan_to_num(t[1][2],nan=1e9))
                    )[0][0]
                else:
                    best_classical = best_cv_name if best_cv_name in classical_candidates else None

                if best_classical is not None:
                    y_class = classical_candidates[best_classical].values
                    y_sn    = sn.values
                    alphas=np.linspace(0,0.6,7); best_a=0; best_mae=1e18
                    for a in alphas:
                        yb=a*y_sn + (1-a)*y_class
                        mae=mean_absolute_error(test.values, yb)
                        if mae<best_mae: best_mae, best_a = mae, a
                    bc = calibrate(test.values, best_a*y_sn + (1-best_a)*y_class)
                    y_blend_bc = (best_a*y_sn + (1-best_a)*y_class)*bc
                    holdout_map["Blend"]=pd.Series(y_blend_bc, index=test.index)
                    metrics.append(("Blend",
                        mean_absolute_error(test,y_blend_bc), rmse(test,y_blend_bc),
                        mape_safe(test.values,y_blend_bc),
                        mase(test.values,y_blend_bc,train.values,SEASONAL_PERIODS),
                        wape(test.values,y_blend_bc), mpe(test.values,y_blend_bc)))
                else:
                    best_a=0; bc=1.0

                # Final winner by MASE→WAPE→RMSE on **holdout** among all (incl. Blend)
                key_holdout=lambda r:(np.nan_to_num(r[4],nan=1e9), np.nan_to_num(r[5],nan=1e9), np.nan_to_num(r[2],nan=1e9))
                best_name = sorted(metrics, key=key_holdout)[0][0]

                # Future forecast from the winner
                if best_name in ["ETS","SARIMA","SARIMAX","Holt-Winters"]:
                    if best_name=="ETS" and efit is not None:
                        future = ets_forecast(efit, args.steps, y.index[-1], use_log)
                    elif best_name=="SARIMA" and sfit is not None:
                        future = sarima_forecast(sfit, args.steps, y.index[-1], use_log)
                    elif best_name=="SARIMAX" and xfit is not None:
                        future = sarimax_forecast(xfit, y.index[-1], args.steps, use_log)
                    else:
                        tmp = hw.forecast(args.steps); tmp=np.expm1(tmp) if use_log else tmp
                        idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=args.steps, freq="MS")
                        future = clip_nonneg(pd.Series(tmp.values, index=idx))
                    best_hold = holdout_map[best_name]
                elif best_name=="Blend" and best_classical is not None:
                    # recompute future: best classical future + SN future, with same alpha & bias
                    if best_classical=="ETS" and efit is not None:
                        fc_class = ets_forecast(efit, args.steps, y.index[-1], use_log).values
                    elif best_classical=="SARIMA" and sfit is not None:
                        fc_class = sarima_forecast(sfit, args.steps, y.index[-1], use_log).values
                    elif best_classical=="SARIMAX" and xfit is not None:
                        fc_class = sarimax_forecast(xfit, y.index[-1], args.steps, use_log).values
                    else:
                        tmp = hw.forecast(args.steps); tmp=np.expm1(tmp) if use_log else tmp
                        fc_class = clip_nonneg(pd.Series(tmp.values,
                                  index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=args.steps, freq="MS"))).values
                    fc_sn = seasonal_naive_forecast(y, SEASONAL_PERIODS, args.steps).values
                    fc = (best_a*fc_sn + (1-best_a)*fc_class)*bc
                    idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=args.steps, freq="MS")
                    future = pd.Series(fc, index=idx)
                    best_hold = holdout_map["Blend"]
                else:
                    future = seasonal_naive_forecast(y, SEASONAL_PERIODS, args.steps)
                    best_hold = sn

                selected = pd.concat([best_hold, future]).sort_index()

            # Plot & save (no clipping)
            png = out_dir / f"{sanitize(desc)}_80_20.png"
            plot_bands_connected(desc, y, train, test, holdout_map, selected, future, metrics, png)

            # Export chosen future forecast
            future.to_csv(out_dir/f"{sanitize(desc)}_forecast.csv", header=["forecast"])

            # Summary row
            mase_map = {r[0]: r[4] for r in metrics}
            rows.append({"description":desc,"chosen":best_name,"MASE_holdout":mase_map.get(best_name,np.nan),"plot":png.name})

        if rows:
            pd.DataFrame(rows).to_csv(out_dir/"ts_summary.csv", index=False)
            print(f"✅ Saved → {out_dir/'ts_summary.csv'}")
        print(f"✅ Top-10 list → {out_dir/'top10_bestsellers.csv'}")
        print(f"📁 Outputs: {out_dir}")

if __name__ == "__main__":
    main()
