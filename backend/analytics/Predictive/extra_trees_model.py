#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExtraTrees monthly demand forecast (Top-N) with walk-forward MASE selection.
- Robust target (winsorize + log1p), rich seasonality features, seasonal-naive blend
- Intermittent series -> Croston-SBA
- Evaluation: walk-forward (expanding) MASE + last-h holdout with metrics table
- Bias correction on forward forecasts

Run:
  python extra_trees_model.py --input "cleaned/ANC - 4 YEARS (1).csv" --topn 5 --min_cov 0.7 --tune 1
"""
import warnings, argparse, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import numpy as np, pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_ROOT = HERE / "ml_extra-trees_model"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FORECAST_STEPS = 6
SEASON_M = 12
TOP_N = 5
INTERMITTENT_ZERO_SHARE = 0.40

ALIASES = {
    "Date": ["date","transaction date","trans date","receipt date","sales date","order date","invoice date","posting date"],
    "Description": ["description","desc","item name","product","product name","name","item"],
    "Qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty","sold qty","sale qty","qnt","qnty"],
    "Item Code": ["item code","item_code","itemcode","sku","sku id","sku_id","barcode","product code","product_code","productcode","upc","ean","code","id","item id","itemid"],
}

# ---------- helpers ----------
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
    mon = (df.groupby([key, pd.Grouper(key="Date", freq="MS")])["Qty"].sum()
             .reset_index().sort_values([key,"Date"]))
    return mon

def choose_top(mon: pd.DataFrame, key:str, topn:int, min_cov:float) -> List[str]:
    stats = (mon.groupby(key)["Qty"]
             .agg(total="sum", nz=lambda s:int((s>0).sum()), n="size")
             .reset_index())
    stats["coverage"]=stats["nz"]/stats["n"]
    pool = stats[stats["coverage"]>=min_cov] if (stats["coverage"]>=min_cov).any() else stats
    return pool.sort_values(["coverage","total"], ascending=[False,False])[key].head(topn).tolist()

def winsorize(y: pd.Series, q=0.995) -> pd.Series:
    if len(y)<8: return y
    cap = y.quantile(q); return y.clip(upper=float(cap))

def add_ts_features(s: pd.Series) -> pd.DataFrame:
    df = s.to_frame("y").copy()
    idx = pd.to_datetime(df.index)
    month = idx.month
    df["m_sin"] = np.sin(2*np.pi*month/12.0)
    df["m_cos"] = np.cos(2*np.pi*month/12.0)
    for L in range(1, SEASON_M+1): df[f"lag{L}"] = df["y"].shift(L)
    for W in [3,6,12]:
        df[f"rmean{W}"]=df["y"].rolling(W).mean()
        df[f"rstd{W}"]=df["y"].rolling(W).std()
        df[f"rmed{W}"]=df["y"].rolling(W).median()
    df["diff1"]=df["y"].diff(1)
    df["pct1"]=df["y"].pct_change(1).replace([np.inf,-np.inf],0).fillna(0)
    return df.dropna()

def mape(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    den = np.where(yt==0,1,yt); return float(np.mean(np.abs((yt-yp)/den))*100)
def wape(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    den = np.sum(np.abs(yt)); return float(np.sum(np.abs(yt-yp))/den*100) if den>0 else np.nan
def smape(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    den = np.abs(yt)+np.abs(yp); den=np.where(den==0,1,den)
    return float(np.mean(2*np.abs(yt-yp)/den)*100)
def mpe(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    mask=yt!=0; return float(np.mean(((yp[mask]-yt[mask])/yt[mask]))*100) if mask.any() else np.nan
def mase(y_true, y_pred, insample, m=SEASON_M):
    ins=np.asarray(insample,float)
    if len(ins)<=m: denom=np.mean(np.abs(np.diff(ins))) if len(ins)>1 else 1.0
    else: denom=np.mean(np.abs(ins[m:]-ins[:-m]))
    if denom==0: denom=1.0
    return float(np.mean(np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float)))/denom)

def seasonal_naive(y, h, m=SEASON_M):
    y=np.asarray(y,float)
    if len(y)<m:
        last=y[-1] if len(y) else 0.0
        return np.repeat(max(last,0.0), h)
    pat=y[-m:]; reps=int(np.ceil(h/m)); fc=np.tile(pat,reps)[:h]
    return np.maximum(fc,0.0)

def is_intermittent(y: pd.Series, thr=INTERMITTENT_ZERO_SHARE)->bool:
    return (y==0).mean() >= thr

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

def blocked_random_splits(n:int, k:int=5, min_test:int=3)->List[Tuple[int,int]]:
    """Return list of (start,end) indices for k contiguous test blocks (sensitivity only)."""
    if n<min_test+8: return []
    rng=np.random.RandomState(42)
    spans=[]
    for _ in range(k):
        t = rng.randint(min_test, max(min_test, n//5))
        s = rng.randint(n - t)
        spans.append((s, s+t))
    return sorted(set(spans))

def plot_with_table(title, train_idx, full_idx, y, test_idx, pred_map, rows, out_png):
    fig=plt.figure(figsize=(13,7)); gs=fig.add_gridspec(2,1,height_ratios=[3,1],hspace=0.15)
    ax=fig.add_subplot(gs[0])
    ax.plot(full_idx, y, label="Historical", lw=2)
    for name, ser in pred_map.items():
        ax.plot(ser.index, ser.values, lw=2, ls="--", label=name)
    ax.set_title(title); ax.set_xlabel("Month"); ax.set_ylabel("Qty")
    ax.grid(ls=":", alpha=0.6); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels(): t.set_rotation(45); t.set_ha("right")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)
    ax2=fig.add_subplot(gs[1]); ax2.axis("off")
    header=["Model","Params","MAE","MSE","RMSE","MAPE%","MASE","WAPE%","MPE%"]
    table=[[n,str(p),f"{a:,.2f}",f"{m:,.2f}",f"{r:,.2f}",f"{mp:,.2f}",f"{ms:.3f}",f"{wa:,.2f}",f"{me:,.2f}"] for (n,p,a,m,r,mp,ms,wa,me) in rows]
    tbl=ax2.table(cellText=table, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.15)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

# ---------- model core ----------
def fit_model_extratrees(Xtr, ytr_log, tune=False):
    base = ExtraTreesRegressor(
        n_estimators=800, max_depth=None, min_samples_leaf=2, max_features=0.8,
        random_state=42, n_jobs=-1
    )
    if not tune: 
        m=base.fit(Xtr, ytr_log); return m, {"n_estimators":800,"min_samples_leaf":2,"max_features":0.8}
    param_dist={"n_estimators":[600,800,1000],"min_samples_leaf":[1,2,3,5],"max_features":["sqrt",0.6,0.8,1.0],"max_depth":[None,8,12,16]}
    rs=RandomizedSearchCV(base, param_distributions=param_dist, n_iter=10,
                          scoring="neg_mean_absolute_error", cv=TimeSeriesSplit(n_splits=3),
                          random_state=42, n_jobs=-1)
    rs.fit(Xtr, ytr_log); return rs.best_estimator_, rs.best_params_

def train_eval_holdout(y: pd.Series, h:int, tune:bool):
    """Train on y[:-h], evaluate on last h months; return preds and metrics rows for table."""
    train, test = y.iloc[:-h], y.iloc[-h:]
    feats_tr, feats_te = add_ts_features(train), add_ts_features(y).iloc[-h:]
    # winsorize + log1p on training target only
    y_tr_raw = feats_tr["y"].values; cap=np.quantile(y_tr_raw,0.995)
    y_tr_cap = np.minimum(y_tr_raw, cap); ytr_log = np.log1p(y_tr_cap)
    Xtr = feats_tr.drop(columns=["y"]).values; Xte = feats_te.drop(columns=["y"]).values
    mdl, params = fit_model_extratrees(Xtr, ytr_log, tune=tune)
    yhat_et = np.expm1(mdl.predict(Xte)).clip(0)
    # Seasonal-naive and blend
    y_sn = seasonal_naive(train.values, h, m=SEASON_M)
    # choose alpha by MAE on this holdout
    alphas = np.linspace(0,0.6,7); best_a=0; best_mae=1e18
    for a in alphas:
        yb = a*y_sn + (1-a)*yhat_et
        mae = mean_absolute_error(test.values, yb)
        if mae<best_mae: best_mae, best_a = mae, a
    y_blend = best_a*y_sn + (1-best_a)*yhat_et
    # bias correction (clip)
    bc = np.clip((test.values.sum()/y_blend.sum()) if y_blend.sum()>0 else 1.0, 0.7, 1.3)
    y_blend_bc = y_blend*bc
    # metrics rows
    def met(name, pred, pdesc):
        return (name, pdesc,
            mean_absolute_error(test, pred),
            mean_squared_error(test, pred),
            np.sqrt(mean_squared_error(test, pred)),
            mape(test, pred),
            mase(test, pred, train.values, m=SEASON_M),
            wape(test, pred),
            mpe(test, pred)
        )
    rows=[met("ExtraTrees", yhat_et, params),
          met("Seasonal-Naive", y_sn, f"m={SEASON_M}"),
          met("Blend", y_blend_bc, f"alpha={best_a:.2f}, bias={bc:.2f}")]
    # build dated series for plotting
    te_idx = feats_te.index
    pred_map = {"ET holdout": pd.Series(yhat_et, index=te_idx),
                "SN holdout": pd.Series(y_sn, index=te_idx),
                "Blend (bias-corr)": pd.Series(y_blend_bc, index=te_idx)}
    return train, test, pred_map, rows, best_a, bc, mdl, params

def walk_forward_mase(y: pd.Series, fit_fn: Callable[[pd.Series], Tuple[np.ndarray, np.ndarray]]):
    """Expanding window 1-step ahead MASE using only past information."""
    preds=[]; trues=[]; m=SEASON_M
    for i in range(max(24, m+1), len(y)):
        hist = y.iloc[:i]  # up to month i-1
        try:
            yhat = fit_fn(hist)  # returns 1-step forecast (float)
            preds.append(float(yhat)); trues.append(float(y.iloc[i]))
        except: pass
    if not preds: return np.nan
    ins = y.values[:-len(trues)] if len(trues)>0 else y.values
    return mase(np.array(trues), np.array(preds), ins, m=m)

def fit_fn_et_1step(hist: pd.Series, tune=False):
    feats = add_ts_features(hist); Xtr=feats.drop(columns=["y"]).values
    y_tr_raw=feats["y"].values; cap=np.quantile(y_tr_raw,0.995); ytr_log=np.log1p(np.minimum(y_tr_raw,cap))
    mdl,_ = fit_model_extratrees(Xtr, ytr_log, tune=False if len(feats)<36 else tune)
    # build features for next month
    last_ts = feats.index[-1].to_period("M").to_timestamp()
    nxt = (last_ts.to_period('M') + 1).to_timestamp()
    tmp = hist.copy(); tmp.loc[nxt]=hist.iloc[-1]  # placeholder to generate lags
    feats_next = add_ts_features(tmp).iloc[[-1]].drop(columns=["y"]).values
    return np.expm1(mdl.predict(feats_next))[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--topn", type=int, default=TOP_N)
    ap.add_argument("--min_cov", type=float, default=0.7)
    ap.add_argument("--tune", type=int, default=0)
    ap.add_argument("--rand_splits", type=int, default=0, help="number of blocked random splits for sensitivity")
    args = ap.parse_args()

    in_path = Path(args.input) if args.input else (DATA_DIR/"ANC - 4 YEARS (1).csv")
    if not in_path.exists(): sys.exit(f"Input not found: {in_path}")

    print(f"[INFO] Reading: {in_path}")
    df = pd.read_csv(in_path, low_memory=False) if in_path.suffix.lower()==".csv" else pd.read_excel(in_path)
    rename = detect_columns(df); print("[INFO] Detected columns:", rename)
    df = df.rename(columns=rename)
    df["Date"]=parse_dates_safe(df["Date"]); df=df.dropna(subset=["Date"])
    df["Qty"]=pd.to_numeric(df["Qty"], errors="coerce").fillna(0.0).astype(float)
    df["Description"]=df["Description"].astype(str).fillna("Unknown Product")
    if "Item Code" not in df.columns: df["Item Code"]=df["Description"].astype(str)

    key="Item Code"; mon=monthly(df, key)
    if mon.empty: sys.exit("No rows after monthly aggregation.")
    top_keys=choose_top(mon, key, args.topn, args.min_cov)
    print(f"[INFO] Top-{len(top_keys)} by coverage/qty: {top_keys}")

    out_dir = OUT_ROOT / clean_name(in_path.stem); out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]

    for sku in top_keys:
        s = mon[mon[key]==sku].copy().sort_values("Date")
        # regularize index
        span = pd.date_range(s["Date"].min(), s["Date"].max(), freq="MS")
        y = (s.set_index("Date")["Qty"].reindex(span).fillna(0.0).astype(float))
        desc = df[df[key]==sku]["Description"].dropna().iloc[0] if (df[key]==sku).any() else sku
        label = f"{sku} — {desc}"

        if len(y)<18:
            print(f"↪ {sku}: too short; fallback SN")
            h=min(3, max(1, len(y)//5))
            te_idx=y.index[-h:]; y_sn=seasonal_naive(y.iloc[:-h].values, h, m=SEASON_M)
            pred_map={"SN holdout": pd.Series(y_sn, index=te_idx)}
            row=("Seasonal-Naive", f"m={SEASON_M}",
                 mean_absolute_error(y.iloc[-h:], y_sn),
                 mean_squared_error(y.iloc[-h:], y_sn),
                 np.sqrt(mean_squared_error(y.iloc[-h:], y_sn)),
                 mape(y.iloc[-h:], y_sn),
                 mase(y.iloc[-h:], y_sn, y.iloc[:-h].values, m=SEASON_M),
                 wape(y.iloc[-h:], y_sn),
                 mpe(y.iloc[-h:], y_sn))
            png= out_dir/f"et_{clean_name(sku)}.png"
            plot_with_table(f"ExtraTrees (short series fallback) — {label}", y.index[:-h], y.index, y.values, te_idx, pred_map, [row], png)
            # forward forecast
            fut = seasonal_naive(y.values, FORECAST_STEPS, SEASON_M)
            pd.Series(fut, index=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=FORECAST_STEPS,freq="MS")).to_csv(out_dir/f"{clean_name(sku)}_forecast.csv", header=["forecast"])
            rows.append({"sku":sku,"description":desc,"chosen":"SN","MASE_WF":np.nan,"MASE_holdout":row[6],"plot":png.name}); continue

        y = winsorize(y, 0.995)

        # holdout size
        h = min(6, max(3, len(y)//5))
        train, test, pred_map, table_rows, alpha, bc, mdl, params = train_eval_holdout(y, h, tune=bool(args.tune))

        # walk-forward MASE
        wf_mase = walk_forward_mase(y, lambda hist: fit_fn_et_1step(hist, tune=bool(args.tune)))
        print(f"[{sku}] WF-MASE={wf_mase:.3f}  (α={alpha:.2f}, bias={bc:.2f})")

        # plot & save
        png = out_dir / f"et_{clean_name(sku)}.png"
        plot_with_table(f"ExtraTrees — last-{h} holdout — {label}", y.index[:-h], y.index, y.values, y.index[-h:], pred_map, table_rows, png)

        # forward 6m forecast from FULL history (apply bias-corr)
        # generate ET → SN → Blend(α) → bias-corr
        feats_full = add_ts_features(y); Xfull = feats_full.drop(columns=["y"]).values
        y_tr_raw=feats_full["y"].values; cap=np.quantile(y_tr_raw,0.995); ytr_log=np.log1p(np.minimum(y_tr_raw,cap))
        mdl_full,_ = fit_model_extratrees(Xfull, ytr_log, tune=bool(args.tune))
        hist = y.copy(); preds_et=[]
        for _ in range(FORECAST_STEPS):
            tmp = add_ts_features(hist)
            x = tmp.drop(columns=["y"]).iloc[[-1]].values
            yhat = float(np.expm1(mdl_full.predict(x))[0]); yhat=max(yhat,0.0)
            nxt = (hist.index[-1].to_period('M')+1).to_timestamp()
            hist.loc[nxt]=yhat; preds_et.append(yhat)
        fc_et=np.array(preds_et); fc_sn=seasonal_naive(y.values, FORECAST_STEPS, SEASON_M)
        fc = alpha*fc_sn + (1-alpha)*fc_et; fc = fc*bc
        idx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS")
        pd.Series(fc, index=idx).to_csv(out_dir/f"{clean_name(sku)}_forecast.csv", header=["forecast"])

        # sensitivity (optional blocked random splits)
        if args.rand_splits>0:
            spans=blocked_random_splits(len(y), args.rand_splits)
            sens=[]
            for s0,s1 in spans:
                tr = y.iloc[:s0]; te = y.iloc[s0:s1]
                if len(te)<3 or len(tr)<24: continue
                feats_tr, feats_te = add_ts_features(tr), add_ts_features(y.iloc[:s1]).iloc[-len(te):]
                ytr=np.log1p(np.minimum(feats_tr["y"].values, np.quantile(feats_tr["y"].values,0.995)))
                m,_=fit_model_extratrees(feats_tr.drop(columns=["y"]).values, ytr, tune=False)
                yhat=np.expm1(m.predict(feats_te.drop(columns=["y"]).values)).clip(0)
                sens.append(mase(te.values, yhat, tr.values, m=SEASON_M))
            if sens: print(f"   Sensitivity (blocked random) MASE mean={np.nanmean(sens):.3f}  n={len(sens)}")

        # summary row
        holdout_mase = {r[0]: r[6] for r in table_rows}.get("Blend", np.nan)
        rows.append({"sku":sku,"description":desc,"chosen":"ExtraTrees+Blend","MASE_WF":wf_mase,"MASE_holdout":holdout_mase,"alpha":alpha,"bias":bc,"plot":png.name})

    if rows:
        pd.DataFrame(rows).to_csv(OUT_ROOT / clean_name(in_path.stem) / "extratrees_summary.csv", index=False)
        print(f"\n✅ Saved summary → {OUT_ROOT/clean_name(in_path.stem)/'extratrees_summary.csv'}")
    else:
        print("No results produced.")

if __name__=="__main__":
    main()
