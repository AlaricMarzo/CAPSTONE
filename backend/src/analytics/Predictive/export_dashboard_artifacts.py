import argparse, re, sys, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ------------------- Config & helpers -------------------
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "cleaned"
PUBLIC_OUT = HERE / "public" / "forecasts"
SEASON_M = 12

ALIASES = {
    "Date": ["date","transaction date","trans date","receipt date","sales date",
             "order date","invoice date","posting date"],
    "Description": ["description","desc","item name","product","product name","name","item"],
    "Qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty",
            "sold qty","sale qty","qnt","qnty"],
    "Item Code": ["item code","item_code","itemcode","sku","sku id","sku_id","barcode",
                  "product code","product_code","productcode","upc","ean","code","id",
                  "item id","itemid"],
}

def clean_name(s:str)->str:
    return re.sub(r"[^a-z0-9_\-]+","_", str(s).strip().lower().replace(" ","_")) or "name"

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
    if not d or not q or not desc:
        raise ValueError("Could not detect Date/Qty/Description columns.")
    ren = {d:"Date", q:"Qty", desc:"Description"}
    if sku: ren[sku]="Item Code"
    return ren

def monthly(df: pd.DataFrame, key:str) -> pd.DataFrame:
    return (df.groupby([key, pd.Grouper(key="Date", freq="MS")])["Qty"]
              .sum().reset_index().sort_values([key,"Date"]))

def choose_top(mon: pd.DataFrame, key:str, topn:int, min_cov:float) -> List[str]:
    stats = (mon.groupby(key)["Qty"].agg(total="sum", nz=lambda s:int((s>0).sum()), n="size")
             .reset_index())
    stats["coverage"]=stats["nz"]/stats["n"]
    pool = stats[stats["coverage"]>=min_cov] if (stats["coverage"]>=min_cov).any() else stats
    return pool.sort_values(["coverage","total"], ascending=[False,False])[key].head(topn).tolist()

def winsorize(y: pd.Series, q=0.995) -> pd.Series:
    if len(y)<8: return y
    return y.clip(upper=float(y.quantile(q)))

def z_from_service(p: float) -> float:
    # Approx inverse-CDF for standard normal (good enough for service levels)
    from math import sqrt, log
    if p<=0 or p>=1: return 0.0
    a1=-39.6968302866538; a2=220.946098424521; a3=-275.928510446969
    a4=138.357751867269; a5=-30.6647980661472; a6=2.50662827745924
    b1=-54.4760987982241; b2=161.585836858041; b3=-155.698979859887
    b4=66.8013118877197; b5=-13.2806815528857
    c1=-7.78489400243029E-03; c2=-0.322396458041136; c3=-2.40075827716184
    c4=-2.54973253934373; c5=4.37466414146497; c6=2.93816398269878
    d1=7.78469570904146E-03; d2=0.32246712907004; d3=2.44513413714299; d4=3.75440866190742
    q = p - 0.5
    if abs(q) <= .425:
        r = .180625 - q*q
        num = (((((a6*r+a5)*r+a4)*r+a3)*r+a2)*r+a1)
        den = (((((b5*r+b4)*r+b3)*r+b2)*r+b1)*r+1)
        return q * num/den
    r = p if q<0 else (1-p)
    r = np.sqrt(-np.log(r))
    if r<=5:
        r -= 1.6
        return ( (((((c6*r+c5)*r+c4)*r+c3)*r+c2)*r+c1) /
                 ((((d4*r+d3)*r+d2)*r+d1)*r+1) ) * (-1 if q<0 else 1)
    r -= 5
    num = (((((c6*r+c5)*r+c4)*r+c3)*r+c2)*r+c1)
    den = ((((d4*r+d3)*r+d2)*r+d1)*r+1)
    return (num/den) * (-1 if q<0 else 1)

# ------------------- Core Exporter -------------------
MODEL_DIRS = {
    # name -> (folder_glob, summary_filename, forecast_suffix)
    "LSTM":              ("ml_lstm",                "lstm_summary.csv",         "_forecast.csv"),
    "ExtraTrees":        ("ml_extra-trees_model",   "extratrees_summary.csv",   "_forecast.csv"),
    "GradientBoosting":  ("ml_gradient_boosting",   "gb_summary.csv",           "_forecast.csv"),
    "RandomForest":      ("ml_random_forest",       "rf_summary.csv",           "_forecast.csv"),
    "XGBoost":           ("ml_xgboost",             "xgb_summary.csv",          "_forecast.csv"),
    "ETS":               ("ml_ets",                 "ets_summary.csv",          "_forecast.csv"),
    "HoltWinters":       ("ml_holtwinters",         "holt_summary.csv",         "_forecast.csv"),
    "SARIMA":            ("ml_sarima",              "sarima_summary.csv",       "_forecast.csv"),
    "SARIMAX":           ("ml_sarimax",             "sarimax_summary.csv",      "_forecast.csv"),
    "Croston":           ("ml_croston",             "croston_summary.csv",      "_forecast.csv"),
}

# any folder at repo root starting with ts_ is treated as a time-series model bucket
TS_FOLDER_GLOB = "ts_*"
TS_SUMMARY_CANDIDATES = [
    "ts_summary.csv",
    "*_summary.csv",
    "random_*_summary.csv"
]

def _read_csv_loose(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, header=None)
        except Exception:
            return None

def load_model_summaries(dataset_slug: str) -> pd.DataFrame:
    rows=[]
    for mname, (folder, summary_file, _suf) in MODEL_DIRS.items():
        root = HERE / folder / dataset_slug
        f = root / summary_file
        if f.exists():
            try:
                df = pd.read_csv(f)
                df["model"]=mname
                rows.append(df)
            except Exception as e:
                print(f"[WARN] Cannot read summary {f}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_all_model_forecasts(dataset_slug: str) -> pd.DataFrame:
    """Return long table: [sku, date, model, forecast] for ML model dirs."""
    frames=[]
    for mname,(folder,_sum,suf) in MODEL_DIRS.items():
        root = HERE / folder / dataset_slug
        if not root.exists(): continue
        for f in root.glob(f"*{suf}"):
            sku = f.stem.replace("_forecast","")
            df = _read_csv_loose(f)
            if df is None or df.empty: continue
            cands = {c.lower(): c for c in df.columns}
            date_col = cands.get("date", list(df.columns)[0])
            val_col  = cands.get("forecast", list(df.columns)[1] if len(df.columns)>1 else list(df.columns)[0])
            tmp = pd.DataFrame({
                "sku":[sku]*len(df),
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "model": mname,
                "forecast": pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
            })
            frames.append(tmp.dropna(subset=["date"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["sku","date","model","forecast"])

def load_ts_summaries(dataset_slug: str) -> pd.DataFrame:
    """Sweep ts_* folders for summary CSVs (best-effort)."""
    rows=[]
    for ts_dir in HERE.glob(TS_FOLDER_GLOB):
        if not ts_dir.is_dir(): continue
        # prefer dataset subfolder if present
        candidates = []
        ds_dir = ts_dir / dataset_slug
        bases = [ds_dir] if ds_dir.exists() else [ts_dir]
        for base in bases:
            for pat in TS_SUMMARY_CANDIDATES:
                for f in base.glob(pat):
                    df = _read_csv_loose(f)
                    if df is None or df.empty: continue
                    df = df.copy()
                    df["ts_bucket"] = ts_dir.name
                    df["source_file"] = f.name
                    rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_ts_forecasts(dataset_slug: str) -> pd.DataFrame:
    """Return long table for classical TS outputs under any ts_* dir."""
    frames=[]
    for ts_dir in HERE.glob(TS_FOLDER_GLOB):
        if not ts_dir.is_dir(): continue
        model_name = ts_dir.name  # keep folder name as the model label
        # Try both patterns:
        #   ts_dir / dataset_slug / <sku>_forecast.csv
        #   ts_dir / <sku>_forecast.csv
        search_dirs = []
        ds_dir = ts_dir / dataset_slug
        if ds_dir.exists(): search_dirs.append(ds_dir)
        search_dirs.append(ts_dir)
        seen=set()
        for base in search_dirs:
            for f in base.glob("*_forecast.csv"):
                # avoid double-counting if both parent & child contain same file
                key=(f.name, base.as_posix())
                if key in seen: continue
                seen.add(key)
                sku = f.stem.replace("_forecast","")
                df = _read_csv_loose(f)
                if df is None or df.empty: continue
                cols = {c.lower(): c for c in df.columns}
                date_col = cols.get("date", list(df.columns)[0])
                val_col  = cols.get("forecast", list(df.columns)[1] if len(df.columns)>1 else list(df.columns)[0])
                tmp = pd.DataFrame({
                    "sku":[sku]*len(df),
                    "date": pd.to_datetime(df[date_col], errors="coerce"),
                    "model": model_name,
                    "forecast": pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
                })
                frames.append(tmp.dropna(subset=["date"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["sku","date","model","forecast"])

def compute_monthly_history(df_in: pd.DataFrame, key="Item Code") -> pd.DataFrame:
    mon = monthly(df_in, key)
    out=[]
    for sku, g in mon.groupby(key):
        span = pd.date_range(g["Date"].min(), g["Date"].max(), freq="MS")
        y = (g.set_index("Date")["Qty"].reindex(span).fillna(0.0).astype(float))
        tmp = y.to_frame("qty_actual")
        tmp["sku"]=sku
        tmp["date"]=tmp.index
        out.append(tmp.reset_index(drop=True)[["sku","date","qty_actual"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["sku","date","qty_actual"])

def seasonal_index(history: pd.DataFrame) -> pd.DataFrame:
    """Compute multiplicative seasonal indices per SKU (month-of-year)."""
    if history.empty: return pd.DataFrame(columns=["sku","month","seasonal_index"])
    h = history.copy()
    h["month"] = pd.to_datetime(h["date"]).dt.month
    base = h.groupby("sku")["qty_actual"].mean().replace(0, np.nan)
    merged = h.merge(base.rename("base"), on="sku", how="left")
    merged["ratio"] = merged["qty_actual"]/merged["base"]
    out = (merged.groupby(["sku","month"])["ratio"].mean()
                 .rename("seasonal_index").reset_index())
    out["seasonal_index"] = out["seasonal_index"].replace([np.inf,-np.inf], np.nan).fillna(1.0)
    return out

def volatility_metrics(history: pd.DataFrame) -> pd.DataFrame:
    """Rolling 6-month std/mean and CV."""
    if history.empty: return pd.DataFrame(columns=["sku","vol_roll6_std","vol_roll6_mean","cv_roll6"])
    h = history.copy().sort_values(["sku","date"])
    h["roll6_std"] = h.groupby("sku")["qty_actual"].rolling(6, min_periods=3).std().reset_index(level=0, drop=True)
    h["roll6_mean"] = h.groupby("sku")["qty_actual"].rolling(6, min_periods=3).mean().reset_index(level=0, drop=True)
    h["cv_roll6"] = (h["roll6_std"]/h["roll6_mean"]).replace([np.inf,-np.inf], np.nan)
    agg = (h.groupby("sku")[["roll6_std","roll6_mean","cv_roll6"]].agg("mean")
             .rename(columns={"roll6_std":"vol_roll6_std","roll6_mean":"vol_roll6_mean","cv_roll6":"cv_roll6"}).reset_index())
    return agg

def demand_insights(history: pd.DataFrame) -> pd.DataFrame:
    """Coverage, intermittency, avg, median, trend slope (OLS), last12 sum."""
    if history.empty:
        cols=["sku","months","nz_months","coverage","zero_share","avg_qty","med_qty","trend_slope","last12_qty"]
        return pd.DataFrame(columns=cols)
    h = history.copy().sort_values(["sku","date"])
    agg = (h.groupby("sku")
             .agg(months=("qty_actual","size"),
                  nz_months=("qty_actual", lambda s: int((s>0).sum())),
                  avg_qty=("qty_actual","mean"),
                  med_qty=("qty_actual","median"),
                  last12_qty=("qty_actual", lambda s: s.tail(12).sum()))
             .reset_index())
    agg["coverage"] = agg["nz_months"]/agg["months"]
    zs = (h.assign(is_zero=(h["qty_actual"]==0).astype(int))
            .groupby("sku")["is_zero"].mean().rename("zero_share").reset_index())
    h["t"] = h.groupby("sku").cumcount()+1
    def slope(g):
        if len(g)<3: return np.nan
        x = g["t"].values; y=g["qty_actual"].values
        xm, ym = x.mean(), y.mean()
        denom = ((x-xm)**2).sum()
        if denom==0: return np.nan
        return ((x-xm)*(y-ym)).sum()/denom
    tr = h.groupby("sku").apply(slope).rename("trend_slope").reset_index()
    out = agg.merge(zs, on="sku", how="left").merge(tr, on="sku", how="left")
    return out

def inventory_recos(history: pd.DataFrame,
                    lead_days:int, service_lvl:float,
                    annual_holding:float, order_cost:float) -> pd.DataFrame:
    """EOQ + ROP using simple stats (per SKU, monthly → daily approx)."""
    if history.empty:
        return pd.DataFrame(columns=["sku","avg_daily_demand","std_daily_demand","ROP","SS","EOQ"])
    h = history.copy()
    agg = (h.groupby("sku")["qty_actual"]
             .agg(monthly_avg="mean", monthly_std="std", annual_demand=lambda s: s.sum()*(12/max(1,len(s)/12)))
             .reset_index())
    agg["avg_daily_demand"] = agg["monthly_avg"]/30.0
    agg["std_daily_demand"] = (agg["monthly_std"].fillna(0.0)/math.sqrt(30.0))
    z = z_from_service(service_lvl)
    agg["SS"]  = (z * agg["std_daily_demand"] * math.sqrt(max(lead_days,1))).clip(lower=0)
    agg["ROP"] = (agg["avg_daily_demand"] * lead_days) + agg["SS"]
    U = 1.0
    H = annual_holding * U
    agg["EOQ"] = np.sqrt((2 * agg["annual_demand"].clip(lower=0.01) * order_cost) / max(H, 1e-6))
    return agg[["sku","avg_daily_demand","std_daily_demand","SS","ROP","EOQ"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--lead_days", type=int, default=10, help="Supplier lead time (days)")
    ap.add_argument("--service_lvl", type=float, default=0.95, help="Cycle service level (0-1)")
    ap.add_argument("--annual_holding", type=float, default=0.20, help="Holding cost rate (0-1) per year")
    ap.add_argument("--order_cost", type=float, default=150.0, help="Per-order fixed cost (currency)")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--min_cov", type=float, default=0.70)
    args = ap.parse_args()

    in_path = Path(args.input) if args.input else (DEFAULT_DATA_DIR/"ANC - 4 YEARS (1).csv")
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    # Read source data
    print(f"[INFO] Reading: {in_path}")
    df = pd.read_csv(in_path, low_memory=False) if in_path.suffix.lower()==".csv" else pd.read_excel(in_path)
    rename = detect_columns(df)
    print(f"[INFO] Detected columns: {rename}")
    df = df.rename(columns=rename)
    df["Date"] = parse_dates_safe(df["Date"])
    df = df.dropna(subset=["Date"])
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0.0).astype(float)
    df["Description"] = df["Description"].astype(str).fillna("Unknown Product")
    if "Item Code" not in df.columns: df["Item Code"] = df["Description"].astype(str)

    dataset_slug = clean_name(in_path.stem)
    out_dir = PUBLIC_OUT / dataset_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute monthly history (for selected top SKUs to keep file sizes manageable)
    mon = monthly(df, "Item Code")
    all_top = choose_top(mon, "Item Code", args.topn, args.min_cov)
    mon_top = mon[mon["Item Code"].isin(all_top)].copy()
    history = compute_monthly_history(df[df["Item Code"].isin(all_top)], key="Item Code")

    # Load model summaries + per-SKU forecasts for ML models
    model_perf_ml = load_model_summaries(dataset_slug)
    all_fc_ml = load_all_model_forecasts(dataset_slug)

    # ---- NEW: time-series (classical) pulls ----
    ts_summ = load_ts_summaries(dataset_slug)     # multiple shapes, best-effort
    all_fc_ts = load_ts_forecasts(dataset_slug)   # long table

    # 1) MODEL PERFORMANCE (dataset-level) — keep ML summaries; TS summaries go to their own file
    if not model_perf_ml.empty:
        model_perf_ml["sku"] = model_perf_ml["sku"].astype(str)
        model_perf_ml.to_csv(out_dir / f"{dataset_slug}_model_performance.csv", index=False)
        print(f"✔ {dataset_slug}_model_performance.csv")
    else:
        pd.DataFrame(columns=["sku","description","chosen","MASE_WF","MASE_holdout","alpha","bias","plot","model"]).to_csv(
            out_dir / f"{dataset_slug}_model_performance.csv", index=False)
        print(f"✔ {dataset_slug}_model_performance.csv (empty)")

    # 1b) TIME SERIES SUMMARY (best-effort union of ts_* summaries)
    if not ts_summ.empty:
        ts_summ.to_csv(out_dir / f"{dataset_slug}_time_series_summary.csv", index=False)
        print(f"✔ {dataset_slug}_time_series_summary.csv")
    else:
        pd.DataFrame().to_csv(out_dir / f"{dataset_slug}_time_series_summary.csv", index=False)
        print(f"✔ {dataset_slug}_time_series_summary.csv (empty)")

    # 2) ALL MODELS FORECAST (long) — ML + TS concatenated
    all_fc_long = []
    if not all_fc_ml.empty:  all_fc_long.append(all_fc_ml)
    if not all_fc_ts.empty:  all_fc_long.append(all_fc_ts)
    if all_fc_long:
        all_fc = pd.concat(all_fc_long, ignore_index=True)
        all_fc["sku"] = all_fc["sku"].astype(str)
        all_fc.sort_values(["sku","model","date"], inplace=True)
        all_fc.to_csv(out_dir / f"{dataset_slug}_all_models_forecast.csv", index=False)
        print(f"✔ {dataset_slug}_all_models_forecast.csv")
    else:
        pd.DataFrame(columns=["sku","date","model","forecast"]).to_csv(
            out_dir / f"{dataset_slug}_all_models_forecast.csv", index=False)
        print(f"✔ {dataset_slug}_all_models_forecast.csv (empty)")

    # 3) FULL TIMELINE (history + wide pivot(s) of any forecasts)
    ft = history.copy()
    if all_fc_long:
        all_fc = pd.concat(all_fc_long, ignore_index=True)
        pivot_fc = all_fc.pivot_table(index=["sku","date"], columns="model", values="forecast", aggfunc="first")
        pivot_fc = pivot_fc.add_suffix("_forecast").reset_index()
        ft = ft.merge(pivot_fc, on=["sku","date"], how="outer")
    ft.sort_values(["sku","date"], inplace=True)
    ft.to_csv(out_dir / f"{dataset_slug}_full_timeline.csv", index=False)
    print(f"✔ {dataset_slug}_full_timeline.csv")

    # 4) DEMAND INSIGHTS
    di = demand_insights(history)
    def demand_class(row):
        if row["avg_qty"]>=50 or row["last12_qty"]>=400: return "High-volume"
        if row["avg_qty"]>=15 or row["last12_qty"]>=150: return "Moderate"
        return "Low-volume"
    if not di.empty:
        di["demand_class"] = di.apply(demand_class, axis=1)
    di.to_csv(out_dir / f"{dataset_slug}_demand_insights.csv", index=False)
    print(f"✔ {dataset_slug}_demand_insights.csv")

    # 5) INVENTORY RECOMMENDATIONS (ROP, SS, EOQ)
    inv = inventory_recos(history, args.lead_days, args.service_lvl, args.annual_holding, args.order_cost)
    inv.to_csv(out_dir / f"{dataset_slug}_inventory_recommendations.csv", index=False)
    print(f"✔ {dataset_slug}_inventory_recommendations.csv")

    # 6) SEASONAL ANALYSIS (month-of-year index)
    seas = seasonal_index(history)
    seas.to_csv(out_dir / f"{dataset_slug}_seasonal_analysis.csv", index=False)
    print(f"✔ {dataset_slug}_seasonal_analysis.csv")

    # 7) VOLATILITY ANALYSIS (rolling stats)
    vol = volatility_metrics(history)
    vol.to_csv(out_dir / f"{dataset_slug}_volatility_analysis.csv", index=False)
    print(f"✔ {dataset_slug}_volatility_analysis.csv")

    print(f"\n✅ Done. See → {out_dir}")

if __name__ == "__main__":
    main()
