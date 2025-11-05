#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARIMA / ETS / SARIMAX (month dummies) with:
- Monthly aggregation from ANC raw
- Winsorization (caps spikes)
- Optional log1p variance stabilize
- Intermittent-demand routing (Croston-SBA)
- Recent-window training (default 24 months)
- Bias calibration from holdout (median test/pred ratio, clipped)
- 80/20 holdout chart with metrics table (MAE, MSE, RMSE, MAPE, MASE, WAPE, MPE)
- Best-by-MASE forward forecast (default 6 months) saved to CSV

Run examples:
  python sarima_ets_sarimax_calibrated.py --file "ANC - 4 YEARS (1).csv" --top 8 --steps 6
  python sarima_ets_sarimax_calibrated.py --recent 24 --winsor 0.95 --no-log
"""
import argparse, warnings, re, sys, math
from pathlib import Path
from typing import Optional, List, Tuple, Dict
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

# ---------- Matplotlib defaults ----------
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 200,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------- Config ----------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cleaned"
OUT_ROOT = HERE / "ts_sarima-ets-sarimax(2,1,2)"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEASONAL_PERIODS = 12
DEFAULT_FORECAST_STEPS = 6
DEFAULT_TOPN = 5

# ---------- Utils ----------
def parse_dates_safe(s: pd.Series) -> pd.Series:
    try: return pd.to_datetime(s, errors="coerce")
    except: return pd.to_datetime(s, errors="coerce", dayfirst=True)

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    lower = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "date": ["date","transaction date","sales date","receipt date","order date","invoice date","trans date","posting date"],
        "description": ["description","item name","product","product name","name","item","desc"],
        "qty": ["qty","quantity","qty sold","units","units sold","quantity sold","sales qty","sold qty","sale qty","qnt"]
    }
    def pick(key):
        if key in lower: return lower[key]
        for a in aliases[key]:
            if a in lower: return lower[a]
        if key=="date":
            for c in df.columns:
                try:
                    if pd.to_datetime(df[c], errors="coerce").notna().sum() >= 10:
                        return c
                except: pass
        return None
    d = pick("date"); desc = pick("description"); q = pick("qty")
    if not (d and desc and q):
        raise ValueError("Could not detect Date/Description/Qty columns in the file.")
    return {d:"Date", desc:"Description", q:"Qty"}

def load_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path) if path.suffix.lower() in [".xlsx",".xls"] else pd.read_csv(path, low_memory=False)
    df = df.rename(columns=detect_columns(df))
    df["Date"] = parse_dates_safe(df["Date"])
    df = df.dropna(subset=["Date"])
    df["Description"] = df["Description"].astype(str).str.strip().str.replace(r"\s+"," ", regex=True)
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0).astype(float)
    monthly = (df.set_index("Date")
                 .groupby("Description")["Qty"]
                 .resample("MS").sum()
                 .reset_index())
    return monthly

def series_for(monthly: pd.DataFrame, desc: str) -> pd.Series:
    return (monthly[monthly["Description"]==desc]
            .set_index("Date")["Qty"]
            .asfreq("MS", fill_value=0.0)
            .astype(float))

def top_by_frequency(monthly: pd.DataFrame, n: int) -> List[str]:
    freq = monthly.groupby("Description")["Qty"].apply(lambda s: (s>0).sum())
    return freq.sort_values(ascending=False).head(n).index.tolist()

def winsorize_series(y: pd.Series, q: float) -> pd.Series:
    if len(y) < 6: return y
    cap = y.quantile(q)
    return y.clip(upper=cap)

def seasonal_naive_forecast(y: pd.Series, m: int, steps: int) -> pd.Series:
    if len(y) < m:
        last = y.iloc[-1] if len(y) else 0.0
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        return pd.Series([last]*steps, index=idx)
    hist = y.iloc[-m:]
    rep = np.resize(hist.values, steps)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(rep, index=idx)

def mape_safe(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100) if mask.any() else np.nan

def wape(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    denom = np.sum(np.abs(yt))
    return float(np.sum(np.abs(yt-yp))/denom*100) if denom > 0 else np.nan

def mpe(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    mask = yt != 0
    return float(np.mean(((yp[mask]-yt[mask])/yt[mask]))*100) if mask.any() else np.nan

def mase(y_true, y_pred, insample, m: int = 12) -> float:
    ins = np.asarray(insample, float)
    if len(ins) <= m:
        denom = np.mean(np.abs(np.diff(ins))) if len(ins) > 1 else 1.0
    else:
        denom = np.mean(np.abs(ins[m:] - ins[:-m]))
    denom = denom if denom != 0 else 1.0
    return float(np.mean(np.abs(np.asarray(y_true,float)-np.asarray(y_pred,float))) / denom)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def is_intermittent(y: pd.Series, thr: float = 0.40) -> bool:
    return (y == 0).mean() >= thr

def clip_nonneg(s: pd.Series) -> pd.Series:
    return s.clip(lower=0)

# ----- Croston-SBA for intermittent demand -----
def croston_sba(y: pd.Series, alpha: float = 0.1, steps: int = DEFAULT_FORECAST_STEPS) -> pd.Series:
    yv = y.values.astype(float)
    n = len(yv); z_hat = None; p_hat = None; interval = 0
    for t in range(n):
        interval += 1
        if yv[t] > 0:
            z_hat = yv[t] if z_hat is None else z_hat + alpha * (yv[t] - z_hat)
            p_hat = interval if p_hat is None else p_hat + alpha * (interval - p_hat)
            interval = 0
    if z_hat is None:
        fc = np.zeros(steps)
    else:
        level = (z_hat / max(p_hat, 1e-8)) * (1 - alpha/2.0)
        fc = np.array([level]*steps)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return pd.Series(fc, index=idx)

# ----- Month dummies for SARIMAX -----
def month_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.get_dummies(index.month)
    d.index = index
    d.columns = [f"m{c:02d}" for c in d.columns]
    return d

def future_month_dummies(last_idx: pd.Timestamp, steps: int) -> pd.DataFrame:
    idx = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return month_dummies(idx)

# ----- Model grids -----
def fit_ets_grid(train: pd.Series, m: int, use_log: bool):
    configs = [
        ("add","add",False,None),
        ("add","add",True,None),
        ("add","mul",False,None),
        ("add","mul",True,None),
        (None,"add",False,None),
        (None,"mul",False,None),
        ("add","add",False,"log"),
        ("add","mul",False,"log"),
    ]
    best_fit, best_aic, best_cfg = None, float("inf"), None
    y = np.log1p(train) if use_log else train
    for trend, seas, damped, boxcox in configs:
        use_season = seas is not None and len(train) >= 24
        try:
            model = ExponentialSmoothing(
                y,
                trend=trend,
                seasonal=(seas if use_season else None),
                seasonal_periods=(m if use_season else None),
                damped_trend=(damped if trend else False),
                initialization_method="estimated",
                use_boxcox=boxcox,
            )
            fit = model.fit(optimized=True, remove_bias=True)
            aic = getattr(fit, "aic", np.inf)
            if aic < best_aic:
                best_fit, best_aic, best_cfg = fit, aic, (trend,seas,damped,"log1p" if use_log else None)
        except: pass
    return best_fit, best_cfg

def ets_forecast(fit, steps, last_idx, use_log):
    fc = fit.forecast(steps)
    if use_log: fc = np.expm1(fc)
    fc.index = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return clip_nonneg(fc)

def fit_sarima_grid(train: pd.Series, m: int, use_log: bool):
    pdq = [(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1),(2,1,2)]
    PDQ = [(0,1,0,m),(0,1,1,m),(1,1,0,m),(1,1,1,m)]
    y = np.log1p(train) if use_log else train
    best_fit, best_aic, best_params = None, float("inf"), None
    for order in pdq:
        for seas in PDQ:
            try:
                fit = SARIMAX(y, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if fit.aic < best_aic:
                    best_fit, best_aic, best_params = fit, fit.aic, (order, seas, "log1p" if use_log else None)
            except: pass
    return best_fit, best_params

def sarima_forecast(fit, steps, last_idx, use_log):
    fc = fit.forecast(steps)
    if use_log: fc = np.expm1(fc)
    fc.index = pd.date_range(last_idx + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    return clip_nonneg(fc)

def fit_sarimax_month_dummies(train: pd.Series, m: int, use_log: bool):
    exog = month_dummies(train.index)
    pdq = [(0,1,1),(1,1,1),(1,0,1)]
    PDQ = [(0,1,1,m),(1,1,1,m),(0,1,0,m)]
    y = np.log1p(train) if use_log else train
    best_fit, best_aic, best_params = None, float("inf"), None
    for order in pdq:
        for seas in PDQ:
            try:
                fit = SARIMAX(y, exog=exog, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if fit.aic < best_aic:
                    best_fit, best_aic, best_params = fit, fit.aic, (order, seas, "month_dummies", "log1p" if use_log else None)
            except: pass
    return best_fit, best_params

def sarimax_forecast(fit, last_idx, steps, use_log):
    exf = future_month_dummies(last_idx, steps)
    fc = fit.forecast(steps, exog=exf)
    if use_log: fc = np.expm1(fc)
    fc.index = exf.index
    return clip_nonneg(fc)

# ---------- Calibration ----------
def calibrate(y_true: np.ndarray, y_pred: np.ndarray, lo: float = 0.3, hi: float = 3.0) -> float:
    """Median ratio of true/pred, clipped—reduces systematic bias."""
    yp = np.maximum(np.asarray(y_pred, float), 1e-8)
    r = np.asarray(y_true, float) / yp
    r = np.clip(r, lo, hi)
    return float(np.median(r))

# ---------- Plotting ----------
def plot_pretty_holdout(desc, train, test, preds_map, metric_rows, out_png, title_prefix):
    fig = plt.figure(figsize=(13,7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)
    ax = fig.add_subplot(gs[0])

    ax.plot(train.index, train.values, label="Training Data", lw=2)
    ax.plot(test.index,  test.values,  label="Test Data", color="black", lw=2)

    palette = {
        "ETS": "tab:orange", "SARIMA": "tab:green",
        "SARIMAX": "tab:red", "Croston-SBA": "tab:cyan"
    }
    for name, series in preds_map.items():
        ax.plot(series.index, series.values, label=f"{name} Forecast", lw=2, ls="--",
                color=palette.get(name, None))

    ax.grid(True, ls=":", alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for t in ax.get_xticklabels():
        t.set_rotation(45); t.set_ha("right")

    ax.set_title(f"{title_prefix} — {desc}")
    ax.set_xlabel("Month"); ax.set_ylabel("Qty")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    header = ["Model","Params","MAE","MSE","RMSE","MAPE%","MASE","WAPE%","MPE%"]
    table_data = []
    for (name, params, mae, mse, rm, mape_v, mase_v, wape_v, mpe_v) in metric_rows:
        table_data.append([
            name, str(params),
            f"{mae:,.2f}", f"{mse:,.2f}", f"{rm:,.2f}",
            f"{mape_v:,.2f}", f"{mase_v:,.3f}", f"{wape_v:,.2f}", f"{mpe_v:,.2f}"
        ])
    tbl = ax2.table(cellText=table_data, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.15)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+","_", name.lower()).strip("_")

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default=None, help="Optional path to a single CSV/XLSX")
    p.add_argument("--top", type=int, default=DEFAULT_TOPN)
    p.add_argument("--steps", type=int, default=DEFAULT_FORECAST_STEPS)
    p.add_argument("--recent", type=int, default=24, help="Use only last N months for model fitting")
    p.add_argument("--winsor", type=float, default=0.98, help="Winsorize upper quantile (0-1)")
    p.add_argument("--log", dest="use_log", action="store_true", help="Use log1p inside models")
    p.add_argument("--no-log", dest="use_log", action="store_false")
    p.set_defaults(use_log=True)
    p.add_argument("--no-calibrate", dest="do_cal", action="store_false")
    p.set_defaults(do_cal=True)
    args = p.parse_args()

    files: List[Path] = [Path(args.file)] if args.file else (sorted(list(DATA_DIR.glob("*.csv")))+sorted(list(DATA_DIR.glob("*.xlsx"))))
    if not files:
        print(f"No files found. Put CSV/XLSX in {DATA_DIR} or pass --file.")
        sys.exit(0)

    for f in files:
        try:
            monthly = load_monthly(f)
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue

        for desc in top_by_frequency(monthly, args.top):
            y = series_for(monthly, desc)
            if len(y) < 12:
                print(f"[skip-short] {desc} in {f.name}")
                continue

            # 1) cap spikes
            y = winsorize_series(y, args.winsor)

            # 2) 80/20 holdout for evaluation
            n = len(y)
            k = max(1, int(round(n*0.20)))
            train_full, test = y.iloc[:-k], y.iloc[-k:]

            # 3) Use RECENT WINDOW for fitting to avoid stale high periods
            train = train_full.copy()
            if len(train) > args.recent:
                train = train.iloc[-args.recent:]

            preds_map = {}
            metric_rows = []

            if is_intermittent(y):
                # Croston route
                cro = croston_sba(train, steps=len(test)); cro.index = test.index
                preds_map["Croston-SBA"] = cro
                mae = mean_absolute_error(test, cro); mse = mean_squared_error(test, cro)
                rm  = rmse(test, cro); mp = mape_safe(test.values, cro.values)
                ms  = mase(test.values, cro.values, train.values, m=SEASONAL_PERIODS)
                wa  = wape(test.values, cro.values); me = mpe(test.values, cro.values)
                metric_rows.append(("Croston-SBA", "alpha=0.1", mae, mse, rm, mp, ms, wa, me))

                # forward forecast (optionally calibrate)
                fc6 = croston_sba(y, steps=args.steps)
                if args.do_cal and len(test) >= 1:
                    cal = calibrate(test.values, cro.values)
                    fc6 = fc6 * cal
                fc6.to_csv(OUT_ROOT / f"{sanitize(desc)}_forecast.csv", header=["forecast"])
            else:
                # ETS
                efit, ecfg = fit_ets_grid(train, SEASONAL_PERIODS, args.use_log)
                if efit is not None:
                    ep = ets_forecast(efit, len(test), train.index[-1], args.use_log); ep.index = test.index
                    preds_map["ETS"] = ep
                    metric_rows.append(("ETS", str(ecfg),
                        mean_absolute_error(test, ep), mean_squared_error(test, ep),
                        rmse(test, ep), mape_safe(test.values, ep.values),
                        mase(test.values, ep.values, train.values, m=SEASONAL_PERIODS),
                        wape(test.values, ep.values), mpe(test.values, ep.values)))

                # SARIMA
                sfit, scfg = fit_sarima_grid(train, SEASONAL_PERIODS, args.use_log)
                if sfit is not None:
                    sp = sarima_forecast(sfit, len(test), train.index[-1], args.use_log); sp.index = test.index
                    preds_map["SARIMA"] = sp
                    metric_rows.append(("SARIMA", str(scfg),
                        mean_absolute_error(test, sp), mean_squared_error(test, sp),
                        rmse(test, sp), mape_safe(test.values, sp.values),
                        mase(test.values, sp.values, train.values, m=SEASONAL_PERIODS),
                        wape(test.values, sp.values), mpe(test.values, sp.values)))

                # SARIMAX with month dummies
                xfit, xcfg = fit_sarimax_month_dummies(train, SEASONAL_PERIODS, args.use_log)
                if xfit is not None:
                    xp = sarimax_forecast(xfit, train.index[-1], len(test), args.use_log); xp.index = test.index
                    preds_map["SARIMAX"] = xp
                    metric_rows.append(("SARIMAX", str(xcfg),
                        mean_absolute_error(test, xp), mean_squared_error(test, xp),
                        rmse(test, xp), mape_safe(test.values, xp.values),
                        mase(test.values, xp.values, train.values, m=SEASONAL_PERIODS),
                        wape(test.values, xp.values), mpe(test.values, xp.values)))

                # choose best by MASE
                if metric_rows:
                    best_row = sorted(metric_rows, key=lambda r: np.nan_to_num(r[6], nan=1e9))[0]
                    best_name = best_row[0]
                else:
                    best_name = None

                # build forward forecast from best model
                if best_name == "ETS" and efit is not None:
                    fc6 = ets_forecast(efit, args.steps, y.index[-1], args.use_log)
                elif best_name == "SARIMA" and sfit is not None:
                    fc6 = sarima_forecast(sfit, args.steps, y.index[-1], args.use_log)
                elif best_name == "SARIMAX" and xfit is not None:
                    fc6 = sarimax_forecast(xfit, y.index[-1], args.steps, args.use_log)
                else:
                    fc6 = seasonal_naive_forecast(y, SEASONAL_PERIODS, args.steps)

                # bias calibration from the 80/20 holdout
                if args.do_cal and best_name in preds_map:
                    cal = calibrate(test.values, preds_map[best_name].values)
                    fc6 = fc6 * cal

                fc6.to_csv(OUT_ROOT / f"{sanitize(desc)}_forecast.csv", header=["forecast"])

            # save the chart for this SKU
            png_path = OUT_ROOT / f"{sanitize(desc)}_80_20.png"
            plot_pretty_holdout(
                desc, train, test, preds_map, metric_rows,
                png_path, title_prefix="SARIMA/ETS/SARIMAX — 80% Train, 20% Test"
            )

    print(f"Done. Check outputs in: {OUT_ROOT}")

if __name__ == "__main__":
    main()
