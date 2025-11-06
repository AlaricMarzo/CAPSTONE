# analytics/descriptive/clustering.py
import os, json, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- plotting (smaller window, no clipping) ---
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (10, 6)  # <— SMALLER DEFAULT WINDOW

warnings.filterwarnings("ignore")

# ---------------------------- helpers ----------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_json(df: pd.DataFrame, path: Path):
    path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

def _fmt_int_axis(ax):
    ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

def _new_fig(figsize=(10, 6), bottom=0.12, top=0.90, left=0.10, right=0.98):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_constrained_layout(False)
    fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    return fig, ax

def _pad_limits(vmin: float, vmax: float, pad_pct: float = 0.05) -> Tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return vmin, vmax
    if vmin == vmax:
        delta = abs(vmin) * pad_pct if vmin != 0 else 1.0
        return vmin - delta, vmax + delta
    span = vmax - vmin
    pad = span * pad_pct
    return vmin - pad, vmax + pad

# ---------------------------- normalization & features ----------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # expected columns: date, item_code, description, category, tab, qty, sales
    # fallback shims
    if "item_code" not in df.columns: df["item_code"] = ""
    if "category" not in df.columns: df["category"] = ""
    if "tab" not in df.columns: df["tab"] = ""
    for c in ["qty", "sales"]:
        if c not in df.columns: df[c] = 0
    # parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    for c in ["qty","sales"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["description"] = df["description"].astype(str).str.strip()
    df = df.dropna(subset=["date","description"])
    return df

def _build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M").astype(str)

    g = df.groupby(["item_code","description","category","tab","ym"], as_index=False).agg(
        qty=("qty","sum"),
        sales=("sales","sum"),
    )

    # per-SKU aggregates
    meta = g.groupby(["item_code","description","category","tab"], as_index=False).agg(
        total_qty=("qty","sum"),
        total_sales=("sales","sum"),
        months_active=("ym","nunique"),
    )
    meta["avg_price"] = meta.apply(
        lambda r: (r["total_sales"]/r["total_qty"]) if r["total_qty"] else np.nan, axis=1
    )

    # per-SKU monthly stats + trend
    def _per_sku_stats(sub: pd.DataFrame):
        sub = sub.sort_values("ym")
        x = np.arange(len(sub))
        y = sub["qty"].astype(float).to_numpy()
        mean_q = np.mean(y) if len(y) else np.nan
        std_q  = np.std(y, ddof=0) if len(y) else np.nan
        cv_q   = (std_q/mean_q) if (mean_q and mean_q != 0) else np.nan
        slope  = np.polyfit(x, y, 1)[0] if len(y) >= 2 else 0.0
        return pd.Series({"mean_monthly_qty": mean_q, "cv_monthly_qty": cv_q, "trend_qty_slope": slope})

    stats = g.groupby(["item_code","description","category","tab"]).apply(_per_sku_stats).reset_index()

    feats = meta.merge(stats, on=["item_code","description","category","tab"], how="left")
    feat_cols = ["total_sales","total_qty","avg_price","months_active","mean_monthly_qty","cv_monthly_qty","trend_qty_slope"]
    feats = feats.assign(
        total_sales=feats["total_sales"].astype(float),
        total_qty=feats["total_qty"].astype(float),
        avg_price=feats["avg_price"].astype(float),
        months_active=feats["months_active"].astype(float),
        mean_monthly_qty=feats["mean_monthly_qty"].astype(float),
        cv_monthly_qty=feats["cv_monthly_qty"].astype(float),
        trend_qty_slope=feats["trend_qty_slope"].astype(float),
    )
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
    return feats

# ---------------------------- self-setting K & fit ----------------------------
def _choose_k(X: np.ndarray, k_min=3, k_max=10, sample_size=5000, random_state=42) -> Tuple[int, float]:
    """
    Self-setting K using silhouette on a sample.
    """
    n = len(X)
    if n < k_min:
        return (0, -1.0)
    rs = np.random.RandomState(random_state)
    idx = np.arange(n)
    if n > sample_size:
        idx = rs.choice(idx, size=sample_size, replace=False)
    Xs = X[idx]

    best_k, best_score = None, -1.0
    for k in range(k_min, min(k_max, n) + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels_full = km.fit_predict(X)
            labels = labels_full[idx]
            score = silhouette_score(Xs, labels, metric="euclidean")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    if best_k is None:
        best_k = max(k_min, 2)
        best_score = -1.0
    return best_k, float(best_score)

def _fit_kmeans(feats: pd.DataFrame, random_state=42) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cols = ["total_sales","total_qty","avg_price","months_active","mean_monthly_qty","cv_monthly_qty","trend_qty_slope"]
    X = feats[cols].to_numpy(dtype=float)
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    k, sil = _choose_k(Xs, k_min=3, k_max=10, random_state=random_state)
    if k <= 1:
        feats = feats.copy()
        feats["cluster"] = 0
        return feats, {"k": 1, "silhouette": sil}

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)

    out = feats.copy()
    out["cluster"] = labels.astype(int)
    return out, {"k": int(k), "silhouette": sil}

# ---------------------------- summaries & plotting ----------------------------
def _summarize(clusters_df: pd.DataFrame) -> pd.DataFrame:
    g = clusters_df.groupby("cluster", as_index=False).agg(
        n=("description","nunique"),
        total_sales=("total_sales","sum"),
        total_qty=("total_qty","sum"),
        avg_price_med=("avg_price","median"),
        months_active_med=("months_active","median"),
        mean_monthly_qty_med=("mean_monthly_qty","median"),
        cv_monthly_qty_med=("cv_monthly_qty","median"),
        trend_qty_slope_med=("trend_qty_slope","median"),
    )

    # quick persona tags
    def tag(r):
        tags = []
        if r["total_sales"] >= g["total_sales"].median(): tags.append("High-Sales")
        if r["total_qty"] >= g["total_qty"].median():     tags.append("High-Volume")
        if r["avg_price_med"] >= g["avg_price_med"].median(): tags.append("Premium")
        if r["trend_qty_slope_med"] > 0: tags.append("Rising")
        if r["trend_qty_slope_med"] < 0: tags.append("Declining")
        return ", ".join(tags) if tags else "Mixed"

    g["persona"] = g.apply(tag, axis=1)
    return g.sort_values("cluster").reset_index(drop=True)

def _plot_scatter(df: pd.DataFrame, title: str, out_png: Path):
    """
    Compact scatter; legend below; no clipping; integer tick labels.
    """
    if df.empty:
        return
    fig, ax = _new_fig(figsize=(10, 6), bottom=0.18, top=0.88, left=0.12, right=0.98)

    # point padding to avoid cutoff at edges
    xmin, xmax = _pad_limits(df["total_sales"].min(), df["total_sales"].max(), 0.05)
    ymin, ymax = _pad_limits(df["total_qty"].min(),   df["total_qty"].max(),   0.05)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.margins(0.02)

    clusters = sorted(df["cluster"].unique().tolist())
    cmap = plt.cm.get_cmap("tab10", len(clusters))
    for i, c in enumerate(clusters):
        sub = df[df["cluster"] == c]
        ax.scatter(sub["total_sales"], sub["total_qty"], s=16, alpha=0.85, label=f"Cluster {c}", color=cmap(i))

    _fmt_int_axis(ax)
    ax.set_title(title)
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Total Quantity")

    # legend under plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(5, len(clusters)), frameon=False)

    # finalize: save with small figure size
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

# ---------------------------- main driver ----------------------------
def cluster_all(df: pd.DataFrame, out_dir: str, random_state=42) -> Dict[str, Any]:
    """
    Produces:
      clustering_output/
        clusters_global.(csv|json)
        clusters_by_tab/<tab>.csv|json
        clusters_by_category/<category>.csv|json
        cluster_summaries/global.(csv|json), by_tab_<tab>.(csv|json), by_category_<cat>.(csv|json)
        PNGs: fig_global.png, fig_tab_<tab>.png, fig_cat_<cat>.png
    """
    base = _ensure_dir(Path(out_dir))
    sub_global = base
    sub_tab = _ensure_dir(base / "clusters_by_tab")
    sub_cat = _ensure_dir(base / "clusters_by_category")
    sub_sum = _ensure_dir(base / "cluster_summaries")

    df = _normalize(df)
    feats = _build_feature_table(df)
    if feats.empty:
        (sub_global / "clusters_global.csv").write_text("", encoding="utf-8")
        return {"k": 0, "silhouette": -1.0, "outputs": []}

    # -------- global --------
    global_df, meta_global = _fit_kmeans(feats, random_state=random_state)
    global_df.to_csv(sub_global / "clusters_global.csv", index=False, encoding="utf-8")
    _to_json(global_df, sub_global / "clusters_global.json")
    _summarize(global_df).to_csv(sub_sum / "global.csv", index=False, encoding="utf-8")
    _to_json(_summarize(global_df), sub_sum / "global.json")
    _plot_scatter(global_df, f"Global Clusters (k={meta_global['k']}, silhouette={meta_global['silhouette']:.3f})",
                  sub_global / "fig_global.png")

    # -------- by TAB --------
    tabs_info = []
    for tab, sub in feats.groupby("tab", dropna=False):
        subk, meta = _fit_kmeans(sub, random_state=random_state)
        fn_root = f"{str(tab)}".replace("/", "_")
        subk.to_csv(sub_tab / f"{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(subk, sub_tab / f"{fn_root}.json")
        sm = _summarize(subk)
        sm.to_csv(sub_sum / f"by_tab_{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(sm, sub_sum / f"by_tab_{fn_root}.json")
        _plot_scatter(subk, f"TAB: {tab} (k={meta['k']}, silhouette={meta['silhouette']:.3f})",
                      sub_tab / f"fig_tab_{fn_root}.png")
        tabs_info.append({"tab": str(tab), "k": meta["k"], "silhouette": meta["silhouette"]})

    # -------- by CATEGORY --------
    cats_info = []
    for cat, sub in feats.groupby("category", dropna=False):
        subk, meta = _fit_kmeans(sub, random_state=random_state)
        fn_root = f"{str(cat)}".replace("/", "_")
        subk.to_csv(sub_cat / f"{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(subk, sub_cat / f"{fn_root}.json")
        sm = _summarize(subk)
        sm.to_csv(sub_sum / f"by_category_{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(sm, sub_sum / f"by_category_{fn_root}.json")
        _plot_scatter(subk, f"CATEGORY: {cat} (k={meta['k']}, silhouette={meta['silhouette']:.3f})",
                      sub_cat / f"fig_cat_{fn_root}.png")
        cats_info.append({"category": str(cat), "k": meta["k"], "silhouette": meta["silhouette"]})

    return {
        "global": {"k": meta_global["k"], "silhouette": meta_global["silhouette"]},
        "by_tab": tabs_info,
        "by_category": cats_info,
        "outputs_dir": str(base)
    }

# ---------------------------- CLI (optional) ----------------------------
if __name__ == "__main__":
    # Example: python clustering.py "path/to/data.csv" "clustering_output"
    import sys
    if len(sys.argv) >= 3:
        csv_path = sys.argv[1]
        out_dir = sys.argv[2]
        df = pd.read_csv(csv_path)
        res = cluster_all(df, out_dir)
        print(json.dumps(res, indent=2))
    else:
        print("Usage: python clustering.py <csv_path> <output_dir>")
