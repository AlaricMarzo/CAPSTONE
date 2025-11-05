# analytics/descriptive/dbscan.py
import os, json, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# --- plotting tuned for readability & side legend ---
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 11

warnings.filterwarnings("ignore")

# ============================ helpers ============================
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_json(df: pd.DataFrame, path: Path):
    path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

def _fmt_int_axis(ax):
    ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

def _new_fig(figsize=(11.5, 6.8), bottom=0.12, top=0.90, left=0.10, right=0.76):
    """
    Leaves ~24% of figure width on the right for the side legend panel.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_constrained_layout(False)
    fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    return fig, ax

def _pad_limits(vmin: float, vmax: float, pad_pct: float = 0.06) -> Tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return vmin, vmax
    if vmin == vmax:
        delta = abs(vmin) * pad_pct if vmin != 0 else 1.0
        return vmin - delta, vmax + delta
    span = vmax - vmin
    pad = span * pad_pct
    return vmin - pad, vmax + pad

def _human(n):
    try:
        return f"{int(round(float(n))):,} "
    except Exception:
        return str(n)

# ============================ normalization & features ============================
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "item_code" not in df.columns: df["item_code"] = ""
    if "category" not in df.columns:  df["category"] = ""
    if "tab" not in df.columns:       df["tab"] = ""
    for c in ["qty", "sales", "profit"]:
        if c not in df.columns: df[c] = 0
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    for c in ["qty","sales","profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["description"] = df["description"].astype(str).str.strip()
    df = df.dropna(subset=["date","description"])
    return df

def _build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M").astype(str)

    g = df.groupby(["item_code","description","category","tab","ym"], as_index=False).agg(
        qty=("qty","sum"), sales=("sales","sum"), profit=("profit","sum"),
    )

    meta = g.groupby(["item_code","description","category","tab"], as_index=False).agg(
        total_qty=("qty","sum"),
        total_sales=("sales","sum"),
        total_profit=("profit","sum"),
        months_active=("ym","nunique"),
    )
    meta["avg_price"] = meta.apply(
        lambda r: (r["total_sales"]/r["total_qty"]) if r["total_qty"] else np.nan, axis=1
    )

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
    feats = feats.assign(
        total_sales=feats["total_sales"].astype(float),
        total_qty=feats["total_qty"].astype(float),
        avg_price=feats["avg_price"].astype(float),
        months_active=feats["months_active"].astype(float),
        mean_monthly_qty=feats["mean_monthly_qty"].astype(float),
        cv_monthly_qty=feats["cv_monthly_qty"].astype(float),
        trend_qty_slope=feats["trend_qty_slope"].astype(float),
    )
    feat_cols = ["total_sales","total_qty","avg_price","months_active","mean_monthly_qty","cv_monthly_qty","trend_qty_slope"]
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)
    return feats

# ============================ self-setting eps for DBSCAN ============================
def _choose_dbscan_params(X: np.ndarray, min_samples=5) -> Tuple[float, int]:
    """
    Compute eps for DBSCAN as the median distance to the min_samples-th nearest neighbor.
    """
    if len(X) < min_samples:
        return 0.0, min_samples
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    # Sort distances to the min_samples-th neighbor
    k_distances = np.sort(distances[:, -1])
    # Use median as eps
    eps = np.median(k_distances)
    return eps, min_samples

def _fit_dbscan(feats: pd.DataFrame, random_state=42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Cluster on 'total_sales' only, but keep 'total_qty' for visualization
    X = feats[["total_sales"]].to_numpy(dtype=float)
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    eps, min_samples = _choose_dbscan_params(Xs, min_samples=5)
    if eps == 0.0:
        feats = feats.copy()
        feats["cluster"] = 0
        return feats, {"n_clusters": 1, "n_noise": 0, "eps": eps, "min_samples": min_samples}

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(Xs)

    out = feats.copy()
    out["cluster"] = labels.astype(int)

    # Compute number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = out[out["cluster"] == -1]["total_qty"].sum() if -1 in labels else 0

    return out, {"n_clusters": n_clusters, "n_noise": n_noise, "eps": eps, "min_samples": min_samples}

# ============================ summaries & plotting ============================
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

def _cluster_legend_blocks(df: pd.DataFrame, max_lines: int = 10) -> List[str]:
    """
    Build human-readable text blocks per cluster for the side legend,
    including the other features actually used by DBSCAN.
    """
    def rng(s):
        s = pd.to_numeric(s, errors="coerce")
        return f"{_human(np.nanmin(s))} to {_human(np.nanmax(s))}"

    blocks = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]

        qty_rng   = rng(sub["total_qty"])
        sales_rng = rng(sub["total_sales"])
        price_med = np.nanmedian(pd.to_numeric(sub["avg_price"], errors="coerce"))
        act_med   = np.nanmedian(pd.to_numeric(sub["months_active"], errors="coerce"))
        mean_med  = np.nanmedian(pd.to_numeric(sub["mean_monthly_qty"], errors="coerce"))
        cv_med    = np.nanmedian(pd.to_numeric(sub["cv_monthly_qty"], errors="coerce"))
        slope_med = np.nanmedian(pd.to_numeric(sub["trend_qty_slope"], errors="coerce"))

        cluster_label = "Noise" if c == -1 else f"Cluster {c}"
        line1 = cluster_label
        line2 = f"Qty: {qty_rng}"
        line3 = f"Sales: {sales_rng}"
        line4 = f"Med Price: ₱{price_med:,.2f}"
        line5 = f"Active mo.: {act_med:,.0f}"
        line6 = f"Med Mean/mo.: {mean_med:,.0f}"
        line7 = f"CV: {cv_med:,.2f}"
        line8 = f"Trend slope (qty/mo.): {slope_med:,.2f}"

        block = [line1, line2, line3, line4, line5, line6, line7, line8]
        blocks.append("\n".join(block[:max_lines]))
    return blocks

def _draw_side_legend(fig, title: str, blocks: List[str]):
    """
    Creates a text panel at the right side with cluster summaries.
    """
    axp = fig.add_axes([0.78, 0.12, 0.20, 0.76])  # [left, bottom, width, height]
    axp.axis("off")
    txt = title + "\n\n" + "\n\n".join(blocks)
    axp.text(0.0, 1.0, txt, va="top", ha="left", fontsize=10, family="monospace", wrap=True)

def _scatter_plot(df: pd.DataFrame, title: str, out_png: Path, use_log: bool):
    """
    Clean scatter with side legend and optional log scaling.
    Saves to out_png. If use_log=True, appends '(log)' in the title.
    """
    if df.empty:
        return

    fig, ax = _new_fig()

    x = df["total_sales"].astype(float).to_numpy()
    y = df["total_qty"].astype(float).to_numpy()

    # Apply log scaling to both axes
    if use_log:
        x = np.clip(x, a_min=0, a_max=None) + 1.0
        y = np.clip(y, a_min=0, a_max=None) + 1.0
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total Sales (log scale)")
        ax.set_ylabel("Total Quantity (log scale)")
    else:
        ax.set_xlabel("Total Sales")
        ax.set_ylabel("Total Quantity")

    # Set axis limits to zoom in on the data range
    x_25, x_75 = np.percentile(x, 25), np.percentile(x, 75)
    y_25, y_75 = np.percentile(y, 25), np.percentile(y, 75)
    x_iqr = x_75 - x_25
    y_iqr = y_75 - y_25

    xmin, xmax = _pad_limits(x_25 - 1.5 * x_iqr, x_75 + 1.5 * x_iqr, 0.05)
    ymin, ymax = _pad_limits(y_25 - 1.5 * y_iqr, y_75 + 1.5 * x_iqr, 0.05)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Scatter plot for clusters
    clusters = sorted(df["cluster"].unique().tolist())
    cmap = plt.cm.get_cmap("tab10", len(clusters))

    for i, c in enumerate(clusters):
        sub = df[df["cluster"] == c]
        color = "gray" if c == -1 else cmap(i % 10)  # Gray for noise
        label = "Noise" if c == -1 else f"Cluster {c}"
        size = 30 if c == -1 else 50  # Smaller for noise
        alpha = 0.6 if c == -1 else 0.85  # More transparent for noise
        ax.scatter(
            sub["total_sales"], sub["total_qty"],
            s=size, alpha=alpha, color=color, edgecolors="none", label=label
        )

    # Title & legend
    full_title = title + (" (log view)" if use_log else " (linear view)")
    ax.set_title(full_title)

    # Standard legend (color ↔ cluster id) under the chart
    ax.legend(loc="upper center", bbox_to_anchor=(0.43, -0.12), ncol=min(5, len(clusters)), frameon=False)

    # Side legend with numeric ranges
    _draw_side_legend(fig, "Cluster summaries:", _cluster_legend_blocks(df))

    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

def _plot_both(df: pd.DataFrame, title_root: str, out_root: Path):
    """
    Saves two figures: linear and log-scale versions.
    """
    _scatter_plot(df, title_root, out_root.with_suffix("").with_name(out_root.stem + "_linear.png"), use_log=False)
    _scatter_plot(df, title_root, out_root.with_suffix("").with_name(out_root.stem + "_log.png"),    use_log=True)

# ============================ main driver ============================
def cluster_all(df: pd.DataFrame, out_dir: str, random_state=42) -> Dict[str, Any]:
    """
    Produces:
      clustering_output/
        clusters_global.(csv|json)
        clusters_by_tab/<tab>.csv|json
        clusters_by_category/<category>.csv|json
        cluster_summaries/global.(csv|json), by_tab_<tab>.(csv|json), by_category_<cat>.(csv|json)
        PNGs per group: *_linear.png and *_log.png (with side legend)
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
        return {"n_clusters": 0, "n_noise": 0, "outputs": []}

    # -------- global --------
    global_df, meta_global = _fit_dbscan(feats, random_state=random_state)
    global_df.to_csv(sub_global / "clusters_global.csv", index=False, encoding="utf-8")
    _to_json(global_df, sub_global / "clusters_global.json")
    sm_global = _summarize(global_df)
    sm_global.to_csv(sub_sum / "global.csv", index=False, encoding="utf-8")
    _to_json(sm_global, sub_sum / "global.json")
    if -1 in sm_global["cluster"].values:
        meta_global['n_noise'] = sm_global.loc[sm_global["cluster"] == -1, "n"].iloc[0]
    _plot_both(global_df, f"Global Clusters (n_clusters={meta_global['n_clusters']})",
               sub_global / "fig_global.png")

    # -------- by TAB --------
    tabs_info = []
    for tab, sub in feats.groupby("tab", dropna=False):
        subk, meta = _fit_dbscan(sub, random_state=random_state)
        fn_root = f"{str(tab)}".replace("/", "_") if str(tab) != "" else "UNKNOWN"
        subk.to_csv(sub_tab / f"{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(subk, sub_tab / f"{fn_root}.json")
        sm = _summarize(subk)
        sm.to_csv(sub_sum / f"by_tab_{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(sm, sub_sum / f"by_tab_{fn_root}.json")
        if -1 in sm["cluster"].values:
            meta['n_noise'] = sm.loc[sm["cluster"] == -1, "total_qty"].iloc[0]
        _plot_both(subk, f"TAB: {tab} (n_clusters={meta['n_clusters']})",
                   sub_tab / f"fig_tab_{fn_root}.png")
        tabs_info.append({"tab": str(tab), "n_clusters": meta["n_clusters"], "n_noise": meta["n_noise"]})

    # -------- by CATEGORY --------
    cats_info = []
    for cat, sub in feats.groupby("category", dropna=False):
        subk, meta = _fit_dbscan(sub, random_state=random_state)
        fn_root = f"{str(cat)}".replace("/", "_") if str(cat) != "" else "UNKNOWN"
        subk.to_csv(sub_cat / f"{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(subk, sub_cat / f"{fn_root}.json")
        sm = _summarize(subk)
        sm.to_csv(sub_sum / f"by_category_{fn_root}.csv", index=False, encoding="utf-8")
        _to_json(sm, sub_sum / f"by_category_{fn_root}.json")
        if -1 in sm["cluster"].values:
            meta['n_noise'] = sm.loc[sm["cluster"] == -1, "total_qty"].iloc[0]
        _plot_both(subk, f"CATEGORY: {cat} (n_clusters={meta['n_clusters']})",
                   sub_cat / f"fig_cat_{fn_root}.png")
        cats_info.append({"category": str(cat), "n_clusters": meta["n_clusters"], "n_noise": meta["n_noise"]})

    return {
        "global": {"n_clusters": meta_global["n_clusters"], "n_noise": meta_global["n_noise"]},
        "by_tab": tabs_info,
        "by_category": cats_info,
        "outputs_dir": str(base)
    }

# ============================ CLI ============================
if __name__ == "__main__":
    # Example: python dbscan.py "path/to/data.csv" "dbscan_output"
    import sys
    if len(sys.argv) >= 3:
        csv_path = sys.argv[1]
        out_dir = sys.argv[2]
        df = pd.read_csv(csv_path)
        res = cluster_all(df, out_dir)
        print(json.dumps(res, indent=2))
    else:
        print("Usage: python dbscan.py <csv_path> <output_dir>")
