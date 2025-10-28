# analytics/descriptive/clustering.py
import os, json, warnings
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.dpi"] = 120

def _new_fig(figsize=(16,10), bottom=0.26, top=0.92, left=0.11, right=0.98):
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    ax.tick_params(labelsize=10)
    return fig, ax

def _finalize(fig: plt.Figure, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    print(f"✓ Saved figure: {path}")
    try:
        plt.show()
    finally:
        plt.close(fig)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _to_json(df: pd.DataFrame, path: Path):
    path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["date","item_code","description","qty","sales","category","tab"]:
        if c not in df.columns: df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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
    meta = g.groupby(["item_code","description","category","tab"], as_index=False).agg(
        total_qty=("qty","sum"),
        total_sales=("sales","sum"),
        months_active=("ym","nunique"),
    )
    meta["avg_price"] = meta.apply(lambda r: (r["total_sales"]/r["total_qty"]) if r["total_qty"] else np.nan, axis=1)

    def _per_sku_stats(sub: pd.DataFrame):
        sub = sub.sort_values("ym")
        x = np.arange(len(sub))
        y = sub["qty"].astype(float).to_numpy()
        mean_q = np.mean(y) if len(y) else np.nan
        std_q  = np.std(y, ddof=0) if len(y) else np.nan
        cv_q   = (std_q/mean_q) if (mean_q and mean_q != 0) else np.nan
        slope  = np.polyfit(x, y, 1)[0] if len(y) >= 2 else 0.0
        return pd.Series({"mean_monthly_qty": mean_q, "std_monthly_qty": std_q, "cv_monthly_qty": cv_q, "trend_qty_slope": slope})

    stats_df = g.groupby(["item_code","description","category","tab"]).apply(_per_sku_stats).reset_index()
    feats = meta.merge(stats_df, on=["item_code","description","category","tab"], how="left")

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

def _choose_k(Xs: np.ndarray, k_min=3, k_max=10, random_state=42,
              use_sampling=False, sample_size=5000) -> Tuple[int, float]:
    n = len(Xs)
    if n < k_min: return (0, -1.0)
    if use_sampling and n > sample_size:
        rng = np.random.RandomState(random_state)
        eval_idx = rng.choice(np.arange(n), size=sample_size, replace=False)
    else:
        eval_idx = np.arange(n)

    best_k, best_score = None, -1.0
    for k in range(k_min, min(k_max, n) + 1):
        try:
            km = KMeans(n_clusters=k, n_init=15, random_state=random_state)
            labels_full = km.fit_predict(Xs)
            score = silhouette_score(Xs[eval_idx], labels_full[eval_idx], metric="euclidean")
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return (best_k or 0, float(best_score))

def _cluster(feats: pd.DataFrame, scaler: RobustScaler, random_state=42,
             use_sampling_for_silhouette=False) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feat_cols = ["total_sales","total_qty","avg_price","months_active","mean_monthly_qty","cv_monthly_qty","trend_qty_slope"]
    X = feats[feat_cols].to_numpy()
    Xs = scaler.fit_transform(X)
    k, sil = _choose_k(
        Xs, k_min=3, k_max=10, random_state=random_state,
        use_sampling=use_sampling_for_silhouette
    )
    if k <= 0:
        feats["cluster"] = -1
        return feats, {"silhouette": -1.0, "k": 0}
    km = KMeans(n_clusters=k, n_init=15, random_state=random_state)
    feats["cluster"] = km.fit_predict(Xs)
    return feats, {"silhouette": sil, "k": int(k)}

def _summarize_clusters(labeled: pd.DataFrame) -> pd.DataFrame:
    feat_cols = ["total_sales","total_qty","avg_price","months_active","mean_monthly_qty","cv_monthly_qty","trend_qty_slope"]
    summary = (labeled
               .groupby("cluster")
               .agg(count=("item_code","count"),
                    total_sales_med=("total_sales","median"),
                    total_qty_med=("total_qty","median"),
                    avg_price_med=("avg_price","median"),
                    months_active_med=("months_active","median"),
                    mean_monthly_qty_med=("mean_monthly_qty","median"),
                    cv_monthly_qty_med=("cv_monthly_qty","median"),
                    trend_qty_slope_med=("trend_qty_slope","median"))
               .reset_index())

    gmed = labeled[feat_cols].median(numeric_only=True)
    def tag(r):
        tags = []
        if r["total_sales_med"] >= gmed["total_sales"] and r["cv_monthly_qty_med"] <= gmed["cv_monthly_qty"]:
            tags.append("Core Staples")
        if r["cv_monthly_qty_med"] > gmed["cv_monthly_qty"] and r["trend_qty_slope_med"] > 0:
            tags.append("Seasonal/Spiky")
        if r["avg_price_med"] > gmed["avg_price"] and r["total_qty_med"] < gmed["total_qty"]:
            tags.append("Niche/High-Price")
        if r["trend_qty_slope_med"] < 0:
            tags.append("Declining")
        if not tags:
            tags.append("Mixed")
        return ", ".join(sorted(set(tags)))
    summary["persona"] = summary.apply(tag, axis=1)
    return summary

def _fmt_int(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x)):,}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x)):,}"))

def _set_zoom(ax, xdata, ydata, xq=(2,98), yq=(2,98)):
    x = pd.Series(xdata).dropna().astype(float)
    y = pd.Series(ydata).dropna().astype(float)
    if len(x) >= 5:
        lo, hi = np.percentile(x, list(xq))
        if lo < hi: ax.set_xlim(lo, hi)
    if len(y) >= 5:
        lo, hi = np.percentile(y, list(yq))
        if lo < hi: ax.set_ylim(lo, hi)

def _legend_below(ax, ncol=6):
    # drop legend farther down; we increased bottom margin to make room
    ax.legend(title="Cluster", loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=ncol, frameon=False)

def _scatter(labeled: pd.DataFrame, title: str, out_file: Path,
             zoom=False, faint_others=False, focus_cluster=None):
    if labeled.empty: return
    fig, ax = _new_fig(figsize=(16,10), bottom=0.32)

    groups = labeled.groupby("cluster")
    for cid, sub in groups:
        kw = {}
        if faint_others and focus_cluster is not None and cid != focus_cluster:
            kw.update(dict(alpha=0.08, color="#888888", s=16))
        else:
            kw.update(dict(alpha=0.7, s=24))
        ax.scatter(sub["total_sales"], sub["total_qty"], label=f"{cid}", **kw)

    ax.set_title(title)
    ax.set_xlabel("Total Sales"); ax.set_ylabel("Total Quantity")
    _fmt_int(ax); ax.grid(True, linewidth=0.3, alpha=0.4)

    if zoom:
        _set_zoom(ax, labeled["total_sales"], labeled["total_qty"], xq=(2,98), yq=(2,98))

    _legend_below(ax)
    _finalize(fig, out_file)

def _export_top_items_json(labeled: pd.DataFrame, path: Path, topn: int = 15):
    rows = []
    for cid, sub in labeled.groupby("cluster"):
        best = (sub.sort_values("total_sales", ascending=False)
                    .head(topn)[["item_code","description","total_sales","total_qty"]]
                    .assign(cluster=int(cid)))
        rows.append(best)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["cluster","item_code","description","total_sales","total_qty"])
    path.write_text(out.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

def cluster_all(df: pd.DataFrame, out_dir: str,
                by_tab: bool=True, by_category: bool=False,
                use_sampling_for_silhouette: bool=False,
                label_points: bool=False, max_point_labels: int=0,
                make_zoomed_variant: bool=True,
                make_per_cluster_panels: bool=True,
                random_state=42) -> Dict[str, Any]:

    base = _ensure_dir(Path(out_dir))
    sub_global = base
    sub_tab = _ensure_dir(base / "clusters_by_tab")
    sub_cat = _ensure_dir(base / "clusters_by_category")
    sub_sum = _ensure_dir(base / "cluster_summaries")
    sub_top = _ensure_dir(base / "top_items")
    sub_panels = _ensure_dir(base / "panels")

    df = _normalize(df)
    feats = _build_feature_table(df)
    if feats.empty:
        (sub_global / "clusters_global.csv").write_text("", encoding="utf-8")
        return {"k": 0, "silhouette": -1.0, "outputs": []}

    scaler = RobustScaler()

    # ---- Global
    labeled_global, meta_global = _cluster(
        feats.copy(), scaler, random_state=random_state,
        use_sampling_for_silhouette=use_sampling_for_silhouette
    )
    (sub_global / "clusters_global.csv").write_text(labeled_global.to_csv(index=False), encoding="utf-8")
    _to_json(labeled_global, sub_global / "clusters_global.json")

    _scatter(
        labeled_global,
        f"Global Clusters (k={meta_global['k']}, silhouette={meta_global['silhouette']:.3f})",
        sub_global / "fig_global_scatter.png",
        zoom=False
    )
    if make_zoomed_variant:
        _scatter(
            labeled_global,
            f"Global Clusters — Zoomed (k={meta_global['k']}, silhouette={meta_global['silhouette']:.3f})",
            sub_global / "fig_global_scatter_zoom.png",
            zoom=True
        )

    if make_per_cluster_panels:
        for cid in sorted(labeled_global["cluster"].unique()):
            _scatter(
                labeled_global,
                f"Global Cluster {cid} (highlighted)",
                sub_panels / f"global_cluster_{cid}.png",
                zoom=True, faint_others=True, focus_cluster=cid
            )

    summary_global = _summarize_clusters(labeled_global)
    (sub_sum / "global.csv").write_text(summary_global.to_csv(index=False), encoding="utf-8")
    _to_json(summary_global, sub_sum / "global.json")

    members = labeled_global[["cluster","item_code","description","category","tab","total_sales","total_qty"]].copy()
    (base / "cluster_members.csv").write_text(members.to_csv(index=False), encoding="utf-8")
    _export_top_items_json(labeled_global, sub_top / "global.json", topn=15)

    # ---- By Tab
    tabs_info = []
    if by_tab:
        for tab, sub in feats.groupby("tab"):
            if not len(sub): continue
            labeled, meta = _cluster(
                sub.copy(), RobustScaler(), random_state=random_state,
                use_sampling_for_silhouette=use_sampling_for_silhouette
            )
            safe = "none" if pd.isna(tab) or str(tab).lower() in ("nan","none") else str(tab).replace("/", "_")
            (sub_tab / f"{safe}.csv").write_text(labeled.to_csv(index=False), encoding="utf-8")
            _to_json(labeled, sub_tab / f"{safe}.json")

            _scatter(
                labeled,
                f"Clusters by Tab = {tab} (k={meta['k']}, sil={meta['silhouette']:.3f})",
                sub_tab / f"fig_tab_{safe}_scatter.png",
                zoom=False
            )
            if make_zoomed_variant:
                _scatter(
                    labeled,
                    f"Clusters by Tab = {tab} — Zoomed (k={meta['k']}, sil={meta['silhouette']:.3f})",
                    sub_tab / f"fig_tab_{safe}_scatter_zoom.png",
                    zoom=True
                )
            if make_per_cluster_panels:
                pdir = _ensure_dir(sub_tab / f"panels_{safe}")
                for cid in sorted(labeled["cluster"].unique()):
                    _scatter(
                        labeled,
                        f"Tab={tab} Cluster {cid} (highlighted)",
                        pdir / f"tab_{safe}_cluster_{cid}.png",
                        zoom=True, faint_others=True, focus_cluster=cid
                    )

            s = _summarize_clusters(labeled)
            (sub_sum / f"by_tab_{safe}.csv").write_text(s.to_csv(index=False), encoding="utf-8")
            _to_json(s, sub_sum / f"by_tab_{safe}.json")
            _export_top_items_json(labeled, sub_top / f"by_tab_{safe}.json", topn=15)
            tabs_info.append({"tab": str(tab), "k": meta["k"], "silhouette": meta["silhouette"]})

    # ---- By Category
    cats_info = []
    if by_category:
        for cat, sub in feats.groupby("category"):
            if not len(sub): continue
            labeled, meta = _cluster(
                sub.copy(), RobustScaler(), random_state=random_state,
                use_sampling_for_silhouette=use_sampling_for_silhouette
            )
            safe = "none" if pd.isna(cat) or str(cat).lower() in ("nan","none") else str(cat).replace("/", "_")
            (sub_cat / f"{safe}.csv").write_text(labeled.to_csv(index=False), encoding="utf-8")
            _to_json(labeled, sub_cat / f"{safe}.json")

            _scatter(
                labeled,
                f"Clusters by Category = {cat} (k={meta['k']}, sil={meta['silhouette']:.3f})",
                sub_cat / f"fig_cat_{safe}_scatter.png",
                zoom=False
            )
            if make_zoomed_variant:
                _scatter(
                    labeled,
                    f"Clusters by Category = {cat} — Zoomed (k={meta['k']}, sil={meta['silhouette']:.3f})",
                    sub_cat / f"fig_cat_{safe}_scatter_zoom.png",
                    zoom=True
                )
            if make_per_cluster_panels:
                pdir = _ensure_dir(sub_cat / f"panels_{safe}")
                for cid in sorted(labeled["cluster"].unique()):
                    _scatter(
                        labeled,
                        f"Category={cat} Cluster {cid} (highlighted)",
                        pdir / f"cat_{safe}_cluster_{cid}.png",
                        zoom=True, faint_others=True, focus_cluster=cid
                    )

            s = _summarize_clusters(labeled)
            (sub_sum / f"by_category_{safe}.csv").write_text(s.to_csv(index=False), encoding="utf-8")
            _to_json(s, sub_sum / f"by_category_{safe}.json")
            _export_top_items_json(labeled, sub_top / f"by_category_{safe}.json", topn=15)
            cats_info.append({"category": str(cat), "k": meta["k"], "silhouette": meta["silhouette"]})

    return {
        "global": {"k": meta_global["k"], "silhouette": meta_global["silhouette"]},
        "by_tab": tabs_info,
        "by_category": cats_info,
        "outputs_dir": str(base)
    }

if __name__ == "__main__":
    pass
