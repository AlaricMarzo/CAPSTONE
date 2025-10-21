# clustering.py
import os, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


@dataclass
class ClusterConfig:
    SMALL_DATASET_THRESHOLD: int = 50_000
    LARGE_DATASET_THRESHOLD: int = 500_000
    SAMPLE_SIZE_LARGE: int = 100_000

    # DBSCAN defaults (for longer windows)
    DBSCAN_EPS_SMALL: float = 0.30
    DBSCAN_EPS_LARGE: float = 0.22
    DBSCAN_MIN_SAMPLES_SMALL: int = 10
    DBSCAN_MIN_SAMPLES_LARGE: int = 20

    # KMeans bounds
    KMEANS_MIN_CLUSTERS: int = 6
    KMEANS_MAX_CLUSTERS: int = 12

    # If DBSCAN puts ≥80% of items in one cluster → fallback to global KMeans
    L1_DOMINANCE_THRESHOLD: float = 0.80

    # Target size used to pick granular K
    TARGET_CLUSTER_SIZE: int = 80

    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 110


# ---------- helpers ----------
def _span_in_months(df: pd.DataFrame) -> float:
    if "date" not in df.columns:
        return np.inf
    d = pd.to_datetime(df["date"], errors="coerce")
    dmin, dmax = d.min(), d.max()
    if pd.isna(dmin) or pd.isna(dmax):
        return np.inf
    return max(1.0, (dmax - dmin).days / 30.44)


def apply_sampling(df: pd.DataFrame, cfg: ClusterConfig):
    n = len(df)
    if n <= cfg.SMALL_DATASET_THRESHOLD:
        print(f"✓ Dataset size ({n:,}) - Using full dataset")
        return df, False
    if n <= cfg.LARGE_DATASET_THRESHOLD:
        print(f"⚠ Medium dataset ({n:,}) - Using full dataset with optimized params")
        return df, False
    sample_size = min(cfg.SAMPLE_SIZE_LARGE, n)
    if "date" in df.columns and np.issubdtype(df["date"].dtype, np.datetime64):
        tmp = df.copy()
        tmp["year_month"] = pd.to_datetime(tmp["date"], errors="coerce").dt.to_period("M")
        k = max(1, sample_size // tmp["year_month"].nunique())
        sampled = tmp.groupby("year_month", group_keys=False).apply(
            lambda x: x.sample(min(len(x), k), random_state=42)
        ).drop(columns="year_month")
    else:
        sampled = df.sample(n=sample_size, random_state=42)
    print(f"✓ Sampled {len(sampled):,} rows")
    return sampled, True


def prepare_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("item_code").agg(
        qty_sum=("qty", "sum"),
        qty_mean=("qty", "mean"),
        qty_std=("qty", "std"),
        sales_sum=("sales", "sum"),
        sales_mean=("sales", "mean"),
        sales_std=("sales", "std"),
        cost_sum=("cost", "sum"),
        cost_mean=("cost", "mean"),
        profit_sum=("profit", "sum"),
        profit_mean=("profit", "mean"),
        receipt_nunique=("receipt", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max")
    ).reset_index()

    # engineered vars for short windows
    adays = (pd.to_datetime(grp["last_date"]) - pd.to_datetime(grp["first_date"])).dt.days.fillna(0)
    aweeks = (adays / 7.0).clip(lower=1.0)
    grp["active_weeks"] = aweeks
    grp["qty_velocity"] = grp["qty_sum"] / grp["active_weeks"]
    grp["sales_velocity"] = grp["sales_sum"] / grp["active_weeks"]
    grp["qty_cv"] = (grp["qty_std"] / grp["qty_mean"]).replace([np.inf, -np.inf], 0).fillna(0)
    grp["sales_cv"] = (grp["sales_std"] / grp["sales_mean"]).replace([np.inf, -np.inf], 0).fillna(0)

    dmax = pd.to_datetime(df["date"]).max() if "date" in df.columns else None
    if pd.notna(dmax):
        grp["recency_days"] = (dmax - pd.to_datetime(grp["last_date"])).dt.days.clip(lower=0)
    else:
        grp["recency_days"] = 0

    if "description" in df.columns:
        grp["description"] = df.groupby("item_code")["description"].first().values

    # IQR trim
    num_cols = [c for c in grp.columns if c not in ["item_code", "description", "first_date", "last_date"]]
    for col in num_cols:
        Q1, Q3 = grp[col].quantile(0.25), grp[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        grp = grp[(grp[col] >= lb) & (grp[col] <= ub)]

    # log1p on heavy-tailed totals/counts
    skew_cols = [c for c in grp.columns if any(s in c for s in ["_sum", "nunique", "velocity"]) or c in ["recency_days"]]
    for col in skew_cols:
        grp[col] = np.log1p(grp[col].clip(lower=0))

    grp = grp.fillna(0)
    print(f"✓ Features ready: {len(grp):,} items after trimming + log1p + engineered vars")
    return grp


def level1_dbscan(features_df: pd.DataFrame, cfg: ClusterConfig, dataset_size: int):
    feature_cols = [c for c in features_df.columns
                    if c not in ["item_code", "description", "first_date", "last_date"]]
    X = features_df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    eps = cfg.DBSCAN_EPS_SMALL if dataset_size < cfg.SMALL_DATASET_THRESHOLD else cfg.DBSCAN_EPS_LARGE
    min_samples = cfg.DBSCAN_MIN_SAMPLES_SMALL if dataset_size < cfg.SMALL_DATASET_THRESHOLD else cfg.DBSCAN_MIN_SAMPLES_LARGE

    print(f"DBSCAN eps={eps}, min_samples={min_samples}")
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(Xs)
    features_df = features_df.copy()
    features_df["cluster_l1"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    print(f"✓ L1 clusters: {n_clusters} | noise: {noise}")
    return features_df, Xs


def _optimal_k(X, min_k, max_k, output_dir: str = None):
    n = len(X)
    if n < 3:
        return 1
    max_k = max(2, min(max_k, n - 1))
    min_k = max(2, min(min_k, max_k))
    Ks = list(range(min_k, max_k + 1))
    wcss, best_k, best_sil, best_db = [], Ks[0], -1.0, float("inf")
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        if len(np.unique(labels)) >= 2:
            try:
                sil = silhouette_score(X, labels)
            except Exception:
                sil = -1.0
            try:
                db = davies_bouldin_score(X, labels)
            except Exception:
                db = float("inf")
        else:
            sil, db = -1.0, float("inf")
        if sil > best_sil or (sil == best_sil and db < best_db):
            best_sil, best_db, best_k = sil, db, k
    if output_dir and len(Ks) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(Ks, wcss, 'bo-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS')
        plt.title('Elbow Method for Optimal K')
        plt.savefig(os.path.join(output_dir, "elbow_plot.png"), dpi=110, bbox_inches="tight")
        plt.close()
    print(f"  Optimal K={best_k} (Sil={best_sil:.3f}, DB={best_db:.3f})")
    return best_k


def _granular_k(n_items: int, cfg: ClusterConfig) -> int:
    k = int(np.clip(np.ceil(n_items / max(1, cfg.TARGET_CLUSTER_SIZE)),
                    cfg.KMEANS_MIN_CLUSTERS, cfg.KMEANS_MAX_CLUSTERS))
    return max(2, k)


def level2_kmeans(features_df: pd.DataFrame, Xs: np.ndarray, cfg: ClusterConfig,
                  output_dir: str, force_global: bool, force_k: int | None = None):
    feats = features_df.copy()
    feats["cluster_l2"] = -1

    if force_global:
        k = force_k if force_k and force_k >= 2 else _optimal_k(Xs, cfg.KMEANS_MIN_CLUSTERS, cfg.KMEANS_MAX_CLUSTERS, output_dir)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        glabels = km.fit_predict(Xs)
        feats["cluster_l1"] = glabels
        feats["cluster_l2"] = 0
        feats["cluster_combined"] = feats["cluster_l1"].astype(str)
        return feats

    for l1_id in sorted(feats["cluster_l1"].unique()):
        if l1_id == -1:
            continue
        idx = feats.index[feats["cluster_l1"] == l1_id]
        X = Xs[feats.index.get_indexer(idx)]
        n = len(idx)
        if n < 3 or np.allclose(X.std(axis=0), 0):
            feats.loc[idx, "cluster_l2"] = 0
            continue
        max_k = min(cfg.KMEANS_MAX_CLUSTERS, n - 1)
        min_k = max(2, cfg.KMEANS_MIN_CLUSTERS)
        if max_k < 2:
            feats.loc[idx, "cluster_l2"] = 0
            continue
        k = _optimal_k(X, min_k, max_k, output_dir)
        if k < 2 or k > n:
            feats.loc[idx, "cluster_l2"] = 0
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        l2 = km.fit_predict(X)
        feats.loc[idx, "cluster_l2"] = l2 if len(np.unique(l2)) >= 2 else 0

    feats["cluster_combined"] = feats.apply(
        lambda r: f"{int(r['cluster_l1'])}_{int(r['cluster_l2'])}" if r["cluster_l1"] != -1 else "-1",
        axis=1
    )
    return feats


def analyze_clusters(features_df: pd.DataFrame, original_df: pd.DataFrame):
    cluster_col = "cluster_combined" if "cluster_combined" in features_df.columns else "cluster_l1"
    cmap = features_df[["item_code", cluster_col]].set_index("item_code")[cluster_col]
    dfw = original_df.copy()
    dfw["cluster"] = dfw["item_code"].map(cmap)
    stats = dfw.groupby("cluster").agg(
        n_items=("item_code", "nunique"),
        total_qty=("qty", "sum"),
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
        n_transactions=("receipt", "nunique")
    ).reset_index()
    stats["avg_profit_margin"] = (stats["total_profit"] / stats["total_sales"] * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    return stats.sort_values("total_sales", ascending=False), dfw


def generate_insights(features_df: pd.DataFrame) -> pd.DataFrame:
    cluster_col = "cluster_combined" if "cluster_combined" in features_df.columns else "cluster_l1"
    cols = ["cluster_id","n_items","avg_total_sales","avg_total_profit","avg_profit_margin",
            "avg_qty_sold","avg_transactions","representative_items"]

    valid_ids = [cid for cid in features_df[cluster_col].unique() if str(cid) != "-1"]
    if len(valid_ids) == 0:
        # Return an empty but well-formed DataFrame (prevents KeyError)
        return pd.DataFrame(columns=cols)

    out = []
    for cid in valid_ids:
        part = features_df[features_df[cluster_col] == cid]
        row = {
            "cluster_id": cid,
            "n_items": len(part),
            "avg_total_sales": part["sales_sum"].mean(),
            "avg_total_profit": part["profit_sum"].mean(),
            "avg_profit_margin": (part["profit_sum"].sum() / part["sales_sum"].sum() * 100) if part["sales_sum"].sum() > 0 else 0,
            "avg_qty_sold": part["qty_sum"].mean(),
            "avg_transactions": part["receipt_nunique"].mean(),
            "representative_items": " | ".join(part.nlargest(3, "sales_sum")["description"].astype(str).tolist()) if "description" in part.columns else "N/A"
        }
        out.append(row)
    return pd.DataFrame(out)[cols].sort_values("avg_total_sales", ascending=False)


def save_clustering_outputs(features_df, cluster_stats, dfw, insights_df, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cols = ["item_code", "cluster_l1"]
    if "cluster_l2" in features_df.columns: cols.append("cluster_l2")
    if "cluster_combined" in features_df.columns: cols.append("cluster_combined")
    if "description" in features_df.columns: cols.append("description")
    features_df[cols].to_csv(os.path.join(output_dir, "clustering_item_clusters.csv"), index=False)
    cluster_stats.to_csv(os.path.join(output_dir, "clustering_cluster_stats.csv"), index=False)
    insights_df.to_csv(os.path.join(output_dir, "clustering_insights.csv"), index=False)
    tx_cols = ["receipt", "item_code", "qty", "sales", "profit", "cluster"]
    if "description" in dfw.columns: tx_cols.insert(2, "description")
    dfw[tx_cols].to_csv(os.path.join(output_dir, "clustering_transactions.csv"), index=False)


def create_clustering_visualizations(features_df: pd.DataFrame, Xs: np.ndarray, output_dir: str, cfg: ClusterConfig):
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    XP = pca.fit_transform(Xs)
    cluster_col = "cluster_combined" if "cluster_combined" in features_df.columns else "cluster_l1"

    plt.figure(figsize=cfg.FIGURE_SIZE)
    uniq = features_df[cluster_col].astype(str).unique()
    idx_map = {c: i for i, c in enumerate(uniq)}
    colors = features_df[cluster_col].astype(str).map(idx_map)
    sc = plt.scatter(XP[:, 0], XP[:, 1], c=colors, cmap="tab20", alpha=0.65, s=40)
    plt.colorbar(sc, label="Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Item Clustering Visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustering_scatter.png"), dpi=cfg.DPI, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=cfg.FIGURE_SIZE)
    sizes = features_df[cluster_col].value_counts().head(20)
    plt.bar(range(len(sizes)), sizes.values)
    plt.xticks(range(len(sizes)), [f"C{i+1}" for i in range(len(sizes))])
    plt.xlabel("Cluster Rank")
    plt.ylabel("Number of Items")
    plt.title("Top 20 Cluster Sizes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_sizes.png"), dpi=cfg.DPI, bbox_inches="tight")
    plt.close()


def run_clustering_pipeline(df: pd.DataFrame, output_dir: str = "analytics_output"):
    cfg = ClusterConfig()
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.fillna(0)

    span_months = _span_in_months(df)
    short_window = span_months <= 6.0  # 6 months or less

    df_s, _ = apply_sampling(df, cfg)
    feats = prepare_clustering_features(df_s)
    feats, Xs = level1_dbscan(feats, cfg, len(df))

    # Decide strategy (robust version)
    counts = feats["cluster_l1"].value_counts(normalize=True)
    n_items = len(feats)
    has_any_cluster = set(feats["cluster_l1"].unique()) - {-1}  # exclude noise
    if len(has_any_cluster) == 0:
        top_frac = np.nan
    else:
        top_frac = counts.drop(labels=-1, errors="ignore").max()

    force_global = short_window or (pd.notna(top_frac) and top_frac >= cfg.L1_DOMINANCE_THRESHOLD) or (len(has_any_cluster) == 0)
    forced_k = _granular_k(n_items, cfg) if force_global else None
    if force_global:
        print(f"⚠ Using GLOBAL K-Means (granular) → K={forced_k} (span={span_months:.1f} months, items={n_items})")

    feats = level2_kmeans(feats, Xs, cfg, output_dir, force_global=force_global, force_k=forced_k)

    stats, dfw = analyze_clusters(feats, df_s)
    insights = generate_insights(feats)

    cluster_col = "cluster_combined" if "cluster_combined" in feats.columns else "cluster_l1"
    labels = feats[cluster_col].values
    if len(set(labels)) > 1:
        overall_sil = silhouette_score(Xs, labels)
        overall_db = davies_bouldin_score(Xs, labels)
        print(f"✓ Overall Quality: Silhouette={overall_sil:.3f}, Davies-Bouldin={overall_db:.3f}")
        with open(os.path.join(output_dir, "clustering_quality.txt"), "w") as f:
            f.write(f"Overall Silhouette Score: {overall_sil:.3f}\n")
            f.write(f"Overall Davies-Bouldin Score: {overall_db:.3f}\n")

    save_clustering_outputs(feats, stats, dfw, insights, output_dir)
    create_clustering_visualizations(feats, Xs, output_dir, cfg)
    print("✓ Clustering complete → CSV/PNG saved.")
