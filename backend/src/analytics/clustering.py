# clustering.py
import os, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

@dataclass
class ClusterConfig:
    SMALL_DATASET_THRESHOLD: int = 50_000
    LARGE_DATASET_THRESHOLD: int = 500_000
    SAMPLE_SIZE_LARGE: int = 100_000
    DBSCAN_EPS_SMALL: float = 0.5
    DBSCAN_EPS_LARGE: float = 0.3
    DBSCAN_MIN_SAMPLES_SMALL: int = 5
    DBSCAN_MIN_SAMPLES_LARGE: int = 10
    KMEANS_MIN_CLUSTERS: int = 3
    KMEANS_MAX_CLUSTERS: int = 15
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 100

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
        tmp["year_month"] = tmp["date"].dt.to_period("M")
        k = max(1, sample_size // tmp["year_month"].nunique())
        sampled = tmp.groupby("year_month", group_keys=False).apply(
            lambda x: x.sample(min(len(x), k), random_state=42)
        ).drop(columns="year_month")
    else:
        sampled = df.sample(n=sample_size, random_state=42)
    print(f"✓ Sampled {len(sampled):,} rows")
    return sampled, True

def prepare_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("item_code").agg({
        "qty": ["sum", "mean", "std"],
        "sales": ["sum", "mean", "std"],
        "cost": ["sum", "mean"],
        "profit": ["sum", "mean"],
        "receipt": "nunique"
    }).reset_index()
    grp.columns = ["_".join([c for c in col if c]).strip("_") for col in grp.columns]
    grp = grp.fillna(0)
    if "description" in df.columns:
        desc_map = df.groupby("item_code")["description"].first()
        grp["description"] = grp["item_code"].map(desc_map)
    print(f"✓ Created {len(grp):,} item-level features")
    return grp

def level1_dbscan(features_df: pd.DataFrame, cfg: ClusterConfig, dataset_size: int):
    feature_cols = [c for c in features_df.columns if c not in ["item_code", "description"]]
    X = features_df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    eps = cfg.DBSCAN_EPS_SMALL if dataset_size < cfg.SMALL_DATASET_THRESHOLD else cfg.DBSCAN_EPS_LARGE
    min_samples = cfg.DBSCAN_MIN_SAMPLES_SMALL if dataset_size < cfg.SMALL_DATASET_THRESHOLD else cfg.DBSCAN_MIN_SAMPLES_LARGE
    print(f"DBSCAN eps={eps}, min_samples={min_samples}")
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(Xs)
    features_df["cluster_l1"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✓ L1 clusters: {n_clusters} | noise: {(labels == -1).sum()}")
    return features_df, Xs

def _optimal_k(X, min_k, max_k):
    Ks = list(range(min_k, max(max_k + 1, min_k + 1)))
    best_k = Ks[0]
    best_sil = -1
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else -1
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k

def level2_kmeans(features_df: pd.DataFrame, Xs: np.ndarray, cfg: ClusterConfig):
    features_df["cluster_l2"] = -1
    for c1 in sorted([c for c in features_df["cluster_l1"].unique() if c != -1]):
        mask = features_df["cluster_l1"] == c1
        chunk = Xs[mask]
        if len(chunk) < cfg.KMEANS_MIN_CLUSTERS:
            features_df.loc[mask, "cluster_l2"] = 0
            continue
        max_k = min(cfg.KMEANS_MAX_CLUSTERS, max(2, len(chunk) // 2))
        k = _optimal_k(chunk, cfg.KMEANS_MIN_CLUSTERS, max_k)
        sub = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(chunk)
        features_df.loc[mask, "cluster_l2"] = sub
        print(f"  L1={c1}: {len(chunk):,} items → {k} sub-clusters")
    features_df["cluster_combined"] = features_df["cluster_l1"].astype(str) + "_" + features_df["cluster_l2"].astype(str)
    print(f"✓ L2 complete: {features_df['cluster_combined'].nunique()} combined clusters")
    return features_df

def analyze_clusters(features_df: pd.DataFrame, original_df: pd.DataFrame):
    cluster_col = "cluster_combined" if "cluster_combined" in features_df.columns else "cluster_l1"
    cmap = features_df[["item_code", cluster_col]].set_index("item_code")[cluster_col]
    dfw = original_df.copy()
    dfw["cluster"] = dfw["item_code"].map(cmap)
    stats = dfw.groupby("cluster").agg({
        "item_code": "nunique",
        "qty": "sum",
        "sales": "sum",
        "profit": "sum",
        "receipt": "nunique"
    }).reset_index()
    stats.columns = ["cluster", "n_items", "total_qty", "total_sales", "total_profit", "n_transactions"]
    stats["avg_profit_margin"] = (stats["total_profit"] / stats["total_sales"] * 100).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    return stats.sort_values("total_sales", ascending=False), dfw

def generate_insights(features_df: pd.DataFrame) -> pd.DataFrame:
    cluster_col = "cluster_combined" if "cluster_combined" in features_df.columns else "cluster_l1"
    feats = [c for c in features_df.columns if c not in ["item_code", "description", "cluster_l1", "cluster_l2", "cluster_combined"]]
    out = []
    for cid in features_df[cluster_col].unique():
        if str(cid) == "-1":  # skip noise
            continue
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
        # simple category logic
        total_sales = part["sales_sum"].sum()
        if total_sales > part["sales_sum"].quantile(0.75):
            row["category"] = "High-Value Premium" if row["avg_profit_margin"] > 20 else "High-Volume Movers"
        elif row["avg_profit_margin"] > 25:
            row["category"] = "Niche Premium"
        elif row["avg_transactions"] > part["receipt_nunique"].quantile(0.75):
            row["category"] = "Frequent Purchases"
        else:
            row["category"] = "Standard Products"
        out.append(row)
    return pd.DataFrame(out).sort_values("avg_total_sales", ascending=False)

def save_clustering_outputs(features_df, cluster_stats, dfw, insights_df, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cols = ["item_code", "cluster_l1", "cluster_l2", "cluster_combined"]
    if "description" in features_df.columns:
        cols.append("description")
    features_df[cols].to_csv(os.path.join(output_dir, "clustering_item_clusters.csv"), index=False)
    cluster_stats.to_csv(os.path.join(output_dir, "clustering_cluster_stats.csv"), index=False)
    insights_df.to_csv(os.path.join(output_dir, "clustering_insights.csv"), index=False)
    tx_cols = ["receipt", "item_code", "qty", "sales", "profit", "cluster"]
    if "description" in dfw.columns:
        tx_cols.insert(2, "description")
    dfw[tx_cols].to_csv(os.path.join(output_dir, "clustering_transactions.csv"), index=False)

def create_clustering_visualizations(features_df: pd.DataFrame, Xs: np.ndarray, output_dir: str, cfg: ClusterConfig):
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    XP = pca.fit_transform(Xs)
    cluster_col = "cluster_combined" if "cluster_combined" in features_df.columns else "cluster_l1"
    # scatter
    plt.figure(figsize=cfg.FIGURE_SIZE)
    # map clusters to integers for colors
    uniq = features_df[cluster_col].astype(str).unique()
    idx_map = {c:i for i,c in enumerate(uniq)}
    colors = features_df[cluster_col].astype(str).map(idx_map)
    sc = plt.scatter(XP[:,0], XP[:,1], c=colors, cmap="tab20", alpha=0.6, s=50)
    plt.colorbar(sc, label="Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Item Clustering Visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustering_scatter.png"), dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    # sizes
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
    # convert dates if present
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.fillna(0)
    df_s, was_sampled = apply_sampling(df, cfg)
    feats = prepare_clustering_features(df_s)
    feats, Xs = level1_dbscan(feats, cfg, len(df))
    feats = level2_kmeans(feats, Xs, cfg)
    stats, dfw = analyze_clusters(feats, df_s)
    insights = generate_insights(feats)
    save_clustering_outputs(feats, stats, dfw, insights, output_dir)
    create_clustering_visualizations(feats, Xs, output_dir, cfg)
    print("✓ Clustering complete → CSV/PNG saved.")
