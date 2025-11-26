# analytics/descriptive/descriptive.py
import os, sys, json
from pathlib import Path
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from kpi import compute_kpis
from mba import run_mba
from dbscan import cluster_all as dbscan_cluster_all
from clustering import cluster_all as kmeans_cluster_all

# ---------------- CSV helpers (kept for potential reuse/debug) ----------------
def _read_csv_robust(path_or_buf):
    """Attempt to read CSV with multiple encodings"""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path_or_buf, encoding=enc, engine="python")
        except Exception:
            continue
    return pd.read_csv(path_or_buf)

def _load_dataframe(source_value: str) -> pd.DataFrame:
    """Load dataframe from local CSV file (unused in DB workflow, but kept for flexibility)"""
    if not source_value or not os.path.exists(source_value):
        print(" No valid file found."); sys.exit(1)
    print(f" Loading CSV from file: {source_value}")
    return _read_csv_robust(source_value)

# ---------------- DB loader ----------------
def load_data_from_database() -> pd.DataFrame:
    """Load data from warehouse.fact_sales in the warehouse DB"""
    print("Loading data from database...")
    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set in environment variables")

    try:
        conn = psycopg2.connect(dsn)
        query = """
        SELECT
            fs.date_key AS date,
            fs.receipt_number AS receipt,
            fs.sales_order_number AS so,
            p.item_code AS item_code,
            p.description AS description,
            p.category AS category,
            p.tab AS tab,
            fs.expiration_date AS expiration,
            fs.quantity_sold AS qty,
            fs.unit AS unit,
            fs.discount_rate AS discount,
            fs.sales_amount AS sales,
            fs.cost_amount AS cost,
            fs.profit_amount AS profit,
            fs.payment AS payment,
            fs.cashier_id AS cashier_id,
            fs.txn_type AS txn_type
        FROM warehouse.fact_sales fs
        JOIN warehouse.dim_product p ON fs.product_key = p.product_key
        JOIN warehouse.dim_date d ON fs.date_key = d.date_key
        ORDER BY fs.date_key, fs.receipt_number
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            raise ValueError("No data found in warehouse.fact_sales table.")

        print(f"Loaded data from database: {len(df):,} rows x {len(df.columns)} columns")
        return df

    except Exception as e:
        print(f" Error loading data from database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ---------------- main driver ----------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    out_dir = os.path.join(script_dir, "descriptive_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Script directory: {script_dir}")
    print(f"Output directory: {out_dir}")

    print("\n" + "=" * 70)
    print("LOADING DATA FROM DATABASE")
    print("=" * 70)

    # --- DATA SOURCE (DB) ---
    df = load_data_from_database()

    # Prepare subfolders
    kpi_out = os.path.join(out_dir, "kpi_output")
    mba_out = os.path.join(out_dir, "mba_output")
    clustering_out = os.path.join(out_dir, "clustering_output")
    os.makedirs(kpi_out, exist_ok=True)
    os.makedirs(mba_out, exist_ok=True)
    os.makedirs(clustering_out, exist_ok=True)

    # Simple per-medicine stats (for summary.json)
    medicine_stats = df.groupby("description").agg({
        "sales": "sum",
        "qty": "sum",
        "profit": "sum"
    }).rename(columns={
        "sales": "total_sales",
        "qty": "total_qty",
        "profit": "total_profit"
    })
    medicine_stats["profit_margin"] = (
        medicine_stats["total_profit"] / medicine_stats["total_sales"] * 100
    ).fillna(0)
    medicine_stats = (
        medicine_stats.reset_index()
        .rename(columns={"description": "medicine"})
    )

    # ---------------- [1/3] KPIs ----------------
    print("\n[1/3] Running KPIs ...")
    kpi_res = compute_kpis(df, kpi_out)
    print(" KPIs done.")
    print(f"   Avg monthly sales growth (CAGR): {kpi_res.get('avg_monthly_growth_rate_cagr')}")

    # ---------------- [2/3] MBA -----------------
    print("\n[2/3] Running Market-Basket (MBA) ...")
    mba_res = run_mba(df, mba_out)
    print(f" MBA done. Rules generated: {mba_res.get('rules_count')}")

    # -------- [3a/3] DBSCAN Clustering (uses new global/by_tab/by_category structure) --------
    print("\n[3a/3] Running DBSCAN Clustering ...")
    dbscan_res = dbscan_cluster_all(df, clustering_out)

    dbscan_global = dbscan_res.get("global", {}) or {}
    dbscan_by_cat = dbscan_res.get("by_category", []) or []

    print(
        " DBSCAN done. "
        f"Global clusters={dbscan_global.get('n_clusters', 0)}, "
        f"Categories with graphs={len(dbscan_by_cat)}"
    )

    # -------- [3b/3] KMeans Clustering (unchanged) --------
    print("\n[3b/3] Running KMeans Clustering ...")
    kmeans_res = kmeans_cluster_all(df, clustering_out)
    kmeans_by_cat = kmeans_res.get("by_category", []) or []
    print(f" KMeans done. Categories with graphs: {len(kmeans_by_cat)}")

    # ---------------- Manifest ------------------
    manifest = {
        "kpi_outputs": [str(p) for p in Path(kpi_out).glob("*.*")],
        "mba_outputs": [str(p) for p in Path(mba_out).glob("*.*")],
        "clustering_outputs": [str(p) for p in Path(clustering_out).rglob("*.*")],
    }
    Path(out_dir, "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # ---------------- High-level summary for frontend ------------------
    summary = {
        "data_overview": {
            "total_rows_analyzed": int(len(df)),
            "total_sales": float(medicine_stats["total_sales"].sum()),
            "total_quantity": int(medicine_stats["total_qty"].sum()),
            "total_profit": float(medicine_stats["total_profit"].sum()),
            "avg_profit_margin_pct": float(medicine_stats["profit_margin"].mean()),
        },
        "top_5_products": medicine_stats.nlargest(
            5, "total_sales"
        )[["medicine", "total_sales", "total_qty", "profit_margin"]].to_dict("records"),
        "kpi_analysis": {
            "kpi_metrics_generated": len([p for p in Path(kpi_out).glob("*.csv")]),
            "avg_monthly_growth_rate_cagr": kpi_res.get("avg_monthly_growth_rate_cagr"),
            "avg_monthly_growth_rate_mean": kpi_res.get("avg_monthly_growth_rate_mean"),
        },
        "market_basket_analysis": {
            "rules_generated": mba_res.get("rules_count", 0),
            "transactions": mba_res.get("transactions", 0),
            "items": mba_res.get("items", 0),
        },
        "clustering": {
            "dbscan": {
                "global_n_clusters": dbscan_global.get("n_clusters", 0),
                "global_n_noise": dbscan_global.get("n_noise", 0),
                "tabs_analyzed": len(dbscan_res.get("by_tab", []) or []),
                "categories_analyzed": len(dbscan_by_cat),
                "dbscan_output_dir": dbscan_res.get("outputs_dir"),
            },
            "kmeans": {
                "tabs_analyzed": len(kmeans_res.get("by_tab", []) or []),
                "categories_analyzed": len(kmeans_by_cat),
                # you can add more KMeans stats here later if your kmeans.cluster_all returns them
            },
            "total_cluster_graphs_generated": len(
                list(Path(clustering_out).rglob("fig_*.png"))
            ),
        },
        "timestamp": str(pd.Timestamp.now()),
    }
    Path(out_dir, "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n All done! Outputs under:", out_dir)

if __name__ == "__main__":
    main()
