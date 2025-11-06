# analytics/descriptive/descriptive.py
import os, sys, json
from pathlib import Path
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from kpi import compute_kpis
from mba import run_mba
from clustering import cluster_all

def load_data_from_database():
    """Load data from the warehouse.fact_sales table in the database"""
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

        print(f"[OK] Loaded data from database: {len(df):,} rows x {len(df.columns)} columns")

        return df

    except Exception as e:
        print(f"X Error loading data from database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    out_dir = os.path.join(script_dir, "descriptive_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Script directory: {script_dir}")
    print(f"Output directory: {out_dir}")

    df = load_data_from_database()

    # Prepare subfolders
    kpi_out = os.path.join(out_dir, "kpi_output")
    mba_out = os.path.join(out_dir, "mba_output")
    clu_out = os.path.join(out_dir, "clustering_output")
    os.makedirs(kpi_out, exist_ok=True)
    os.makedirs(mba_out, exist_ok=True)
    os.makedirs(clu_out, exist_ok=True)

    medicine_stats = df.groupby('description').agg({
        'sales': 'sum',
        'qty': 'sum',
        'profit': 'sum'
    }).rename(columns={'sales': 'total_sales', 'qty': 'total_qty', 'profit': 'total_profit'})
    medicine_stats['profit_margin'] = (medicine_stats['total_profit'] / medicine_stats['total_sales'] * 100).fillna(0)
    medicine_stats = medicine_stats.reset_index().rename(columns={'description': 'medicine'})

    # ---------------- [1/3] KPIs ----------------
    print("\n[1/3] Running KPIs ...")
    kpi_res = compute_kpis(df, kpi_out)
    print("KPIs done.")
    print(f"   Avg monthly sales growth: {kpi_res.get('avg_monthly_growth_rate')}")

    # ---------------- [2/3] MBA -----------------
    print("\n[2/3] Running Market-Basket (MBA) ...")
    mba_res = run_mba(df, mba_out)
    print(f"MBA done. Rules generated: {mba_res.get('rules_count')}")

    # ---------------- [3/3] Clustering ----------
    print("\n[3/3] Running Clustering (DBSCAN) ...")
    clu_res = cluster_all(df, clu_out)
    print(f"Clustering done. Global n_clusters={clu_res['global']['n_clusters']}, n_noise={clu_res['global']['n_noise']}")

    # ---------------- Manifest ------------------
    manifest = {
        "kpi_outputs": [str(p) for p in Path(kpi_out).glob("*.*")],
        "mba_outputs": [str(p) for p in Path(mba_out).glob("*.*")],
        "clustering_outputs": [str(p) for p in Path(clu_out).rglob("*.*")],
    }
    Path(out_dir, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "data_overview": {
            "total_rows_analyzed": len(df),
            "total_sales": float(medicine_stats['total_sales'].sum()),
            "total_quantity": int(medicine_stats['total_qty'].sum()),
            "total_profit": float(medicine_stats['total_profit'].sum()),
            "avg_profit_margin_pct": float(medicine_stats['profit_margin'].mean()),
        },
        "top_5_products": medicine_stats.nlargest(5, 'total_sales')[['medicine', 'total_sales', 'total_qty', 'profit_margin']].to_dict('records'),
        "kpi_analysis": {
            "kpi_metrics_generated": len([p for p in Path(kpi_out).glob("*.csv")]),
            "avg_monthly_growth_rate": kpi_res.get('avg_monthly_growth_rate'),
        },
        "market_basket_analysis": {
            "rules_generated": mba_res.get('rules_count', 0),
            "support_threshold": mba_res.get('support', 0.01),
            "confidence_threshold": mba_res.get('confidence', 0.1),
        },
        "clustering_analysis": {
            "clusters_identified": clu_res['global']['n_clusters'],
            "noise_points": clu_res['global']['n_noise'],
            "total_points_clustered": len(df),
        },
        "timestamp": str(pd.Timestamp.now()),
    }
    Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAll done! Outputs under:", out_dir)

if __name__ == "__main__":
    main()
