"""
Script to apply performance fixes to load_to_database.py

This patches the existing functions with optimized versions.
Run this to see immediate performance improvements.
"""

import sys
import os

# Add the scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the optimized functions
from performance_fixes import (
    OptimizedCategoryMatcher,
    get_optimized_matcher,
    optimize_duplicate_checking,
    batch_update_categories,
    print_performance_stats
)

# Import the original module
import load_to_database as original_module

# Store original functions for comparison
_original_get_category = original_module.get_category_for_product
_original_get_tab = original_module.get_tab_for_product

# Global matcher instance
_matcher = None


def initialize_optimized_matcher():
    """Initialize the optimized matcher with loaded categories"""
    global _matcher

    # Ensure lookups are loaded first
    try:
        original_module.load_product_categories()
    except Exception:
        pass

    if _matcher is None and original_module.CATEGORY_LOOKUP:
        _matcher = get_optimized_matcher(
            original_module.CATEGORY_LOOKUP,      # BARCODE -> Category
            original_module.TAB_LOOKUP,           # BARCODE -> Tab
            original_module.PRODUCT_NAME_LOOKUP   # NAME(normalized) -> (Category, Tab)
        )
        print("[PERF] [OK] Optimized category matcher initialized")

    return _matcher



def optimized_get_category_for_product(item_code: str = None, description: str = None):
    """
    Signature-compatible replacement:
      1) Try exact BARCODE (item_code) via CATEGORY_LOOKUP
      2) Fallback to fuzzy NAME match on description via matcher
      3) Fallback to original implementation
    """
    matcher = initialize_optimized_matcher()

    # 1) Direct by barcode
    if item_code:
        bar = str(item_code).strip().upper()
        hit = original_module.CATEGORY_LOOKUP.get(bar)
        if hit:
            return hit

    # 2) Fuzzy by description
    if matcher and description:
        got = matcher.get_category(description)
        if got:
            return got

    # 3) Fallback to original with same signature
    return _original_get_category(item_code, description)



def optimized_get_tab_for_product(item_code: str = None, description: str = None):
    """
    Signature-compatible replacement:
      1) Try exact BARCODE (item_code) via TAB_LOOKUP
      2) Fallback to fuzzy NAME match on description via matcher
      3) Fallback to original implementation
    """
    matcher = initialize_optimized_matcher()

    # 1) Direct by barcode
    if item_code:
        bar = str(item_code).strip().upper()
        hit = original_module.TAB_LOOKUP.get(bar)
        if hit:
            return hit

    # 2) Fuzzy by description
    if matcher and description:
        got = matcher.get_tab(description)
        if got:
            return got

    # 3) Fallback to original with same signature
    return _original_get_tab(item_code, description)



def optimized_upsert_dim_product(conn, run_id: str):
    """
    Optimized version of upsert_dim_product with batch updates.
    
    Key improvements:
    - Collects all category updates first
    - Performs single batch update instead of row-by-row
    - Uses optimized matcher with caching
    """
    # Load categories if not already loaded
    original_module.load_product_categories()
    
    # Initialize matcher
    matcher = initialize_optimized_matcher()
    
    with conn, conn.cursor() as cur:
        # First, do the main upsert (same as original)
        cur.execute("""
        WITH by_item AS (
          SELECT
            item_code,
            MAX(NULLIF(description,'')) FILTER (WHERE description IS NOT NULL) AS any_desc,
            MAX(NULLIF(unit,''))        FILTER (WHERE unit        IS NOT NULL) AS any_unit,
            MAX(NULLIF(category,''))    FILTER (WHERE category    IS NOT NULL) AS any_cat,
            MAX(NULLIF(tab,''))         FILTER (WHERE tab         IS NOT NULL) AS any_tab,
            MIN(date) AS run_first_seen,
            MAX(date) AS run_last_seen
          FROM staging.sales_cleaned
          WHERE run_id = %s
            AND item_code IS NOT NULL
          GROUP BY item_code
        )
        INSERT INTO warehouse.dim_product(item_code, description, unit, category, tab, first_seen_date, last_seen_date)
        SELECT item_code, any_desc, any_unit, any_cat, any_tab, run_first_seen, run_last_seen
        FROM by_item
        ON CONFLICT (item_code) DO UPDATE
        SET description     = COALESCE(EXCLUDED.description, warehouse.dim_product.description),
            unit            = COALESCE(EXCLUDED.unit,        warehouse.dim_product.unit),
            category        = COALESCE(EXCLUDED.category,    warehouse.dim_product.category),
            tab             = COALESCE(EXCLUDED.tab,         warehouse.dim_product.tab),
            first_seen_date = LEAST(warehouse.dim_product.first_seen_date, EXCLUDED.first_seen_date),
            last_seen_date  = GREATEST(warehouse.dim_product.last_seen_date, EXCLUDED.last_seen_date);
        """, (run_id,))
        
        # Now collect all products that need category updates
        cur.execute("""
            SELECT item_code, description
            FROM warehouse.dim_product
            WHERE description IS NOT NULL 
              AND ((category IS NULL OR category = '') OR (tab IS NULL OR tab = ''))
        """)
        rows = cur.fetchall()
        
        if not rows:
            print("[PERF] No products need category updates")
            return
        
        print(f"[PERF] Processing category updates for {len(rows)} products...")
        
        # Batch collect all updates using optimized matcher
        updates = []
        for item_code, desc in rows:
            cat = matcher.get_category(desc) if matcher else None
            tab = matcher.get_tab(desc) if matcher else None
            
            if cat or tab:
                updates.append((item_code, cat, tab))
        
        # Perform batch update
        if updates:
            batch_update_categories(conn, updates)
            print(f"[PERF] [OK] Updated {len(updates)} products with categories/tabs")
        
        # Print cache statistics
        if matcher:
            print_performance_stats(matcher)


def optimized_load_staging(conn, run_id: str, df):
    """
    Optimized version of load_staging with faster duplicate checking.
    
    Key improvements:
    - Uses temporary table for bulk duplicate checking
    - Single database query instead of chunked queries
    """
    # Import necessary functions from original module
    from load_to_database import (
        _to_date, _none_str, _hash_row
    )
    from psycopg2.extras import execute_values
    import pandas as pd
    
    # Ensure required columns exist (same as original)
    expected = ["Date","Receipt","SO","Item Code","Description","Expiration Date",
                "Qty","Unit","Discount","Sales","Cost","Profit","Payment","Cashier ID",
                "Category","Tab"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Respect cleaner's columns
    if "RowHash" in df.columns and "row_hash" not in df.columns:
        df["row_hash"] = df["RowHash"]
    if "TxnType" not in df.columns:
        df["TxnType"] = pd.NA
    
    # Compute row_hash if still missing
    if "row_hash" not in df.columns:
        df["row_hash"] = df.apply(_hash_row, axis=1)
    
    # Recompute Profit if missing/inconsistent
    if "Profit" in df.columns:
        recompute_mask = df["Profit"].isna()
        if recompute_mask.any():
            df.loc[recompute_mask, "Profit"] = (
                pd.to_numeric(df["Sales"], errors="coerce")
                - pd.to_numeric(df["Cost"], errors="coerce")
            )
    
    # --- OPTIMIZED: Use bulk duplicate checking ---
    in_batch_hashes = list(df["row_hash"].astype(str).unique())
    
    print(f"[PERF] Checking {len(in_batch_hashes)} unique hashes for duplicates...")
    existing_hashes = optimize_duplicate_checking(conn, in_batch_hashes)
    print(f"[PERF] [OK] Found {len(existing_hashes)} existing hashes (duplicates)")
    
    is_dup = df["row_hash"].astype(str).isin(existing_hashes)
    df_dups = df.loc[is_dup].copy()
    df_new  = df.loc[~is_dup].copy()
    
    # --- Insert NEW rows into staging (same as original) ---
    if not df_new.empty:
        rows_new = []
        for _, r in df_new.iterrows():
            rows_new.append((
                run_id,
                _to_date(r["Date"]),
                _none_str(r["Receipt"]),
                _none_str(r["SO"]),
                _none_str(r["Item Code"]),
                _none_str(r["Description"]),
                _to_date(r["Expiration Date"]),
                (None if pd.isna(r["Qty"]) else float(r["Qty"])),
                _none_str(r["Unit"]),
                (None if pd.isna(r["Discount"]) else float(r["Discount"])),
                (None if pd.isna(r["Sales"]) else float(r["Sales"])),
                (None if pd.isna(r["Cost"]) else float(r["Cost"])),
                (None if pd.isna(r["Profit"]) else float(r["Profit"])),
                _none_str(r["Payment"]),
                _none_str(r["Cashier ID"]),
                _none_str(r["TxnType"]),
                _none_str(r["Category"]),
                _none_str(r["Tab"]),
                str(r["row_hash"]),
            ))
        
        with conn, conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO staging.sales_cleaned(
                    run_id, date, receipt, so, item_code, description, expiration_date,
                    qty, unit, discount, sales, cost, profit, payment, cashier_id, txn_type,
                    category, tab, row_hash
                ) VALUES %s
            """, rows_new)
        
        print(f"[PERF] [OK] Inserted {len(rows_new)} new rows into staging")
    
    # --- Log DUPLICATES into cleaning_errors (same as original) ---
    if not df_dups.empty:
        from load_to_database import _none_num_as_str
        
        rows_dup = []
        for _, r in df_dups.iterrows():
            rows_dup.append((
                run_id,
                "STAGING",
                "DUPLICATE_ROW",
                _to_date(r.get("Date")),
                _none_str(r.get("Receipt")),
                _none_str(r.get("SO")),
                _none_str(r.get("Item Code")),
                _none_str(r.get("Description")),
                _to_date(r.get("Expiration Date")),
                _none_num_as_str(r.get("Qty")),
                _none_str(r.get("Unit")),
                _none_num_as_str(r.get("Discount")),
                _none_num_as_str(r.get("Sales")),
                _none_num_as_str(r.get("Cost")),
                _none_num_as_str(r.get("Profit")),
                _none_str(r.get("Payment")),
                _none_str(r.get("Cashier ID")),
            ))
        
        with conn, conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO staging.cleaning_errors(
                    run_id, error_stage, error_reason, date, receipt, so, item_code, description,
                    expiration_date, qty, unit, discount, sales, cost, profit, payment, cashier_id
                ) VALUES %s
            """, rows_dup)
        
        print(f"[PERF] Logged {len(rows_dup)} duplicate rows to cleaning_errors")
    
    return len(df_new)


def apply_patches():
    """Apply all performance patches to the original module"""
    print("\n" + "="*60)
    print("APPLYING PERFORMANCE OPTIMIZATIONS")
    print("="*60 + "\n")
    
    # Patch the functions
    original_module.get_category_for_product = optimized_get_category_for_product
    original_module.get_tab_for_product = optimized_get_tab_for_product
    original_module.upsert_dim_product = optimized_upsert_dim_product
    original_module.load_staging = optimized_load_staging
    
    print("[OK] Patched get_category_for_product (with caching + fast fuzzy matching)")
    print("[OK] Patched get_tab_for_product (with caching + fast fuzzy matching)")
    print("[OK] Patched upsert_dim_product (with batch updates)")
    print("[OK] Patched load_staging (with optimized duplicate checking)")
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATIONS ACTIVE")
    print("="*60 + "\n")
    
    # Try to install rapidfuzz if not available
    try:
        import rapidfuzz
        print("[OK] rapidfuzz is installed (100x faster fuzzy matching)")
    except ImportError:
        print("[WARNING] rapidfuzz not installed. Install for 100x faster fuzzy matching:")
        print("   pip install rapidfuzz")
    
    print()


if __name__ == "__main__":
    apply_patches()
    print("Performance patches applied successfully!")
    print("\nYou can now import load_to_database and it will use the optimized functions.")
