import os
import uuid
import hashlib
from contextlib import contextmanager
from typing import Optional, Dict, Iterable, Set

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv

load_dotenv()

# ---------- small helpers ----------
def _summarize_errors(errors_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    if errors_df is None or errors_df.empty:
        return None
    cols = [c for c in ["error_stage", "error_reason"] if c in errors_df.columns]
    if not cols:
        return None
    grp = errors_df.groupby(cols, dropna=False).size().reset_index(name="count")
    out: Dict[str, Dict[str, int]] = {}
    for _, r in grp.iterrows():
        stage = str(r.get("error_stage") or "UNKNOWN")
        reason = str(r.get("error_reason") or "UNKNOWN")
        out.setdefault(stage, {})
        out[stage][reason] = int(r["count"])
    return out

def _none(v):
    try:
        return None if pd.isna(v) else v
    except Exception:
        return v

def _none_str(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return None if s == "" else s

def _none_num_as_str(v):
    try:
        return None if pd.isna(v) else str(v)
    except Exception:
        return str(v) if v is not None else None

def _hash_row(row: pd.Series) -> str:
    keys = [
        str(row.get("Date","")).strip(),
        str(row.get("Receipt","")).strip(),
        str(row.get("SO","")).strip(),
        str(row.get("Item Code","")).strip(),
        str(row.get("Qty","")).strip(),
        str(row.get("Sales","")).strip(),
        str(row.get("Cost","")).strip(),
    ]
    return hashlib.sha256("|".join(keys).encode("utf-8","ignore")).hexdigest()

def _to_date(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

def _normalize_exp_str(date_str):
    try:
        date_obj = pd.to_datetime(date_str)
        return date_obj.strftime('%Y-%m-%d')
    except Exception:
        return date_str

# ---------- DB ----------
def get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    return dsn

@contextmanager
def db():
    conn = psycopg2.connect(get_dsn())
    try:
        yield conn
    finally:
        conn.close()

# ---------- ETL run audit ----------
def begin_run(conn, file_name:str) -> str:
    run_id = str(uuid.uuid4())
    with conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ops.etl_runs(run_id, file_name, status, started_at)
            VALUES (%s, %s, 'RUNNING', now())
        """, (run_id, file_name))
    return run_id

def finalize_run(conn, run_id:str, status:str, rows_in:int, rows_out:int, rows_removed:int,
                 removed_breakdown:Optional[Dict]=None, notes:str=None):
    with conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE ops.etl_runs
               SET status=%s,
                   finished_at=now(),
                   rows_in=%s,
                   rows_out=%s,
                   rows_removed=%s,
                   removed_breakdown=%s,
                   notes=%s
             WHERE run_id=%s
        """, (
            status,
            rows_in,
            rows_out,
            rows_removed,
            Json(removed_breakdown) if removed_breakdown is not None else None,
            notes,
            run_id
        ))


# ---------- ensure objects ----------
def ensure_indexes(conn):
    with conn, conn.cursor() as cur:
        cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_stg_rowhash ON staging.sales_cleaned(row_hash);""")
        cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_fact_rowhash ON warehouse.fact_sales(row_hash);""")

def seed_dim_date(conn):
    with conn, conn.cursor() as cur:
        cur.execute("""
        INSERT INTO warehouse.dim_date (date_key, day, week, month, quarter, year, dow, is_weekend)
        SELECT d::date, EXTRACT(DAY FROM d)::INT, EXTRACT(WEEK FROM d)::INT,
               EXTRACT(MONTH FROM d)::INT, EXTRACT(QUARTER FROM d)::INT,
               EXTRACT(YEAR FROM d)::INT, EXTRACT(DOW FROM d)::INT,
               (EXTRACT(DOW FROM d) IN (0,6))
        FROM generate_series('2020-01-01'::date, '2045-12-31', interval '1 day') AS t(d)
        ON CONFLICT (date_key) DO NOTHING;
        """)

# ---------- utility ----------
def _fetch_existing_hashes(cur, table: str, hashes: Iterable[str]) -> Set[str]:
    """Return the subset of hashes that already exist in table.row_hash."""
    hashes = list({h for h in hashes if h})
    if not hashes:
        return set()
    cur.execute(f"SELECT row_hash FROM {table} WHERE row_hash = ANY(%s)", (hashes,))
    return {r[0] for r in cur.fetchall()}

# ---------- loaders ----------
def load_staging(conn, run_id: str, df: pd.DataFrame):
    """
    Insert cleaned rows into staging, but if a row_hash is already present
    (i.e., user re-uploads the same file or lines), we DO NOT delete the old
    row. Instead, we skip inserting it and log the duplicate to staging.cleaning_errors.
    """

    # Ensure required columns exist
    expected = ["Date","Receipt","SO","Item Code","Description","Expiration Date",
                "Qty","Unit","Discount","Sales","Cost","Profit","Payment","Cashier ID"]
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

    # --- 1) Find duplicates vs existing staging by row_hash
    in_batch_hashes = list(df["row_hash"].astype(str).unique())
    existing_hashes = set()

    if in_batch_hashes:
        # chunk to avoid very large IN lists
        CHUNK = 1000
        with conn, conn.cursor() as cur:
            for i in range(0, len(in_batch_hashes), CHUNK):
                slice_hashes = tuple(in_batch_hashes[i:i+CHUNK])
                cur.execute(
                    "SELECT row_hash FROM staging.sales_cleaned WHERE row_hash = ANY(%s)",
                    (list(slice_hashes),)
                )
                existing_hashes.update(h for (h,) in cur.fetchall())

    is_dup = df["row_hash"].astype(str).isin(existing_hashes)
    df_dups = df.loc[is_dup].copy()
    df_new  = df.loc[~is_dup].copy()

    # --- 2) Insert NEW rows into staging
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
                str(r["row_hash"]),
            ))

        with conn, conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO staging.sales_cleaned(
                    run_id, date, receipt, so, item_code, description, expiration_date,
                    qty, unit, discount, sales, cost, profit, payment, cashier_id, txn_type, row_hash
                ) VALUES %s
            """, rows_new)
    # --- 3) Log DUPLICATES into cleaning_errors
    if not df_dups.empty:
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
    # always return how many new rows went into staging
    return len(df_new)


def upsert_dim_product(conn, run_id:str):
    with conn, conn.cursor() as cur:
        cur.execute("""
        WITH by_item AS (
          SELECT
            item_code,
            MAX(NULLIF(description,'')) FILTER (WHERE description IS NOT NULL) AS any_desc,
            MAX(NULLIF(unit,''))        FILTER (WHERE unit        IS NOT NULL) AS any_unit,
            MIN(date) AS run_first_seen,
            MAX(date) AS run_last_seen
          FROM staging.sales_cleaned
          WHERE run_id = %s
            AND item_code IS NOT NULL
          GROUP BY item_code
        )
        INSERT INTO warehouse.dim_product(item_code, description, unit, first_seen_date, last_seen_date)
        SELECT item_code, any_desc, any_unit, run_first_seen, run_last_seen
        FROM by_item
        ON CONFLICT (item_code) DO UPDATE
        SET description     = COALESCE(EXCLUDED.description, warehouse.dim_product.description),
            unit            = COALESCE(EXCLUDED.unit,        warehouse.dim_product.unit),
            first_seen_date = LEAST(warehouse.dim_product.first_seen_date, EXCLUDED.first_seen_date),
            last_seen_date  = GREATEST(warehouse.dim_product.last_seen_date, EXCLUDED.last_seen_date);
        """, (run_id,))

def load_fact_sales(conn, run_id:str) -> int:
    with conn, conn.cursor() as cur:
        # Insert facts (idempotent via unique row_hash)
        cur.execute("""
            INSERT INTO warehouse.fact_sales(
  run_id, date_key, product_key, receipt_number, sales_order_number,
  quantity_sold, unit, unit_price, discount_rate,
  sales_amount, gross_amount, cost_amount, profit_amount,
  payment, cashier_id, txn_type, expiration_date, row_hash
)
SELECT
  s.run_id,
  s.date::date                        AS date_key,
  p.product_key,
  s.receipt                           AS receipt_number,
  s.so                                AS sales_order_number,
  s.qty                               AS quantity_sold,
  s.unit,
  CASE WHEN s.qty IS NOT NULL AND s.qty <> 0
       THEN s.sales / s.qty
       ELSE NULL END                  AS unit_price,
  CASE WHEN s.sales IS NOT NULL OR s.discount IS NOT NULL
       THEN COALESCE(s.discount,0) / NULLIF(s.sales + COALESCE(s.discount,0), 0)
       ELSE NULL END                  AS discount_rate,
  s.sales                             AS sales_amount,
  (s.sales + COALESCE(s.discount,0))  AS gross_amount,
  s.cost                              AS cost_amount,
  s.profit                            AS profit_amount,
  s.payment,
  s.cashier_id,
  COALESCE(s.txn_type,'SALE')         AS txn_type,
  s.expiration_date,
  s.row_hash
FROM staging.sales_cleaned s
JOIN warehouse.dim_product p ON p.item_code = s.item_code
JOIN warehouse.dim_date    d ON d.date_key = s.date::date
WHERE s.run_id = %s
ON CONFLICT (row_hash) DO NOTHING;

        """, (run_id,))

        # how many inserted for this run?
        cur.execute("SELECT COUNT(*) FROM warehouse.fact_sales WHERE run_id = %s", (run_id,))
        return cur.fetchone()[0]


def load_error_rows(conn, run_id:str, errors_df: pd.DataFrame):
    if errors_df is None or errors_df.empty:
        return
    df = errors_df.copy()
    df = df.astype(object).where(pd.notna(df), None)

    rename = {
        "Date":"date","Receipt":"receipt","SO":"so","Item Code":"item_code",
        "Description":"description","Expiration Date":"expiration_date",
        "Qty":"qty","Unit":"unit","Discount":"discount","Sales":"sales",
        "Cost":"cost","Profit":"profit","Payment":"payment","Cashier ID":"cashier_id",
        "error_stage":"error_stage","error_reason":"error_reason"
    }
    df.rename(columns={k:v for k,v in rename.items() if k in df.columns}, inplace=True)

    rows = []
    for _, r in df.iterrows():
        rows.append((
            run_id,
            _none_str(r.get("error_stage")),
            _none_str(r.get("error_reason")),
            _to_date(r.get("date")),
            _none_str(r.get("receipt")),
            _none_str(r.get("so")),
            _none_str(r.get("item_code")),
            _none_str(r.get("description")),
            _to_date(r.get("expiration_date")),
            _none_num_as_str(r.get("qty")),
            _none_str(r.get("unit")),
            _none_num_as_str(r.get("discount")),
            _none_num_as_str(r.get("sales")),
            _none_num_as_str(r.get("cost")),
            _none_num_as_str(r.get("profit")),
            _none_str(r.get("payment")),
            _none_str(r.get("cashier_id")),
        ))

    with conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO staging.cleaning_errors(
                run_id, error_stage, error_reason, date, receipt, so, item_code, description,
                expiration_date, qty, unit, discount, sales, cost, profit, payment, cashier_id
            ) VALUES %s
        """, rows)

# ---------- Orchestrator ----------
def run_full_load(file_name: str, raw_df: pd.DataFrame, ensure_schema_once: bool = False) -> str:
    # If raw_df is a list (multiple dataframes), concatenate them into one dataframe
    if isinstance(raw_df, list):  # Check if it's a list of dataframes (multiple files)
        print("[INFO] Combining multiple dataframes...")
        raw_df = pd.concat(raw_df, ignore_index=True)  # Concatenate all dataframes into one
        print(f"[INFO] Combined dataframe shape: {raw_df.shape}")

    # Proceed with the existing functionality
    with db() as conn:
        ensure_indexes(conn)  # Ensure database indexes are in place
        seed_dim_date(conn)  # Seed the date dimension if required

        run_id = begin_run(conn, file_name=file_name)  # Start the ETL run

        try:
            # Clean data types (apply the clean_data_types_improved function)
            cleaned_df, errors_df = clean_data_types_improved(raw_df)

            # Remove incomplete rows (apply remove_incomplete_rows function)
            cleaned_df, additional_errors_df = remove_incomplete_rows(cleaned_df)
            if not additional_errors_df.empty:
                errors_df = pd.concat([errors_df, additional_errors_df], ignore_index=True)

            rows_in = len(cleaned_df)

            # Load cleaned data to the staging area
            inserted_stg = load_staging(conn, run_id, cleaned_df)
            # Upsert product data
            upsert_dim_product(conn, run_id)
            # Load the fact sales data
            rows_out = load_fact_sales(conn, run_id)
            # Load any error rows
            load_error_rows(conn, run_id, errors_df)

            # Calculate removed rows: rows_in - inserted_stg (duplicates counted here) + errors_df rows
            dup_count = rows_in - inserted_stg
            err_count = 0 if errors_df is None else len(errors_df)
            rows_removed = max(0, dup_count) + err_count

            removed_breakdown = {
                "staging_duplicates": dup_count,
                "cleaner_errors": err_count,
                "by_reason": _summarize_errors(errors_df) or {}
            }

            # Finalize the ETL run
            finalize_run(
                conn, run_id, "SUCCEEDED",
                rows_in=rows_in,
                rows_out=rows_out,
                rows_removed=rows_removed,
                removed_breakdown=removed_breakdown,
                notes=None
            )

            return run_id  # Return the run ID for tracking

        except Exception as e:
            # If there's an error, finalize the run with failure status
            finalize_run(conn, run_id, "FAILED", rows_in=0, rows_out=0,
                         rows_removed=0, removed_breakdown=None, notes=str(e))
            raise


def clean_data_types_improved(df):
    print("\nCLEANING DATA TYPES:")
    print("-" * 30)
    df_clean = df.copy()

    # Numeric columns
    numeric_cols = ['Qty', 'Discount', 'Sales', 'Cost', 'Profit']
    for col in numeric_cols:
        if col in df_clean.columns:
            original_count = df_clean[col].notna().sum()
            df_clean[col] = (
                df_clean[col].astype(str)
                .str.replace(r'[^\d.-]', '', regex=True)
                .replace(['', 'nan', 'none', 'NaN', 'None'], pd.NA)
            )
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            cleaned_count = df_clean[col].notna().sum()
            print(f"  {col}: {original_count} -> {cleaned_count} valid values")

    # Dates  — treat 1970-01-01 as missing (common when raw has 0)
    date_cols = ['Date']
    for col in date_cols:
        if col in df_clean.columns:
            original_count = df_clean[col].notna().sum()
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            epoch_mask = df_clean[col] == pd.Timestamp('1970-01-01')
            if epoch_mask.any():
                df_clean.loc[epoch_mask, col] = pd.NaT
            cleaned_count = df_clean[col].notna().sum()
            print(f"  {col}: {original_count} -> {cleaned_count} valid dates")

    # Expiration Date: keep blanks as NaN; valid dates -> YYYY-MM-DD strings
    if 'Expiration Date' in df_clean.columns:
        df_clean['Expiration Date'] = df_clean['Expiration Date'].apply(
            lambda x: pd.NA if (x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip()=='')) else _normalize_exp_str(str(x))
        )
        # Re-assign empty string from _normalize_exp_str to pd.NA
        df_clean['Expiration Date'] = df_clean['Expiration Date'].apply(lambda x: pd.NA if (isinstance(x, str) and x.strip()=='') else x)

    # Text columns (includes Unit)
    string_cols = ['Receipt', 'SO', 'Item Code', 'Description', 'Payment', 'Cashier ID', 'Unit']
    for col in string_cols:
        if col in df_clean.columns:
            original_count = df_clean[col].notna().sum()
            df_clean[col] = (
                df_clean[col].astype(str).str.strip()
                .replace(['nan', 'none', 'null', '', 'NaN', 'None', 'NULL'], pd.NA)
            )
            if col == 'Unit':
                df_clean[col] = df_clean[col].str.lower()
            cleaned_count = df_clean[col].notna().sum()
            print(f"  {col}: {original_count} -> {cleaned_count} valid values")

            if col == 'Item Code':
                df_clean[col] = df_clean[col].str.strip().str.upper()

    # Recompute Profit once (after the loop)
    if {'Sales', 'Cost'}.issubset(df_clean.columns):
        calc_profit = pd.to_numeric(df_clean['Sales'], errors='coerce') - pd.to_numeric(df_clean['Cost'], errors='coerce')
        if 'Profit' not in df_clean.columns or df_clean['Profit'].isna().any():
            df_clean['Profit'] = df_clean['Profit'].fillna(calc_profit)

    return df_clean, pd.DataFrame(columns=list(df_clean.columns) + ['error_reason', 'error_stage'])

def remove_incomplete_rows(df):
    print("\nREMOVING INCOMPLETE ROWS:")
    print("-" * 30)

    df_clean = df.copy()
    errors = []

    initial_count = len(df_clean)

    supporting_columns = ['Item Code', 'Description']
    financial_columns = ['Sales', 'Cost', 'Profit']

    rows_to_remove = []
    for idx in df_clean.index:
        has_financial_data = any(
            (col in df_clean.columns) and pd.notna(df_clean.loc[idx, col]) and df_clean.loc[idx, col] != 0
            for col in financial_columns
        )
        if has_financial_data:
            missing_supporting = []
            for col in supporting_columns:
                if col in df_clean.columns:
                    value = df_clean.loc[idx, col]
                    if pd.isna(value) or str(value).strip().lower() in ('', 'nan', 'none', 'null'):
                        missing_supporting.append(col)
            if missing_supporting:
                rows_to_remove.append(idx)

    if rows_to_remove:
        removed = df_clean.loc[rows_to_remove].copy()
        removed['error_reason'] = 'financial data present but missing supporting fields'
        removed['error_stage']  = 'remove_incomplete_rows'
        errors.append(removed)
        df_clean = df_clean.drop(rows_to_remove)
        print(f"[OK] Removed {len(rows_to_remove)} incomplete rows")
        print(f"  Remaining rows: {len(df_clean)}")
    else:
        print("[OK] No incomplete rows found - all data appears complete")

    important_columns = ['Item Code', 'Description', 'Qty', 'Sales', 'Cost']
    available_important = [c for c in important_columns if c in df_clean.columns]
    if available_important:
        empty_mask = df_clean[available_important].isna().all(axis=1)
        if empty_mask.any():
            removed2 = df_clean.loc[empty_mask].copy()
            removed2['error_reason'] = 'all important fields empty'
            removed2['error_stage']  = 'remove_incomplete_rows'
            errors.append(removed2)
            df_clean = df_clean.loc[~empty_mask]
            print(f"[OK] Removed {int(empty_mask.sum())} completely empty data rows")

    # Placeholder/header rows & non-item adjustments clean (from CLEAN.py)
    num_cols = [c for c in ['Qty', 'Sales', 'Cost', 'Profit'] if c in df_clean.columns]
    if {'Item Code', 'Description'}.issubset(df_clean.columns) and len(num_cols) > 0:
        ic   = df_clean['Item Code'].astype('string').str.strip()
        desc = df_clean['Description'].astype('string').str.strip()

        ic_digits_only = ic.str.fullmatch(r'\d+')
        ic_non_numeric = ic.notna() & ~ic_digits_only
        account_like   = ic.str.contains(r'^(account|customer|member)\s*:?', case=False, na=False)
        desc_blank     = desc.isna() | (desc == '')
        numeric_zeros  = (
            df_clean[num_cols]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .eq(0)
            .all(axis=1)
        )
        placeholder_mask = numeric_zeros & (ic_non_numeric | desc_blank | account_like)

        if placeholder_mask.any():
            removed3 = df_clean.loc[placeholder_mask].copy()
            removed3['error_reason'] = 'placeholder/non-item row (e.g., header like "Account : ...")'
            removed3['error_stage']  = 'remove_incomplete_rows'
            errors.append(removed3)
            df_clean = df_clean.loc[~placeholder_mask]
            print(f"[OK] Removed {int(placeholder_mask.sum())} placeholder/non-item rows")

    fin_cols = [c for c in ['Sales', 'Cost', 'Profit'] if c in df_clean.columns]
    if fin_cols:
        fin_zero = (
            df_clean[fin_cols]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .eq(0)
            .all(axis=1)
        )
        desc_l = (
            df_clean['Description'].astype('string').str.lower().str.strip()
            if 'Description' in df_clean.columns else
            pd.Series('', index=df_clean.index, dtype='string')
        )
        ic_l = (
            df_clean['Item Code'].astype('string').str.lower().str.strip()
            if 'Item Code' in df_clean.columns else
            pd.Series('', index=df_clean.index, dtype='string')
        )

        kw_desc  = r'(sales\s*discount|^discount$|less\s*discount|void|cancel|round(?:ing|[-\s]*off)|change|senior|pwd|rebate|price\s*adj|adjustment)'
        kw_codes = r'^(sd|disc|discount|void|sc|pwd|rebate)$'

        is_adjustment = desc_l.str.contains(kw_desc, na=False) | ic_l.str.match(kw_codes, na=False)
        adj_mask = fin_zero & is_adjustment

        if adj_mask.any():
            removed_adj = df_clean.loc[adj_mask].copy()
            removed_adj['error_reason'] = 'non-item adjustment (e.g., Sales Discount/VOID/rounding)'
            removed_adj['error_stage']  = 'remove_incomplete_rows'
            errors.append(removed_adj)
            df_clean = df_clean.loc[~adj_mask]
            print(f"[OK] Removed {int(adj_mask.sum())} non-item adjustment rows")

    final_count = len(df_clean)
    print(f"\nData validation summary:")
    print(f"  Initial rows: {initial_count}")
    print(f"  Removed rows: {initial_count - final_count}")
    print(f"  Final rows: {final_count}")

    errors_df = (
        pd.concat(errors, ignore_index=True)
        if errors
        else pd.DataFrame(columns=list(df_clean.columns) + ['error_reason', 'error_stage'])
    )
    return df_clean, errors_df
