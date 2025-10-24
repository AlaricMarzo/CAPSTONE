import pandas as pd
import requests
from io import StringIO
import re
import os
import sys
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import json  # Added json import for JSON output

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply performance optimizations first
try:
    from apply_performance_fixes import apply_patches
    apply_patches()
    print("[PERF] ✅ Performance optimizations applied", file=sys.stderr)
except ImportError as e:
    print(f"[PERF] ⚠️  Could not apply performance fixes: {e}", file=sys.stderr)
    print("[PERF] Continuing with standard (slower) functions", file=sys.stderr)

try:
    from load_to_database import run_full_load, load_product_categories, get_category_for_product, get_tab_for_product
except ImportError as e:
    print(f"ERROR: Could not import required functions: {e}", file=sys.stderr)  # Redirect to stderr
    print(f"Python path: {sys.path}", file=sys.stderr)
    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}", file=sys.stderr)
    raise


# put this near the top of CLEAN_merged.py
REQUIRED_HEADER = [
    "Date","Receipt","SO","Item Code","Description",
    "Qty","Discount","Sales","Cost","Profit","Payment","Cashier ID"
]

def _norm_col(c: str) -> str:
    # normalize: strip, collapse spaces, remove BOMs, lowercase compare
    return (
        str(c)
        .replace("\ufeff", "")
        .strip()
        .replace("\xa0", " ")
        .replace("  ", " ")
    )

def analyze_csv_structure_strict(csv_content: str) -> int:
    """
    Find the line whose comma-separated values match REQUIRED_HEADER exactly
    (order and names must match after normalization). If none is found, raise.
    """
    lines = csv_content.splitlines()
    target = [_norm_col(x) for x in REQUIRED_HEADER]

    for i, line in enumerate(lines):
        if line.count(",") < (len(REQUIRED_HEADER)-1):
            continue
        cols = [_norm_col(x) for x in line.split(",")]
        if cols == target:
            # exact match
            return i

    raise ValueError(
        "Header not found. The file must contain this exact header line:\n"
        + ",".join(REQUIRED_HEADER)
    )



# ==============================================
# Helpers: Units
# ==============================================
UNIT_ALIASES = {
    'pcs': {'pc','pcs','piece','pieces','unit','units'},
    'box': {'box','boxes','bx'},
    'pack': {'pack','packs','pkt','packet','packets'},
    'kg': {'kg','kilogram','kilograms'},
    'g': {'g','gram','grams'},
    'l': {'l','liter','liters'},
    'ml': {'ml','milliliter','milliliters'},
}

def canonical_unit(text):
    if not isinstance(text, str):
        return pd.NA
    t = text.strip().lower()
    for canon, aliases in UNIT_ALIASES.items():
        if t == canon or t in aliases:
            return canon
    # loose regex catch
    if re.search(r'\bpcs?\b', t): return 'pcs'
    if re.search(r'\bpieces?\b', t): return 'pcs'
    if re.search(r'\bboxes?\b', t): return 'box'
    if re.search(r'\bpacks?\b|\bpkt\b|\bpackets?\b', t): return 'pack'
    if re.search(r'\bkg\b|\bkilograms?\b', t): return 'kg'
    if re.search(r'\bgrams?\b|\bg\b', t): return 'g'
    if re.search(r'\bliters?\b|\bl\b', t): return 'l'
    if re.search(r'\bmilliliters?\b|\bml\b', t): return 'ml'
    return pd.NA

def normalize_units_on_df(df):
    """
    Fill/standardize Unit. If there's no usable Unit column (missing or all NA),
    infer from Description. Also default to 'pcs' when nothing obvious is found.
    (Combines logic from both versions.)
    """
    df = df.copy()

    # --- 1) find a usable unit source column ---
    unit_source_col = None
    for cand in ['Unit','Units','UOM','uom','unit','units']:
        if cand in df.columns:
            unit_source_col = cand
            break

    def col_is_empty(series):
        if series is None:
            return True
        s = series.astype(str).str.strip().replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'NULL': pd.NA})
        return s.isna().all()

    need_infer = (unit_source_col is None) or col_is_empty(df.get(unit_source_col))

    # --- 2) infer from Description if needed ---
    if need_infer and 'Description' in df.columns:
        unit_source_col = 'Unit'  # write inference into 'Unit'
        pattern = r'''(?ix)
            (?<![A-Za-z])(
                pc|pcs|piece|pieces|unit|units|
                box|boxes|pack|packs|pkt|packet|packets|
                sachet|sachets|vial|vials|amp|ampule|ampoule|
                tube|tubes|bottle|bottles|bot|btl|
                jar|jars|can|cans|tin|tins|
                strip|blister|blisters|
                cap|caps|capsule|capsules|
                tab|tabs|tablet|tablets|
                roll|rolls|pair|pairs|set|sets|
                kg|g|mg|ml|iu
            )(?![A-Za-z])
        '''
        df[unit_source_col] = df['Description'].astype(str).str.extract(pattern)[0]

    # --- 3) if still nothing, bail gracefully ---
    if unit_source_col is None:
        return df, "No unit column found; skipped unit normalization."

    # --- 4) canonicalize tokens to a small set ---
    def canon(u):
        u = canonical_unit(u)  # will map pcs/box/pack/kg/g/l/ml
        if pd.isna(u):
            return pd.NA

        # extra mappings for pharma forms -> treat as piece-like
        tablety = {'tab','tabs','tablet','tablets'}
        capsuly  = {'cap','caps','capsule','capsules'}
        vials    = {'vial','vials','amp','ampule','ampoule'}
        sachets  = {'sachet','sachets'}
        bottles  = {'bottle','bottles','bot','btl'}
        tubes    = {'tube','tubes'}
        containers = {'jar','jars','can','cans','tin','tins'}
        misc_piece = {'strip','blister','blisters','roll','rolls','pair','pairs','set','sets'}
        dosage = {'mg','mcg','iu'}  # dosage units; treat as piece-like by default

        t = str(u).lower().strip()
        if t in tablety or t in capsuly or t in vials or t in sachets or t in bottles or t in tubes or t in containers or t in misc_piece or t in dosage:
            return 'pcs'
        return t

    df['Unit_std'] = df[unit_source_col].apply(canon)

    # --- 5) optional scaling for g->kg and ml->l only (leave mg/mcg alone) ---
    if 'Qty' in df.columns:
        q = pd.to_numeric(df['Qty'], errors='coerce')
        mask_g  = df['Unit_std'].eq('g')
        mask_ml = df['Unit_std'].eq('ml')
        if mask_g.any():
            df.loc[mask_g, 'Qty'] = q.where(~mask_g, q/1000.0)
            df.loc[mask_g, 'Unit_std'] = 'kg'
        if mask_ml.any():
            df.loc[mask_ml, 'Qty'] = q.where(~mask_ml, q/1000.0)
            df.loc[mask_ml, 'Unit_std'] = 'l'

    # --- 6) default any remaining NA to 'pcs' (typical POS) ---
    if 'Qty' in df.columns:
        mask_na = df['Unit_std'].isna() & pd.to_numeric(df['Qty'], errors='coerce').notna()
        df.loc[mask_na, 'Unit_std'] = 'pcs'
    else:
        df.loc[df['Unit_std'].isna(), 'Unit_std'] = 'pcs'

    df['Unit'] = df['Unit_std']
    df.drop(columns=['Unit_std'], inplace=True)
    return df, "Units normalized (including inference from Description; defaulted missing to 'pcs')."

# ==============================================
# Input loading
# ==============================================
def load_data_from_source(source):
    if source.startswith(('http://', 'https://')):
        print(f"Fetching data from URL: {source}", file=sys.stderr)
        response = requests.get(source)
        response.raise_for_status()
        print("Data fetched successfully from URL!", file=sys.stderr)
        return response.text
    else:
        print(f"Reading data from file: {source}", file=sys.stderr)
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        with open(source, 'r', encoding='utf-8') as file:
            content = file.read()
        print("Data loaded successfully from file!", file=sys.stderr)
        return content

def analyze_csv_structure(csv_content):
    lines = csv_content.strip().split('\n')
    print(f"Total lines in file: {len(lines)}", file=sys.stderr)
    print("\nFirst 15 lines of the file:", file=sys.stderr)
    for i, line in enumerate(lines[:15]):
        print(f"Line {i}: {repr(line)}", file=sys.stderr)
    data_start_row = 0
    header_found = False
    for i, line in enumerate(lines):
        if line.count(",") >= 3:
            if any(keyword in line.lower() for keyword in ['date', 'receipt', 'item', 'description', 'qty', 'sales', 'total', 'so', 'code']):
                data_start_row = i
                header_found = True
                print(f"Found header at line {i}: {line}", file=sys.stderr)
                break
            elif re.search(r'\d+[,.]?\d*', line) and ',' in line:
                data_start_row = i
                print(f"Found potential data start at line {i}: {line}", file=sys.stderr)
                break
    if not header_found:
        print("No clear header found, looking for any data rows...", file=sys.stderr)
        for i, line in enumerate(lines):
            if ',' in line and len(line.strip()) > 10:
                data_start_row = i
                print(f"Using line {i} as data start: {line}", file=sys.stderr)
                break
    return data_start_row

# ==============================================
# Expiration parsing (merged strict + cleanup)
# ==============================================
def _normalize_exp_str(date_str):
    """Return YYYY-MM-DD if valid and in sane range (2020-2045), else ''."""
    if not isinstance(date_str, str):
        date_str = str(date_str) if pd.notna(date_str) else ''
    date_str = date_str.strip()
    if not date_str:
        return ''
    fmts = ['%Y-%m-%d','%m/%d/%Y','%m-%d-%Y','%m/%d/%y','%m-%d-%y']
    dt = None
    for f in fmts:
        try:
            dt = datetime.strptime(date_str, f)
            break
        except Exception:
            continue
    if dt is None:
        try:
            dt = pd.to_datetime(date_str, errors='coerce')
        except Exception:
            dt = None
        if pd.isna(dt):
            dt = None
    if dt is None:
        return ''
    if not (2020 <= int(dt.year) <= 2045):
        return ''
    return pd.Timestamp(dt).strftime('%Y-%m-%d')

def extract_expiration_from_description(description):
    """
    Extract expiration strictly from description; return (cleaned_description, yyyy-mm-dd or pd.NA).
    Never fabricate. Cleans out EXP/BB fragments from the text.
    """
    if pd.isna(description) or not isinstance(description, str) or not description.strip():
        return description, pd.NA

    text = description

    patterns = [
        r'[#\$\s]?EXP\s*(\d{4}-\d{2}-\d{2})\$?',
        r'[#\$\s]?exp\s*(\d{4}-\d{2}-\d{2})\$?',
        r'(?:exp(?:iry|iration)?|best\s*by|use\s*by|expires?|bb)[:\s\-]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:exp|bb)\b',
        r'\b(\d{4}-\d{2}-\d{2})\b'
    ]

    cleaned = text
    m = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        norm = _normalize_exp_str(m.group(1))
        if norm:
            cleaned = re.sub(pat, '', text, flags=re.IGNORECASE)
            break

    # Remove general EXP/BB fragments even if no valid date
    cleanup_patterns = [
        r'#exp\w*', r'exp\d+', r'bb\d+', r'#bb\w*', r'exp[:\s]*$', r'bb[:\s]*$', r'#\s*$', r'\s+#\s*'
    ]
    for cp in cleanup_patterns:
        cleaned = re.sub(cp, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' ,-#:()')
    norm_final = _normalize_exp_str(m.group(1)) if m else ''
    return (cleaned if cleaned else description, pd.NA if not norm_final else norm_final)

# ==============================================
# Core cleaning pipeline
# ==============================================
def project_to_target(df_in, column_mapping, target_columns):
    out = pd.DataFrame()
    for t in target_columns:
        src = column_mapping.get(t)
        if src is not None and src in df_in.columns:
            out[t] = df_in[src]
        else:
            out[t] = pd.NA
    if '_row_id' in df_in.columns:
        out['_row_id'] = df_in['_row_id']
    return out

def remove_incomplete_rows(df):
    print("\nREMOVING INCOMPLETE ROWS:", file=sys.stderr)
    print("-" * 30, file=sys.stderr)

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
        print(f"[OK] Removed {len(rows_to_remove)} incomplete rows", file=sys.stderr)
        print(f"  Remaining rows: {len(df_clean)}", file=sys.stderr)
    else:
        print("[OK] No incomplete rows found - all data appears complete", file=sys.stderr)

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
            print(f"[OK] Removed {int(empty_mask.sum())} completely empty data rows", file=sys.stderr)

    # Placeholder/header rows & non-item adjustments clean (from CLEAN.py)
    num_cols = [c for c in ['Qty', 'Sales', 'Cost', 'Profit'] if c in df_clean.columns]
    if {'Item Code', 'Description'}.issubset(df_clean.columns) and len(num_cols) > 0:
        ic   = df_clean['Item Code'].astype('string').str.strip()
        desc = df_clean['Description'].astype('string').str.strip()

        ic_digits_only = ic.str.fullmatch(r'\d+')
        ic_non_numeric = ic.notna() & ~ic_digits_only
        account_like   = ic.str.contains(r'^(?:account|customer|member)\s*:?', case=False, na=False)
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
            print(f"[OK] Removed {int(placeholder_mask.sum())} placeholder/non-item rows", file=sys.stderr)

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
            print(f"[OK] Removed {int(adj_mask.sum())} non-item adjustment rows", file=sys.stderr)

    final_count = len(df_clean)
    print(f"\nData validation summary:", file=sys.stderr)
    print(f"  Initial rows: {initial_count}", file=sys.stderr)
    print(f"  Removed rows: {initial_count - final_count}", file=sys.stderr)
    print(f"  Final rows: {final_count}", file=sys.stderr)

    errors_df = (
        pd.concat(errors, ignore_index=True)
        if errors
        else pd.DataFrame(columns=list(df_clean.columns) + ['error_reason', 'error_stage'])
    )
    return df_clean, errors_df

def map_columns_improved(df, target_columns):
    print("\nMAPPING COLUMNS:", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    column_mapping = {}
    available_cols = df.columns.tolist()
    print("Available columns:", available_cols, "\n", file=sys.stderr)

    mapping_patterns = {
        'Date': ['date','time','datetime','day','created','timestamp','when','dt',
                 'trans date','transaction date','trx date','posting date'],
        'Receipt': ['receipt','rcpt','ticket','trans','transaction','ref','reference','no',
                    'invoice','invoice no','invoice number','doc no','document no'],
        'SO': ['so','sales order','order','order no','order number','sales_order','ord'],
        'Item Code': ['item','code','sku','product code','prod code','barcode','item_code',
                      'product_code','plu','item no','item number','product id','productid'],
        'Description': ['description','desc','product','item name','name','title',
                        'product_name','item_name','product description','item description'],
        'Expiration Date': ['exp','expiry','expiration','best by','use by','expires','bb',
                            'expiry date','exp date'],
        'Qty': ['qty','quantity','amount','count','qnty','qnt','qty sold','quantity sold',
                'order qty','ordered qty'],
        'Unit': ['unit','units','uom','pack','packet','packets','box','boxes','pcs','pc','piece','pieces',
                 'kg','g','l','ml','uom code','unit of measure'],
        'Discount': ['discount','disc','off','reduction','rebate','promo','less discount','line discount'],
        'Sales': ['sales','total','amount','price','value','revenue','sale','net','gross',
                  'line amount','line total','extended','ext amount','net amount',
                  'net sales','line price','unit price','amount (net)','net line amount'],
        'Cost': ['cost','cogs','unit cost','purchase','buy','wholesale','line cost','total cost'],
        'Profit': ['profit','margin','gain','net profit','gp','gross profit'],
        'Payment': ['payment','pay','method','type','pay_method','tender','tender type','payment type'],
        'Cashier ID': ['cashier','user','employee','staff','operator','clerk','cashier_id','user_id','emp',
                       'cashier name','encoded by'],
    }

    for target_col, patterns in mapping_patterns.items():
        best_match = None
        best_score = 0

        for col in available_cols:
            if col in column_mapping.values():
                continue
            col_lower = str(col).lower().strip()
            score = 0
            if col_lower in [p.lower() for p in patterns]:
                score = 100
            else:
                for pattern in patterns:
                    p = pattern.lower()
                    if p in col_lower:
                        score = max(score, 80)
                    elif col_lower in p:
                        score = max(score, 70)
                    elif any(word in col_lower for word in p.split()):
                        score = max(score, 50)
                    elif any(word in p for word in col_lower.split('_')):
                        score = max(score, 30)
            if score > best_score:
                best_score = score
                best_match = col

        # Prefer literal column names that usually hold line amounts for Sales
        if target_col == 'Sales':
            preferred_aliases = {
                'sales', 'sale', 'sales amount', 'sales_amt',
                'line amount', 'line total', 'extended', 'ext amount',
                'net amount', 'net sales', 'line price', 'unit price'
            }
            exact = [c for c in available_cols if str(c).strip().lower() in preferred_aliases]
            if exact:
                best_match = exact[0]
                best_score = 101

        if best_match and best_score >= 20:
            column_mapping[target_col] = best_match
            print(f"  [OK] {target_col} <- '{best_match}' (confidence: {best_score}%)", file=sys.stderr)
        else:
            print(f"  [WARNING] {target_col} <- NO MATCH FOUND (best was '{best_match}' with {best_score}%)", file=sys.stderr)

    print("Final column mapping:", column_mapping, file=sys.stderr)
    return column_mapping


def _parse_money_series(s: pd.Series) -> pd.Series:
    """
    Parse money-like strings to numbers, handling:
      (1,234.56) -> -1234.56
      1,234.56-  -> -1234.56
      'CR' (credit) -> negative; 'DR' -> positive
      EU decimal comma: 1.234,56 -> 1234.56
    """
    if s is None:
        return pd.Series(dtype='float64')

    x = s.astype(str).str.strip()
    # accounting negatives
    x = x.str.replace(r'^$$([^)]+)$$$', r'-\1', regex=True) # Corrected regex
    x = x.str.replace(r'^\s*([0-9.,]+)\s*-\s*$', r'-\1', regex=True)
    # CR/DR markers
    has_cr = x.str.contains(r'\bCR\b', case=False, regex=True)
    x = x.str.replace(r'\b[CD]R\b', '', regex=True, case=False).str.strip()
    x = x.mask(has_cr, '-' + x)
    # keep digits, comma, dot, minus
    x = x.str.replace(r'[^\d,.-]', '', regex=True) # Allow '.' and ',' and '-' and digits

    # if one comma and no dot -> treat comma as decimal; else remove commas
    def _commas_to_decimal(t):
        if pd.isna(t) or not isinstance(t, str): return t
        t = t.strip()
        # Handle cases like '1.234,56' (European) vs '1,234.56' (US)
        if t.count(',') == 1 and t.count('.') > 0: # Has both a comma and a dot
            # If the comma appears after the last dot, it's likely a decimal separator
            if t.rfind(',') > t.rfind('.'):
                return t.replace('.', '').replace(',', '.')
            # Otherwise, assume dot is decimal, comma is thousands
            else:
                return t.replace(',', '')
        elif t.count(',') == 1 and t.count('.') == 0: # Only a comma, e.g., '123,45'
            return t.replace(',', '.')
        elif t.count('.') == 1 and t.count(',') == 0: # Only a dot, e.g., '123.45'
            return t
        elif t.count(',') > 1 or t.count('.') > 1: # Multiple thousands separators without clear decimal
            # Attempt to infer by looking at the last separator
            if t.rfind(',') > t.rfind('.'): # Comma is last, likely decimal
                return t.replace('.', '').replace(',', '.')
            else: # Dot is last, likely decimal
                return t.replace(',', '')
        return t # No commas or dots, or only one type used as expected

    x = x.apply(_commas_to_decimal)

    return pd.to_numeric(x, errors='coerce')


def _recover_sales_if_empty(df_final, df_clean, column_mapping):
    """
    Populate df_final['Sales'] when mapping missed it or mapped a useless column.
    Treat 'empty' as: all NaN OR all zeros. Prefer true line-amount columns.
    Falls back to 'Payment' when it clearly looks line-level (not a repeated receipt total).
    """
    if 'Sales' not in df_final.columns:
        df_final['Sales'] = pd.NA

    def _nonzero_cnt(s):
        x = pd.to_numeric(s, errors='coerce')
        return (x.fillna(0).abs() > 1e-9).sum()

    # If Sales already has real numbers, keep it
    if _nonzero_cnt(df_final['Sales']) > 0:
        return df_final

    # 1) Literal column names that usually hold line amounts
    literal_candidates = {
        'sales', 'sale', 'sales amount', 'sales_amt',
        'line amount', 'line total', 'extended', 'ext amount',
        'net amount', 'net sales', 'line price', 'unit price'
    }
    for col in df_clean.columns:
        if str(col).strip().lower() in literal_candidates:
            parsed = _parse_money_series(df_clean[col])
            if (parsed.fillna(0).abs() > 1e-9).any():
                df_final['Sales'] = parsed
                print(f"🔎 Recovered Sales from raw column '{col}' ({parsed.notna().sum()} values).", file=sys.stderr)
                return df_final

    # 2) Regex fallback (but avoid a bare 'total')
    candidates = re.compile(
        r'(sales?|sale\s*amount|sales?\s*(?:amount|value|price)|'
        r'line\s*(?:amount|total)|extended|ext|net\s*(?:amount|total|sales?)|'
        r'line\s*price|unit\s*price)',
        re.I
    )
    reserved = set(v for v in column_mapping.values() if v is not None)
    best_col, best_series, best_non_null = None, None, -1
    for col in df_clean.columns:
        if col in reserved:
            continue
        if not candidates.search(str(col)):
            continue
        parsed = _parse_money_series(df_clean[col])
        nn = parsed.notna().sum()
        if nn > best_non_null:
            best_col, best_series, best_non_null = col, parsed, nn
    if best_series is not None and (best_series.fillna(0).abs() > 1e-9).any():
        df_final['Sales'] = best_series
        print(f"🔎 Recovered Sales from raw column '{best_col}' ({best_non_null} values).", file=sys.stderr)
        return df_final

    # 3) Heuristic: use Payment if it behaves like a line amount (varies within a receipt)
    if 'Payment' in df_final.columns:
        pay = _parse_money_series(df_final['Payment'])
        if (pay.fillna(0).abs() > 1e-9).any():
            if 'Receipt' in df_final.columns:
                uniq_per_receipt = df_final.groupby('Receipt')['Payment'].nunique(dropna=True)
                # If ≥30% of receipts have >1 unique Payment value, treat it as line-level
                if (uniq_per_receipt > 1).mean() >= 0.30:
                    df_final['Sales'] = pay
                    print("🔎 Recovered Sales from 'Payment' (appears to be line-level).", file=sys.stderr)
                    return df_final
                else:
                    print("⚠ Skipped Payment fallback: looks like a repeated receipt total.", file=sys.stderr)
            else:
                df_final['Sales'] = pay
                print("🔎 Recovered Sales from 'Payment' (no receipt grouping to check).", file=sys.stderr)
                return df_final

    print("⚠ Could not recover Sales from raw columns.", file=sys.stderr)
    return df_final


def clean_data_types_improved(df):
    print("\nCLEANING DATA TYPES:", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    df_clean = df.copy()

    # Numeric columns
    numeric_cols = ['Qty', 'Discount', 'Sales', 'Cost', 'Profit']
    for col in numeric_cols:
        if col in df_clean.columns:
            original_count = df_clean[col].notna().sum()
            # Handle potential money symbols etc. more robustly
            # Keep digits, dot, comma, minus, and potentially leading/trailing symbols like '$' or '()'
            df_clean[col] = (
                df_clean[col].astype(str)
                .str.replace(r'^[$$$()]+|[$$$()]+$', '', regex=True) # Remove leading/trailing $ or ()
                .str.replace(r'[^\d,.-]', '', regex=True) # Keep digits, comma, dot, minus
                .replace(['', 'nan', 'none', 'NaN', 'None', 'NULL'], pd.NA)
            )
            # Apply European comma as decimal separator logic if needed
            def _handle_commas(t):
                if pd.isna(t) or not isinstance(t, str): return t
                t = t.strip()
                # Handle cases like '1.234,56' (European) vs '1,234.56' (US)
                if t.count(',') == 1 and t.count('.') > 0: # Has both a comma and a dot
                    # If the comma appears after the last dot, it's likely a decimal separator
                    if t.rfind(',') > t.rfind('.'):
                        return t.replace('.', '').replace(',', '.')
                    # Otherwise, assume dot is decimal, comma is thousands
                    else:
                        return t.replace(',', '')
                elif t.count(',') == 1 and t.count('.') == 0: # Only a comma, e.g., '123,45'
                    return t.replace(',', '.')
                elif t.count('.') == 1 and t.count(',') == 0: # Only a dot, e.g., '123.45'
                    return t
                elif t.count(',') > 1 or t.count('.') > 1: # Multiple thousands separators without clear decimal
                    # Attempt to infer by looking at the last separator
                    if t.rfind(',') > t.rfind('.'): # Comma is last, likely decimal
                        return t.replace('.', '').replace(',', '.')
                    else: # Dot is last, likely decimal
                        return t.replace(',', '')
                return t # No commas or dots, or only one type used as expected

            df_clean[col] = df_clean[col].apply(_handle_commas)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            cleaned_count = df_clean[col].notna().sum()
            print(f"  {col}: {original_count} -> {cleaned_count} valid values", file=sys.stderr)

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
            print(f"  {col}: {original_count} -> {cleaned_count} valid dates", file=sys.stderr)

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
            print(f"  {col}: {original_count} -> {cleaned_count} valid values", file=sys.stderr)

            if col == 'Item Code':
                df_clean[col] = df_clean[col].str.strip().str.upper()

    # Recompute Profit once (after the loop)
    if {'Sales', 'Cost'}.issubset(df_clean.columns):
        # Ensure we are using the cleaned numeric versions
        sales = pd.to_numeric(df_clean['Sales'], errors='coerce')
        cost = pd.to_numeric(df_clean['Cost'], errors='coerce')
        calc_profit = sales - cost

        # Only update Profit if it's missing or if calc_profit provides new valid values
        if 'Profit' not in df_clean.columns or df_clean['Profit'].isna().all():
            df_clean['Profit'] = calc_profit
            print(f"  Profit: Calculated and populated ({calc_profit.notna().sum()} values)", file=sys.stderr)
        else:
            # Update existing Profit column where calculation is valid and original was NA
            mask_update = df_clean['Profit'].isna() & calc_profit.notna()
            if mask_update.any():
                df_clean.loc[mask_update, 'Profit'] = calc_profit[mask_update]
                print(f"  Profit: Updated {mask_update.sum()} missing values", file=sys.stderr)

    return df_clean



def classify_txn_type(row):
    # default
    t = "SALE"
    desc = str(row.get('Description', '') or '').lower()
    code = str(row.get('Item Code', '') or '').lower()
    qty  = pd.to_numeric(row.get('Qty', pd.NA), errors='coerce')

    # adjustments keywords (if they survived filters)
    if re.search(r'(discount|void|round(?:ing|[-\s]*off)|price\s*adj|adjustment|rebate|change|senior|pwd)', desc):
        return "ADJUSTMENT"

    # returns: negative qty or obvious words
    if (pd.notna(qty) and qty < 0) or re.search(r'(return|refund|rtn|rtv)', desc):
        return "RETURN"

    return t

def _row_hash_fn(r):
    keys = [
        str(r.get('Date', '')).strip(),
        str(r.get('Receipt', '')).strip(),
        str(r.get('SO', '')).strip(),
        str(r.get('Item Code', '')).strip(),
        str(r.get('Qty', '')).strip(),
        str(r.get('Sales', '')).strip(),
        str(r.get('Cost', '')).strip(),
    ]
    concat = '|'.join(keys)
    # short, deterministic string id:
    return pd.util.hash_pandas_object(pd.Series(concat)).astype(str).iloc[0]



def clean_dataframe_improved(df):
    print("\n" + "="*50, file=sys.stderr)
    print("CLEANING DATAFRAME", file=sys.stderr)
    print("="*50, file=sys.stderr)

    df_clean = df.copy()
    df_clean['_row_id'] = df_clean.index

    print(f"Starting with {len(df_clean)} rows and {len(df_clean.columns)} columns", file=sys.stderr)
    print("Original columns:", df_clean.columns.tolist(), file=sys.stderr)
    print("\nSample of raw data (first 5 rows):", file=sys.stderr)
    print(df_clean.head().to_string(), file=sys.stderr)

    # -------- capture removals BEFORE mapping --------
    errors_pre = []

    # 1) Remove summary/total rows
    total_patterns = ['grand total', 'total:', 'subtotal']
    rows_before = len(df_clean)
    for pattern in total_patterns:
        mask = df_clean.astype(str).apply(
            lambda x: x.str.contains(pattern, case=False, na=False, regex=False)
        ).any(axis=1)
        if mask.any():
            removed = df_clean.loc[mask].copy()
            removed['error_reason'] = f"summary/total row matched '{pattern}'"
            removed['error_stage']  = 'pre-map'
            errors_pre.append(removed)
            df_clean = df_clean.loc[~mask]
    print(f"[OK] Removed {rows_before - len(df_clean)} total/summary rows", file=sys.stderr)

    # 2) Remove completely empty rows
    empty_mask_all = df_clean.isna().all(axis=1)
    if empty_mask_all.any():
        removed = df_clean.loc[empty_mask_all].copy()
        removed['error_reason'] = 'completely empty row'
        removed['error_stage']  = 'pre-map'
        errors_pre.append(removed)
        df_clean = df_clean.loc[~empty_mask_all]
    print(f"[OK] After removing completely empty rows: {len(df_clean)} rows", file=sys.stderr)

    if len(df_clean) == 0:
        print("WARNING: No data remaining after cleaning!", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()

    print(f"\nData after initial cleaning ({len(df_clean)} rows):", file=sys.stderr)
    print(df_clean.head().to_string(), file=sys.stderr)

    # -------- mapping into target schema --------
    target_columns = [
        'Date', 'Receipt', 'SO', 'Item Code', 'Description', 'Expiration Date',
        'Qty', 'Unit', 'Discount', 'Sales', 'Cost', 'Profit', 'Payment', 'Cashier ID',
        'Category', 'Tab'
    ]
    column_mapping = map_columns_improved(df_clean, target_columns)

    df_final = pd.DataFrame()
    for target_col in target_columns:
        if target_col in column_mapping:
            source_col = column_mapping[target_col]
            # Ensure we don't try to access a column that doesn't exist in df_clean
            if source_col in df_clean.columns:
                df_final[target_col] = df_clean[source_col].copy()
                print(f"[OK] Mapped '{source_col}' -> '{target_col}' ({df_clean[source_col].notna().sum()} values)", file=sys.stderr)
            else:
                print(f"[WARNING] Mapped source column '{source_col}' not found in raw data for '{target_col}'.", file=sys.stderr)
                df_final[target_col] = pd.NA
        else:
            df_final[target_col] = pd.NA
            print(f"[WARNING] Created empty column for '{target_col}' (no source found)", file=sys.stderr)
    df_final['_row_id'] = df_clean['_row_id'].copy()

    # -------- apply categorization BEFORE cleaning --------
    print("\nAPPLYING CATEGORIZATION:", file=sys.stderr)
    print("-" * 30, file=sys.stderr)

    # Load product categories early
    load_product_categories()

    # Apply categorization to each row based on Item Code (much faster and more accurate!)
    categorized_count = 0
    for idx in df_final.index:
        item_code = str(df_final.loc[idx, 'Item Code'] or '').strip()
        description = str(df_final.loc[idx, 'Description'] or '').strip()
        if item_code:
            category = get_category_for_product(item_code, description)
            tab = get_tab_for_product(item_code, description)
            if category:
                df_final.loc[idx, 'Category'] = category
                categorized_count += 1
            if tab:
                df_final.loc[idx, 'Tab'] = tab

    print(f"[OK] Applied categorization to {categorized_count} rows (by Item Code)", file=sys.stderr)

    # --- sanity: did mapping leave everything empty in key fields?
    important = ['Item Code','Description','Qty','Sales','Cost']
    present = [c for c in important if c in df_final.columns]
    if present:
        pct_all_na = (df_final[present].isna().all(axis=1)).mean()
        if pct_all_na > 0.95:
            print("🚨 WARNING: >95% of rows have all important fields empty after mapping.", file=sys.stderr)
            print("Columns in raw file:", list(df.columns), file=sys.stderr)
            print("Column mapping used:", column_mapping, file=sys.stderr)
            # Return what we have so you can inspect it instead of dropping to empty
            return df_final, pd.DataFrame(columns=df_final.columns.tolist() + ['error_reason','error_stage'])


    # 👉 recover Sales before unit/type coercions
    df_final = _recover_sales_if_empty(df_final, df_clean, column_mapping)


    # Project early removals into target schema so the error file is consistent
    errors_aligned = []
    for e in errors_pre:
        aligned = project_to_target(e, column_mapping, target_columns)
        # Ensure _row_id exists in the source 'e' before accessing it
        if '_row_id' in e.columns:
            aligned['_row_id'] = e['_row_id'].values
        else:
            aligned['_row_id'] = pd.NA # Or handle as appropriate if _row_id might be missing

        aligned['error_reason'] = e['error_reason'].values
        aligned['error_stage']  = e['error_stage'].values
        errors_aligned.append(aligned)

    # -------- extract expiration dates from descriptions --------
    if 'Description' in df_final.columns:
        print("\nExtracting expiration dates from descriptions...", file=sys.stderr)
        if 'Expiration Date' not in df_final.columns:
            df_final['Expiration Date'] = None
        df_final['Expiration Date'] = df_final['Expiration Date'].astype('object')
        
        descriptions_with_dates = 0
        for idx in df_final.index:
            original_desc = df_final.loc[idx, 'Description']
            cleaned_desc, exp_date = extract_expiration_from_description(original_desc)
            df_final.loc[idx, 'Description'] = cleaned_desc
            if pd.notna(exp_date) and str(exp_date).strip() != '':
                df_final.loc[idx, 'Expiration Date'] = exp_date
                descriptions_with_dates += 1
        print(f"[OK] Extracted expiration dates from {descriptions_with_dates} descriptions", file=sys.stderr)


    # -------- unit normalization --------
    df_final, unit_msg = normalize_units_on_df(df_final)
    print(f"[OK] {unit_msg}", file=sys.stderr)

    # -------- type cleaning --------
    df_final = clean_data_types_improved(df_final)

    # -------- remove incomplete rows (and collect errors) --------
    df_final, errs_incomplete = remove_incomplete_rows(df_final)

    # -------- drop duplicates (and collect) --------
    errors_post = []
    dupe_subset = [c for c in ['Date','Receipt','SO','Item Code','Qty','Sales','Cost'] if c in df_final.columns]
    if dupe_subset:
        # Ensure subset columns are not all NA before dropping duplicates based on them
        valid_subset = [c for c in dupe_subset if not df_final[c].isna().all()]
        if valid_subset:
            dup_mask = df_final.duplicated(subset=valid_subset, keep='first')
        else:
            dup_mask = df_final.duplicated(keep='first') # Fallback if all key columns are NA
    else:
        dup_mask = df_final.duplicated(keep='first')

    rows_before_dupe_check = len(df_final) # Capture length before dropping duplicates

    if dup_mask.any():
        removed_dup = df_final.loc[dup_mask].copy()
        removed_dup['error_reason'] = 'duplicate row'
        removed_dup['error_stage']  = 'duplicates'
        errors_post.append(removed_dup)
        df_final = df_final.loc[~dup_mask]
    print(f"[OK] Removed {rows_before_dupe_check - len(df_final)} duplicate rows", file=sys.stderr)

    # (Optional) remove rows effectively empty across important fields (after type clean)
    important = [c for c in ['Item Code','Description','Qty','Sales','Cost'] if c in df_final.columns]
    if important:
        mask_all_null = df_final[important].isna().all(axis=1)
        if mask_all_null.any():
            removed_empty = df_final.loc[mask_all_null].copy()
            removed_empty['error_reason'] = 'all important fields empty (post-map)'
            removed_empty['error_stage']  = 'post-map'
            errors_post.append(removed_empty)
            df_final = df_final.loc[~mask_all_null]
            print(f"[OK] Removed {int(mask_all_null.sum())} rows with all key fields null", file=sys.stderr)

    # -------- assemble all errors --------
    errors_all = []
    if errors_aligned:
        errors_all.append(pd.concat(errors_aligned, ignore_index=True))
    if not errs_incomplete.empty:
        errors_all.append(errs_incomplete)
    if errors_post:
        errors_all.append(pd.concat(errors_post, ignore_index=True))
    errors_all = pd.concat(errors_all, ignore_index=True) if errors_all else pd.DataFrame(
        columns=target_columns + ['_row_id','error_reason','error_stage']
    )

    # --- Fill-down for Receipt and SO (carry values into blanks) ---
    for c in ('Receipt', 'SO'):
        if c in df_final.columns:
            df_final[c] = (
                df_final[c]
                .astype('string')
                .replace({'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA, 'None': pd.NA})
                .ffill()
            )
    if not errors_all.empty:
        for c in ('Receipt', 'SO'):
            if c in errors_all.columns:
                errors_all[c] = (
                    errors_all[c]
                    .astype('string')
                    .replace({'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA, 'None': pd.NA})
                    .ffill()
                )

    # -------- final summary --------
    print(f"\n[SUCCESS] Final cleaned DataFrame: {df_final.shape[0]} rows x {df_final.shape[1]} columns", file=sys.stderr)
    if not df_final.empty:
        print("\nSample of final cleaned data:", file=sys.stderr)
        print(df_final.head().to_string(), file=sys.stderr)

    # Drop helper id from the cleaned output (keep it in errors)
    if '_row_id' in df_final.columns:
        df_final = df_final.drop(columns=['_row_id'])

    # --- deterministic RowHash for idempotent loads ---
    df_final['RowHash'] = df_final.apply(_row_hash_fn, axis=1)

    # --- classify transaction type (optional but useful) ---
    df_final['TxnType'] = df_final.apply(classify_txn_type, axis=1)

    return df_final, errors_all


# ==============================================
# I/O helpers
# ==============================================
def save_error_report(errors_df, filename='cleaning_errors.xlsx'):
    if errors_df is None or errors_df.empty:
        print("No removed rows to save. Skipping errors file.", file=sys.stderr)
        return None
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.csv':
        errors_df.to_csv(filename, index=False)
        print(f"[OK] Errors CSV saved to '{filename}' ({len(errors_df)} rows).", file=sys.stderr)
        return filename
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            errors_df.to_excel(writer, index=False, sheet_name='removed_rows')
        print(f"[OK] Errors workbook saved to '{filename}' ({len(errors_df)} rows).", file=sys.stderr)
        return filename
    except Exception as e:
        print(f"Could not write Excel file ({e}). Saving CSV fallback.", file=sys.stderr)
        fallback = os.path.splitext(filename)[0] + '.csv'
        errors_df.to_csv(fallback, index=False)
        print(f"[OK] Errors CSV saved to '{fallback}' ({len(errors_df)} rows).", file=sys.stderr)
        return fallback

def save_cleaned_data(df, filename='cleaned_sales_data.csv'):
    df_out = df.copy()
    # ensure expiration is string for CSV
    if 'Expiration Date' in df_out.columns:
        df_out['Expiration Date'] = df_out['Expiration Date'].astype(str).replace({'NaT':'', 'nan':'', 'None':''})
    df_out.to_csv(filename, index=False)
    print(f"[OK] Cleaned data saved to '{filename}'")
    print(f"Final dataset contains {len(df_out)} records")

    # Optional summary
    print("\nData Summary:")
    print("=" * 50)
    for col in df_out.columns:
        non_null_count = df_out[col].replace(['', 'nan'], pd.NA).notna().sum()
        print(f"{col}: {non_null_count} non-null values")
    return filename

# Replaced browse_for_file with browse_for_files
def browse_for_files():
    """Browse for multiple CSV files"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        print("Opening file browser for multiple files...")
        file_paths = filedialog.askopenfilenames(
            title="Select CSV Data Files (you can select multiple)",
            filetypes=[("CSV files", "*.csv"),("All files", "*.*")],
            initialdir=os.getcwd()
        )
        root.destroy()
        if file_paths:
            print(f"Selected {len(file_paths)} file(s):")
            for fp in file_paths:
                print(f"  - {fp}")
            return list(file_paths)
        else:
            print("No files selected.")
            return []
    except Exception as e:
        print(f"Error opening file browser: {e}")
        print("File browser not available. Please enter file paths manually.")
        return []

# Replaced get_data_source with get_data_sources
def get_data_sources():
    """Get multiple data sources (files or URLs)"""
    print("Data Cleaning Program")
    print("=" * 50)
    print("You can import data from:")
    print("1. Browse for local CSV files (multiple selection)")
    print("2. Enter file paths manually (comma-separated)")
    print("3. Enter URLs to CSV files (comma-separated)")
    print()
    while True:
        choice = input("Choose option (1/2/3) or press Enter to browse: ").strip()
        if not choice or choice == '1':
            sources = browse_for_files()
            if sources:
                return sources
            else:
                retry = input("Would you like to try a different method? (y/n): ").strip().lower()
                if retry != 'y':
                    return []
                continue
        elif choice == '2':
            paths_input = input("Enter file paths (comma-separated): ").strip()
            if not paths_input:
                print("Please enter valid file paths.")
                continue
            sources = [p.strip() for p in paths_input.split(',')]
            valid_sources = []
            for source in sources:
                if os.path.exists(source):
                    print(f"[OK] File found: {source}")
                    valid_sources.append(source)
                else:
                    print(f"[WARNING] File not found: {source}")
            if valid_sources:
                return valid_sources
            else:
                print("No valid files found.")
                continue
        elif choice == '3':
            urls_input = input("Enter URLs (comma-separated): ").strip()
            if not urls_input:
                print("Please enter valid URLs.")
                continue
            sources = [u.strip() for u in urls_input.split(',')]
            valid_sources = []
            for source in sources:
                if source.startswith(('http://', 'https://')):
                    print(f"[OK] URL detected: {source}")
                    valid_sources.append(source)
                else:
                    print(f"[WARNING] Invalid URL: {source}")
            if valid_sources:
                return valid_sources
            else:
                print("No valid URLs found.")
                continue
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

# ==============================================
# Driver
# ==============================================
def fetch_and_clean_sales_data(source):
    print(f"[INFO] Starting to process data from {source}", file=sys.stderr)  # Added log to track when processing starts
    if isinstance(source, list):  # Check if source is a list of files
        all_cleaned_dfs = []  # List to store cleaned dataframes for each file
        all_errors_dfs = []  # List to store error dataframes for each file

        for file in source:  # Loop through each file path in the list
            print(f"[INFO] Processing file {file}...", file=sys.stderr)  # Added log for each file processed
            try:
                csv_content = load_data_from_source(file)  # Load content from the file
                header_row = analyze_csv_structure_strict(csv_content)  # Analyze header
                lines = csv_content.splitlines()
                csv_from_header = "\n".join(lines[header_row:])
                df_raw = pd.read_csv(StringIO(csv_from_header))  # Read the CSV into DataFrame
                print(f"[INFO] Data loaded and processed successfully from {file}", file=sys.stderr)  # Success log

                # Process the dataframe as usual
                df_cleaned, errors_df = clean_dataframe_improved(df_raw)
                all_cleaned_dfs.append(df_cleaned)  # Append cleaned data
                all_errors_dfs.append(errors_df)  # Append errors data
            except Exception as e:
                print(f"[ERROR] Error processing {file}: {str(e)}", file=sys.stderr)  # Error log for file processing failure
                import traceback
                traceback.print_exc()
                continue

        # Combine all cleaned dataframes and errors dataframes
        combined_cleaned_df = pd.concat(all_cleaned_dfs, ignore_index=True)
        combined_errors_df = pd.concat(all_errors_dfs, ignore_index=True)

        return combined_cleaned_df, combined_errors_df  # Return the concatenated results
    else:  # Single file case (previous functionality)
        print(f"[INFO] Processing single file {source}...", file=sys.stderr)  # Added log for single file processing
        csv_content = load_data_from_source(source)
        header_row = analyze_csv_structure_strict(csv_content)
        lines = csv_content.splitlines()
        csv_from_header = "\n".join(lines[header_row:])
        df_raw = pd.read_csv(StringIO(csv_from_header))
        print(f"[INFO] Data loaded and processed successfully from {source}", file=sys.stderr)  # Success log

        # Process the single file as before
        return clean_dataframe_improved(df_raw)


if __name__ == "__main__":
    # Default sample URL (kept for local tests)
    default_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/testdata-D3IKHaVvsV4pdgUGLj27PXAagiN6WO.csv"
    NON_INTERACTIVE = os.getenv("NON_INTERACTIVE", "0") == "1"

    try:
        # Node.js passes: [output_file_path, input_file1, input_file2, ...]
        if len(sys.argv) > 2:
            # First arg is output file path, rest are input files
            output_file_path = sys.argv[1]
            data_sources = sys.argv[2:]
            print(f"Using {len(data_sources)} data source(s) from command line")
            print(f"Output will be saved to: {output_file_path}")
        elif len(sys.argv) == 2:
            # Single argument - treat as input file, use default output
            data_sources = [sys.argv[1]]
            output_file_path = 'cleaned_sales_data_combined.csv'
            print(f"Using 1 data source from command line")
        elif NON_INTERACTIVE:
            print("Running in NON_INTERACTIVE mode...")
            data_sources = [default_url]
            output_file_path = 'cleaned_sales_data_combined.csv'
        else:
            data_sources = get_data_sources()
            if not data_sources:
                print("No data sources provided. Using default test data...")
                data_sources = [default_url]
            output_file_path = 'cleaned_sales_data_combined.csv'

        # -------- CLEAN MULTIPLE FILES --------
        all_cleaned_dfs = []
        all_errors_dfs = []
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(data_sources)} FILE(S)")
        print(f"{'='*60}")
        
        for idx, data_source in enumerate(data_sources, 1):
            print(f"\n[{idx}/{len(data_sources)}] Processing: {data_source}")
            print("-" * 60)
            
            try:
                cleaned_df, errors_df = fetch_and_clean_sales_data(data_source)
                
                # Add source file info to track which file each row came from
                source_name = os.path.basename(data_source) if not data_source.startswith(('http://', 'https://')) else data_source
                cleaned_df['_source_file'] = source_name
                if not errors_df.empty:
                    errors_df['_source_file'] = source_name
                
                all_cleaned_dfs.append(cleaned_df)
                all_errors_dfs.append(errors_df)
                
                print(f"[OK] Cleaned {len(cleaned_df)} rows from {source_name}")
                
            except Exception as e:
                print(f"[ERROR] Error processing {data_source}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_cleaned_dfs:
            raise ValueError("No files were successfully processed!")
        
        print(f"\n{'='*60}")
        print("COMBINING ALL CLEANED DATA")
        print(f"{'='*60}")
        
        combined_cleaned_df = pd.concat(all_cleaned_dfs, ignore_index=True)
        combined_errors_df = pd.concat(all_errors_dfs, ignore_index=True) if all_errors_dfs else pd.DataFrame()
        
        print(f"[OK] Combined {len(all_cleaned_dfs)} file(s) into {len(combined_cleaned_df)} total rows")
        print(f"  Breakdown by file:")
        for df in all_cleaned_dfs:
            source = df['_source_file'].iloc[0] if '_source_file' in df.columns and len(df) > 0 else 'unknown'
            print(f"    - {source}: {len(df)} rows")
        
        # Remove the helper column before saving
        if '_source_file' in combined_cleaned_df.columns:
            combined_cleaned_df = combined_cleaned_df.drop(columns=['_source_file'])
        if '_source_file' in combined_errors_df.columns:
            combined_errors_df = combined_errors_df.drop(columns=['_source_file'])

        save_cleaned_data(combined_cleaned_df, output_file_path)
        errors_file = output_file_path.replace('.csv', '_errors.csv')
        save_error_report(combined_errors_df, errors_file)

        print(f"\n[SUCCESS] Data cleaning completed successfully!")
        print(f"[FILE] Combined output file: {output_file_path}")
        print(f"[FILE] Total cleaned rows: {len(combined_cleaned_df)}")

        # -------- LOAD (Postgres CAPSTONE) --------
        file_name_for_run = f"Combined_{len(data_sources)}_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n[LOADING] Loading combined cleaned data into Postgres (CAPSTONE)...")
        run_id = run_full_load(
            file_name=file_name_for_run,
            raw_df=combined_cleaned_df,       # <-- use raw_df
            ensure_schema_once=False
        )

        print(f"[SUCCESS] ETL load completed. run_id = {run_id}")
        print(f"   Loaded {len(combined_cleaned_df)} rows from {len(data_sources)} source file(s)", file=sys.stderr)

        print(json.dumps({
            "success": True,
            "message": f"Successfully processed {len(data_sources)} file(s) and loaded {len(combined_cleaned_df)} rows",
            "cleanedFile": output_file_path,
            "rowsProcessed": len(combined_cleaned_df),
            "filesProcessed": len(data_sources),
            "runId": run_id
        }))



    except Exception as e:
        print(f"[ERROR] Error occurred: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))
        sys.exit(1)
