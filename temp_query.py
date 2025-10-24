import sys
sys.path.append('backend/scripts')
from load_to_database import db

with db() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT item_code, description, category, tab
            FROM warehouse.dim_product
            WHERE item_code = '123'
        """)
        row = cur.fetchone()
        if row:
            print(f"Item Code: {row[0]}")
            print(f"Description: {row[1]}")
            print(f"Category: {row[2]}")
            print(f"Tab: {row[3]}")
        else:
            print("No data found for item_code '123'")
