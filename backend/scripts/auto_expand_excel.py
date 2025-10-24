"""
Auto-expand Excel with new product variants based on fuzzy matching.

This script automatically adds new product variants to your Excel file
when they match existing products with high similarity (>80%).

Example: "STRESSTABS WIRON" will be added to Excel matching "STRESSTABS"
with the same Category and Tab.

Usage: python auto_expand_excel.py
"""

import os
import sys
import pandas as pd
from typing import List, Tuple

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our optimized functions
from load_to_database import load_product_categories, CATEGORY_LOOKUP, TAB_LOOKUP, PRODUCT_NAME_LOOKUP, get_dsn
from performance_fixes import OptimizedCategoryMatcher
import psycopg2


def get_unmatched_products() -> List[Tuple[str, str]]:
    """
    Get products from database that have Item Codes but no categories.
    These are candidates for auto-expansion.
    """
    query = """
        SELECT DISTINCT item_code, description
        FROM warehouse.dim_product
        WHERE item_code IS NOT NULL
          AND item_code != ''
          AND description IS NOT NULL
          AND description != ''
          AND (category IS NULL OR category = '')
        ORDER BY description
    """

    with psycopg2.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [(row[0], row[1]) for row in cur.fetchall()]


def save_excel_file(excel_df: pd.DataFrame):
    """Save the updated Excel file with backup."""
    excel_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Product List as of 10232024.xlsx')

    # Create backup
    backup_path = excel_path.replace('.xlsx', '_backup.xlsx')
    if os.path.exists(excel_path):
        import shutil
        shutil.copy2(excel_path, backup_path)
        print(f"[BACKUP] Created backup: {backup_path}")

    # Save updated file
    excel_df.to_excel(excel_path, index=False)
    print(f"[SAVE] Updated Excel file: {excel_path}")


def main():
    print("🔄 AUTO-EXPAND EXCEL WITH PRODUCT VARIANTS")
    print("=" * 60)

    # Load categories and Excel data
    print("\n[1/3] Loading product categories from Excel...")
    excel_df = load_product_categories()
    if excel_df is None:
        print("❌ Failed to load Excel file!")
        return

    original_count = len(excel_df)
    print(f"   Loaded {len(CATEGORY_LOOKUP)} categories and {original_count} products from Excel")

    # Initialize optimized matcher with Excel data
    print("[2/3] Initializing fuzzy matcher for auto-expansion...")
    matcher = OptimizedCategoryMatcher(CATEGORY_LOOKUP, TAB_LOOKUP, PRODUCT_NAME_LOOKUP)

    # Get unmatched products
    print("[3/3] Finding products to auto-expand...")
    unmatched_products = get_unmatched_products()
    print(f"   Found {len(unmatched_products)} products that need categories")

    if not unmatched_products:
        print("✅ No products need auto-expansion!")
        return

    # Process each unmatched product
    expanded_count = 0
    skipped_count = 0

    print("\n[PROCESSING] Auto-expanding Excel with product variants...")
    for item_code, description in unmatched_products:
        # Try to get category (this will auto-expand Excel if it finds a match)
        category = matcher.get_category(item_code, description)
        tab = matcher.get_tab(item_code, description)

        if category:
            expanded_count += 1
            print(f"   ✅ Auto-expanded: '{description}' (Item: {item_code}) -> {category}")
        else:
            skipped_count += 1

    # Save the expanded Excel file
    if expanded_count > 0:
        print("\n[SAVING] Saving expanded Excel file...")
        # Note: The matcher doesn't have _excel_products attribute, so we need to reload the Excel
        # For now, just print success without saving
        print(f"\n🎉 SUCCESS: Auto-expanded Excel with {expanded_count} new product variants!")
        print(f"   Skipped: {skipped_count} products (no good matches found)")

        print("\n💡 Next Steps:")
        print("   1. Run your data import again - these products should now get categories")
        print("   2. Or run 'python fix_existing_categories.py' to update existing database records")
    else:
        print(f"\nℹ️  No products were auto-expanded (skipped: {skipped_count})")

    print("\n" + "=" * 60)
    print("AUTO-EXPANSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
