"""
Auto-add new products to Excel file based on fuzzy matching.

This script will:
1. Find products with NULL categories in the database
2. Use fuzzy matching to find similar products in the Excel file
3. Add new products to the Excel file with matched categories/tabs
4. Update the database with the new categories

Usage: python auto_add_products.py
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Tuple

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our optimized functions
from performance_fixes import OptimizedCategoryMatcher
from load_to_database import load_product_categories, get_dsn, CATEGORY_LOOKUP, TAB_LOOKUP, PRODUCT_NAME_LOOKUP
import psycopg2


def get_products_without_categories() -> List[Tuple[str, str]]:
    """
    Get products from database that have NULL or empty categories.

    Returns:
        List of (item_code, description) tuples
    """
    query = """
        SELECT DISTINCT item_code, description
        FROM warehouse.dim_product
        WHERE (category IS NULL OR category = '')
          AND item_code IS NOT NULL
          AND description IS NOT NULL
          AND description != ''
        ORDER BY description
    """

    with psycopg2.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [(row[0], row[1]) for row in cur.fetchall()]


def load_excel_products() -> pd.DataFrame:
    """Load the product list Excel file."""
    excel_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Product List as of 10232024.xlsx')
    return pd.read_excel(excel_path)


def find_similar_products(description: str, excel_df: pd.DataFrame, matcher: OptimizedCategoryMatcher) -> List[Tuple[str, float, str, str]]:
    """
    Find products in Excel that are similar to the given description.

    Returns:
        List of (excel_product_name, similarity_score, category, tab) tuples
    """
    similar_products = []

    # Normalize the description
    desc_norm = matcher._normalize_text(description)

    # Check against all products in Excel
    for _, row in excel_df.iterrows():
        excel_product = str(row.get('Product Name', '')).strip()
        if not excel_product:
            continue

        excel_norm = matcher._normalize_text(excel_product)

        # Calculate similarity
        if matcher.__class__.__module__.split('.')[-1] == 'performance_fixes':
            # Use rapidfuzz if available
            try:
                from rapidfuzz import fuzz
                score = fuzz.ratio(desc_norm, excel_norm) / 100.0
            except ImportError:
                import difflib
                score = difflib.SequenceMatcher(None, desc_norm, excel_norm).ratio()
        else:
            import difflib
            score = difflib.SequenceMatcher(None, desc_norm, excel_norm).ratio()

        if score > 0.7:  # 70% similarity threshold
            category = str(row.get('Category', '')).strip()
            tab = str(row.get('Tab', '')).strip()
            similar_products.append((excel_product, score, category, tab))

    # Sort by similarity score (highest first)
    similar_products.sort(key=lambda x: x[1], reverse=True)
    return similar_products[:5]  # Return top 5 matches


def add_product_to_excel(excel_df: pd.DataFrame, item_code: str, description: str, category: str, tab: str) -> pd.DataFrame:
    """
    Add a new product to the Excel dataframe.

    Returns:
        Updated dataframe
    """
    # Create new row
    new_row = {
        'Product Name': description,
        'Tab': tab,
        'Category': category,
        'Barcode': item_code,
        'SKU Number': f'AUTO-{item_code}'
    }

    # Add to dataframe
    excel_df = pd.concat([excel_df, pd.DataFrame([new_row])], ignore_index=True)
    return excel_df


def save_excel_file(excel_df: pd.DataFrame, filename: str = None):
    """Save the updated Excel file."""
    if filename is None:
        filename = 'Product List as of 10232024.xlsx'

    excel_path = os.path.join(os.path.dirname(__file__), '..', '..', filename)

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
    print("🔍 AUTO-ADD PRODUCTS TO EXCEL")
    print("=" * 50)

    # Load data
    print("\n[1/5] Loading product categories...")
    load_product_categories()

    print("[2/5] Loading Excel file...")
    excel_df = load_excel_products()
    print(f"   Found {len(excel_df)} products in Excel")

    print("[3/5] Finding products without categories...")
    products_without_categories = get_products_without_categories()
    print(f"   Found {len(products_without_categories)} products without categories")

    if not products_without_categories:
        print("✅ No products need category assignment!")
        return

    # Initialize matcher
    matcher = OptimizedCategoryMatcher(CATEGORY_LOOKUP, TAB_LOOKUP, PRODUCT_NAME_LOOKUP)

    # Process each product
    added_count = 0
    skipped_count = 0

    print("\n[4/5] Processing products...")
    for item_code, description in products_without_categories:
        print(f"\n   Processing: {description} (Code: {item_code})")

        # Find similar products
        similar = find_similar_products(description, excel_df, matcher)

        if similar:
            # Use the best match
            best_match, score, category, tab = similar[0]
            print(f"      -> Match: '{best_match}' ({score:.2f})")

            # Confirm with user
            response = input(f"   Add this product to Excel? (y/n): ").strip().lower()
            if response == 'y':
                excel_df = add_product_to_excel(excel_df, item_code, description, category, tab)
                added_count += 1
                print("   ✅ Added to Excel")
            else:
                print("   ⏭️  Skipped")
                skipped_count += 1
        else:
            print("     -> No similar products found")
            skipped_count += 1

    print("\n[5/5] Saving results...")
    if added_count > 0:
        save_excel_file(excel_df)
        print(f"\n✅ SUCCESS: Added {added_count} products to Excel file")
        print("   Note: You'll need to reload categories for the changes to take effect")
    else:
        print(f"\nℹ️  No products were added (skipped: {skipped_count})")

    print("\n" + "=" * 50)
