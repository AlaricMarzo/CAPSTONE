#!/usr/bin/env python3
"""
Test script to verify category matching works with numbers retained in normalize_text.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'scripts'))

try:
    from load_to_database import normalize_text, load_product_categories, get_category_for_product, get_tab_for_product  # type: ignore
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_normalize_text():
    """Test that normalize_text retains numbers."""
    print("Testing normalize_text function...")

    test_cases = [
        ("Product 123", "product 123"),
        ("Item #456", "item 456"),
        ("Test-789", "test789"),
        ("ABC 123 DEF", "abc 123 def"),
        ("No Numbers", "no numbers"),
    ]

    for input_text, expected in test_cases:
        result = normalize_text(input_text)
        if result == expected:
            print(f"✓ '{input_text}' -> '{result}'")
        else:
            print(f"✗ '{input_text}' -> '{result}' (expected '{expected}')")
            return False

    print("normalize_text tests passed!")
    return True

def test_category_matching():
    """Test category matching with sample data."""
    print("\nTesting category matching...")

    # Load categories
    load_product_categories()

    # Test cases with item codes and descriptions that might have numbers
    test_cases = [
        ("12345", "Sample Product 123"),  # Exact barcode match
        ("99999", "Unknown Product 456"),  # No match, fallback to fuzzy
        ("", "Another Product 789"),  # No barcode, fuzzy match
    ]

    for item_code, description in test_cases:
        category = get_category_for_product(item_code, description)
        tab = get_tab_for_product(item_code, description)
        print(f"Item Code: '{item_code}', Description: '{description}' -> Category: '{category}', Tab: '{tab}'")

    print("Category matching tests completed!")
    return True

if __name__ == "__main__":
    print("Running category matching tests...\n")

    success = True
    success &= test_normalize_text()
    success &= test_category_matching()

    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
