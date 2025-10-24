# TODO: Fix Product Categorization and Normalization

## Approved Plan
- Move category loading to the beginning of `run_full_load` function before any cleaning steps to ensure categorization uses numbers for better matching.
- Modify `normalize_text` function to retain numbers by removing the digit-stripping line.

## Steps to Complete
- [x] Step 1: Add `load_product_categories()` call at the start of `run_full_load` function, right after `begin_run`.
- [x] Step 2: Remove the line `text = re.sub(r'\d+', '', text)` from the `normalize_text` function.
- [ ] Step 3: Test the changes to ensure logic is not broken (e.g., run a test script to verify category matching works with numbers retained).
