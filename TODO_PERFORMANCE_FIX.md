# Performance Optimization Plan for Data Import

## Issues Identified

### 1. **CRITICAL: O(N²) Category Matching** ⚠️
- Current: 3 full loops through all categories for each product
- Fuzzy matching with `difflib.SequenceMatcher` is extremely slow
- For 1000 products × 1000 rows = 3,000,000 comparisons

### 2. **Inefficient Database Queries**
- Duplicate checking in chunks causes multiple round-trips
- Row-by-row category updates instead of batch operations

### 3. **Redundant Category Lookups**
- Categories looked up multiple times for same products
- No caching of results

## Proposed Solutions

### Phase 1: Optimize Category Matching (HIGHEST PRIORITY)
1. **Pre-compute normalized lookup keys**
   - Build reverse index: normalized_desc -> category
   - Use set operations for partial matching
   
2. **Replace fuzzy matching with faster alternatives**
   - Use trigram similarity (PostgreSQL pg_trgm)
   - Or use rapidfuzz library (10-100x faster than difflib)
   - Only fuzzy match on cache misses

3. **Add caching layer**
   - Cache category lookups during processing
   - Avoid repeated lookups for same descriptions

### Phase 2: Optimize Database Operations
1. **Batch duplicate checking**
   - Single query with temporary table
   - Use PostgreSQL's COPY for bulk operations

2. **Bulk category updates**
   - Collect all updates, execute once
   - Use execute_values for batch inserts

### Phase 3: Parallel Processing
1. **Process files in parallel** (if multiple files)
2. **Parallelize category matching** (if single large file)

## Implementation Priority

1. ✅ **IMMEDIATE**: Add category lookup caching
2. ✅ **HIGH**: Replace difflib with rapidfuzz
3. ✅ **HIGH**: Batch database operations
4. ⏳ **MEDIUM**: Pre-compute lookup indexes
5. ⏳ **LOW**: Parallel processing

## Expected Performance Improvement

- Current: ~5-10 minutes for 1000 rows
- After optimization: ~10-30 seconds for 1000 rows
- **50-100x speedup expected**
