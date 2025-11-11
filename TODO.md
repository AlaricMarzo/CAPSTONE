# TODO: Fix descriptive.py and dbscan.py for cluster graphs

## Steps:
1. Fix axis limit bug in dbscan.py _scatter_plot: change x_iqr to y_iqr in ymax calculation.
2. Fix n_noise inconsistency in cluster_all: use "total_qty" for global noise instead of "n".
3. Test the fixes by running descriptive.py to ensure all graphs generate.

## Status:
- [ ] Step 1: Edit dbscan.py for axis limit fix.
- [ ] Step 2: Edit dbscan.py for n_noise fix.
- [ ] Step 3: Run descriptive.py and verify outputs.
