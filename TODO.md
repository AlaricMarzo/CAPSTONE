# TODO: Fix Descriptive Analytics Dashboard

## Backend Updates
- [ ] Update /descriptive route in analytics.js to fetch all KPI outputs (seasonal indices, yearly data, top10 by qty, PNG images)
- [ ] Add fetching for clustering by category and tab (summaries and images)
- [ ] Encode PNG images to base64 data URLs for frontend display
- [ ] Include all MBA outputs (currently only rules)
- [ ] Test backend response includes all data accurately

## Frontend Updates
- [ ] Add new chart for monthly sales growth rate
- [ ] Add trend chart for active SKUs over time
- [ ] Add seasonal index visualization
- [ ] Add top 10 products by quantity chart
- [ ] Expand clustering section to display category and tab clusters with images
- [ ] Display KPI PNG images in appropriate cards
- [ ] Ensure all data is displayed accurately and correctly
- [ ] Test dashboard loads and displays all outputs

## Testing
- [ ] Run descriptive analytics pipeline to generate fresh outputs
- [ ] Verify backend API returns complete data
- [ ] Check frontend renders all sections without errors
- [ ] Validate data accuracy against generated files
