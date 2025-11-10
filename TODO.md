# TODO: Add Reports Tab UI for Downloading CSV and PNG Files

## Backend Changes
- [ ] Add route to list available CSV and PNG files from output directories (descriptive_output, prescriptive_output, etc.)
- [ ] Add route to download specific files by path

## Frontend Changes
- [ ] Create ReportsPage component (frontend/src/pages/Reports.tsx)
- [ ] Add download functionality for CSV and PNG files
- [ ] Update Dashboard.tsx to render ReportsPage for "reports" tab

## Testing
- [ ] Test file listing and downloads
- [ ] Ensure UI is responsive and user-friendly

## Predictive Analytics Updates
- [x] Remove hardcoded data from predictive route in analytics.js
- [x] Set rf_predicted and xgb_predicted to 0 in forecast data
- [x] Set random_forest and xgboost model performance metrics to 0
- [x] Remove feature_importance from response
- [x] Use actual lower_bound and upper_bound from SARIMA forecasts for confidence intervals
- [ ] Run predictive models to generate fresh outputs for RF and XGBoost
- [ ] Update API to read actual RF and XGBoost outputs if available
- [ ] Verify frontend displays updated predictive data correctly
