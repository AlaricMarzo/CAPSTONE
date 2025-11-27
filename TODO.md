# Fix ERR_CONNECTION_REFUSED and Python Spawn Issues

## Frontend Port Fixes
- [ ] Update vite.config.ts proxy targets from 5050 to 8080
- [ ] Update Reports.tsx hardcoded localhost:5050 to 8080
- [ ] Update PredictiveAnalytics.tsx hardcoded localhost:5050 to 8080

## Backend Python Spawn Fixes
- [ ] Update upload.js spawn from "python" to "python3"
- [ ] Update analytics.js spawn from "python" to "python3"

## Testing
- [ ] Restart frontend and backend
- [ ] Test API calls work
- [ ] Test Python scripts run without ENOENT
