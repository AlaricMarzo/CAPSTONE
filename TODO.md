# TODO: Fix Polling Mismatch for Upload Progress

## Steps to Complete
- [x] Edit backend/src/server.js: Change app.use("/api", uploadRouter) to app.use("/api/upload", uploadRouter)
- [x] Edit backend/src/routes/upload.js: Change router.post("/upload", ...) to router.post("/", ...)
- [x] Edit frontend/src/pages/upload.tsx: Change fetch("/api/job/${jobId}") to fetch("/api/upload/job/${jobId}") in pollJobStatus
- [x] Test changes locally or redeploy to Railway to verify polling works and progress reaches 100%
