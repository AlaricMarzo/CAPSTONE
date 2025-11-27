import express from "express"
import cors from "cors"
import dotenv from "dotenv"
import morgan from "morgan"
import path from "path"
import uploadRouter from "./routes/upload.js"
import analyticsRouter from "./routes/analytics.js"

dotenv.config()
const app = express()

app.use(cors())
app.use(express.json())
// Remove morgan logging to prevent terminal spam from polling
// app.use(morgan("dev"))

app.get("/health", (_, res) => res.json({ status: "OK" }))

app.use("/api", uploadRouter)
app.use("/api/analytics", analyticsRouter)

// Serve static files from the frontend dist directory
app.use(express.static(path.join(process.cwd(), '../frontend/dist')))

// Catch-all handler: send back index.html for any non-API routes (for SPA routing)
app.get('*', (req, res) => {
  res.sendFile(path.join(process.cwd(), '../frontend/dist/index.html'));
});

import multer from "multer";

// Global error handler — catches any remaining Multer or server errors
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    return res
      .status(400)
      .json({ success: false, error: `Upload error: ${err.message}` });
  }
  if (err) {
    return res
      .status(500)
      .json({ success: false, error: err.message || "Server error" });
  }
  next();
});


const PORT = process.env.PORT || 5050
app.listen(PORT, () => console.log(`✅ Server running on http://localhost:${PORT}`))
