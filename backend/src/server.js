import express from "express"
import cors from "cors"
import dotenv from "dotenv"
import morgan from "morgan"
import uploadRouter from "./routes/upload.js"
import analyticsRouter from "./routes/analytics.js"
import path from "path"
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

dotenv.config()
const app = express()

app.use(cors())
app.use(express.json())
// Remove morgan logging to prevent terminal spam from polling
// app.use(morgan("dev"))

// Serve static files from the frontend build directory
app.use(express.static(path.join(__dirname, '../../frontend/dist')))

app.get("/health", (_, res) => res.json({ status: "OK" }))

app.use("/api", uploadRouter)
app.use("/api/analytics", analyticsRouter)

// Catch all handler: send back index.html for any non-API routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../../frontend/dist/index.html'))
})

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
