import express from "express"
import cors from "cors"
import dotenv from "dotenv"
import morgan from "morgan"
import uploadRouter from "./routes/upload.js"

dotenv.config()
const app = express()

app.use(cors())
app.use(express.json())
// Remove morgan logging to prevent terminal spam from polling
// app.use(morgan("dev"))

app.get("/health", (_, res) => res.json({ status: "OK" }))

app.use("/api", uploadRouter)

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
