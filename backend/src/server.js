import express from "express"
import cors from "cors"
import dotenv from "dotenv"
import morgan from "morgan"
import uploadRouter from "./routes/upload.js"

dotenv.config()
const app = express()

app.use(cors())
app.use(express.json())
app.use(morgan("dev"))

app.get("/health", (_, res) => res.json({ status: "OK" }))

app.use("/api", uploadRouter)

const PORT = process.env.PORT || 4000
app.listen(PORT, () => console.log(`✅ Server running on http://localhost:${PORT}`))
