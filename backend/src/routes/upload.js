import express from "express"
import multer from "multer"
import path from "path"
import fs from "fs"
import { spawn } from "child_process"
import { v4 as uuidv4 } from "uuid"
import pg from "pg"
import { fileURLToPath } from "url"
import { dirname } from "path"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const router = express.Router()

// In-memory job storage (fallback)
const jobs = new Map()

// Database functions for job persistence
async function saveJobToDB(jobId, jobData) {
  try {
    const dsn = process.env.DATABASE_URL
    if (!dsn) return // Skip if no database

    const client = new pg.Client({ connectionString: dsn })
    await client.connect()

    await client.query(`
      INSERT INTO job_status (job_id, status, progress, message, data, updated_at)
      VALUES ($1, $2, $3, $4, $5, NOW())
      ON CONFLICT (job_id)
      DO UPDATE SET
        status = EXCLUDED.status,
        progress = EXCLUDED.progress,
        message = EXCLUDED.message,
        data = EXCLUDED.data,
        updated_at = NOW()
    `, [jobId, jobData.status, jobData.progress, jobData.message, JSON.stringify(jobData)])

    await client.end()
  } catch (error) {
    console.error("[Backend] Error saving job to DB:", error)
  }
}

async function getJobFromDB(jobId) {
  try {
    const dsn = process.env.DATABASE_URL
    if (!dsn) return null

    const client = new pg.Client({ connectionString: dsn })
    await client.connect()

    const result = await client.query(`
      SELECT status, progress, message, data
      FROM job_status
      WHERE job_id = $1
      ORDER BY updated_at DESC
      LIMIT 1
    `, [jobId])

    await client.end()

    if (result.rows.length > 0) {
      const row = result.rows[0]
      return {
        status: row.status,
        progress: row.progress,
        message: row.message,
        ...JSON.parse(row.data || '{}')
      }
    }
  } catch (error) {
    console.error("[Backend] Error getting job from DB:", error)
  }
  return null
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "../../uploads")
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true })
    }
    cb(null, uploadDir)
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9)
    cb(null, file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname))
  },
})

const upload = multer({
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "text/csv" || file.originalname.endsWith(".csv")) {
      cb(null, true)
    } else {
      cb(new Error("Only CSV files are allowed"))
    }
  },
})

// -------------------- Background Processing Function --------------------
async function processUpload(jobId, files) {
  try {
    const jobData1 = { status: 'processing', progress: 10, message: 'Files uploaded, starting data cleaning...' }
    jobs.set(jobId, jobData1)
    await saveJobToDB(jobId, jobData1)

    const cleanedDir = path.join(__dirname, "../../cleaned")
    if (!fs.existsSync(cleanedDir)) fs.mkdirSync(cleanedDir, { recursive: true })

    const uploadedFiles = files.map((file) => file.path)
    const timestamp = Date.now()
    const outputPath = path.join(cleanedDir, `combined_cleaned_${timestamp}.csv`)
    const cleanScriptPath = path.join(__dirname, "../../scripts/clean_data.py")

    console.log("[Backend] Starting data cleaning and loading...")
    const jobData2 = { status: 'processing', progress: 20, message: 'Validating file headers...' }
    jobs.set(jobId, jobData2)
    await saveJobToDB(jobId, jobData2)

    // Simulate header validation progress
    await new Promise(resolve => setTimeout(resolve, 500))
    const jobData3 = { status: 'processing', progress: 40, message: 'Normalizing units and extracting dates...' }
    jobs.set(jobId, jobData3)
    await saveJobToDB(jobId, jobData3)

    // Simulate unit normalization progress
    await new Promise(resolve => setTimeout(resolve, 500))
    const jobData4 = { status: 'processing', progress: 60, message: 'Filtering data quality...' }
    jobs.set(jobId, jobData4)
    await saveJobToDB(jobId, jobData4)

    // Simulate data quality filtering progress
    await new Promise(resolve => setTimeout(resolve, 500))
    const jobData5 = { status: 'processing', progress: 80, message: 'Running data cleaning script...' }
    jobs.set(jobId, jobData5)
    await saveJobToDB(jobId, jobData5)

    // Run the Python cleaning script
    const pythonProcess = spawn("python3", [cleanScriptPath, ...uploadedFiles, outputPath], {
      stdio: ["ignore", "pipe", "pipe"],
    })

    let stdout = ""
    let stderr = ""

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString()
    })

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString()
    })

    pythonProcess.on("close", async (code) => {
      if (code === 0) {
        console.log("[Backend] Data cleaning completed successfully")
        const jobData6 = { status: 'completed', progress: 100, message: 'Data cleaning completed successfully', outputPath }
        jobs.set(jobId, jobData6)
        await saveJobToDB(jobId, jobData6)
      } else {
        console.error("[Backend] Python script failed:", stderr)
        const jobData6 = { status: 'failed', progress: 100, message: `Data cleaning failed: ${stderr}`, error: stderr }
        jobs.set(jobId, jobData6)
        await saveJobToDB(jobId, jobData6)
      }
    })

    pythonProcess.on("error", async (error) => {
      console.error("[Backend] Failed to start Python process:", error)
      const jobData6 = { status: 'failed', progress: 100, message: 'Failed to execute data cleaning script', error: error.message }
      jobs.set(jobId, jobData6)
      await saveJobToDB(jobId, jobData6)
    })
  } catch (error) {
    console.error("[Backend] Error in processUpload:", error)
    const jobDataError = { status: 'failed', progress: 100, message: 'Unexpected error occurred', error: error.message }
    jobs.set(jobId, jobDataError)
    await saveJobToDB(jobId, jobDataError)
  }
}

// Upload endpoint
router.post("/upload", upload.array("files", 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ success: false, error: "No files uploaded" })
    }

    const jobId = uuidv4()
    const jobData = { status: 'pending', progress: 0, message: 'Upload received, processing...' }
    jobs.set(jobId, jobData)
    await saveJobToDB(jobId, jobData)

    // Start background processing
    processUpload(jobId, req.files)

    res.json({
      success: true,
      jobId,
      message: "Files uploaded successfully. Processing started.",
      status: jobData.status,
    })
  } catch (error) {
    console.error("[Backend] Upload error:", error)
    res.status(500).json({ success: false, error: "Upload failed", details: error.message })
  }
})

// Job status endpoint
router.get("/status/:jobId", async (req, res) => {
  try {
    const { jobId } = req.params

    // Check in-memory first
    let job = jobs.get(jobId)

    // If not found in memory, check database
    if (!job) {
      job = await getJobFromDB(jobId)
    }

    if (!job) {
      return res.status(404).json({ error: "Job not found" })
    }

    res.json(job)
  } catch (error) {
    console.error("[Backend] Status check error:", error)
    res.status(500).json({ error: "Failed to get job status", details: error.message })
  }
})

export default router
