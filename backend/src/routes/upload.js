import express from "express"
import multer from "multer"
import path from "path"
import fs from "fs"
import { spawn } from "child_process"
import { fileURLToPath } from "url"
import { dirname } from "path"
import { v4 as uuidv4 } from "uuid"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const router = express.Router()

// In-memory job store (for demo; use Redis/DB in production)
const jobs = new Map()

// -------------------- Multer setup --------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "../../uploads")
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true })
    }
    cb(null, uploadDir)
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`)
  },
})

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "text/csv" || file.originalname.endsWith(".csv")) {
      cb(null, true)
    } else {
      cb(new Error("Only CSV files are allowed"))
    }
  },
})

// -------------------- Helper: Run Python Script --------------------
function runPythonScript(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.env.PYTHON_CMD || "python";
    const python = spawn(pythonCmd, [scriptPath, ...args], {
      env: { ...process.env, NON_INTERACTIVE: "1" }
    });

    let output = "";
    let errorOutput = "";

    python.stdout.on("data", (data) => {
      output += data.toString();
      console.log("[Python stdout]:", data.toString());
    });

    python.stderr.on("data", (data) => {
      errorOutput += data.toString();
      console.error("[Python stderr]:", data.toString());
    });

    // Timeout after 60 seconds
    const timeout = setTimeout(() => {
      reject(new Error("Python script execution timed out"));
      python.kill();
    }, 20 * 60 * 1000); // 5 minutes
    // Timeout after 60 seconds

    python.on("close", (code) => {
      clearTimeout(timeout);  // Clear timeout if script finishes
      if (code !== 0) {
        console.error("[Backend] Python error output:", errorOutput);
        reject(new Error(`Python script failed with code ${code}: ${errorOutput}`));
        return;
      }

      try {
        const jsonMatch = output.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const result = JSON.parse(jsonMatch[0]);
          resolve(result);
        } else {
          resolve({ output: output.trim(), success: false, error: "No JSON output from script" });
        }
      } catch (e) {
        console.error("[Backend] JSON parse error:", e.message);
        resolve({ output: output.trim(), success: false, error: `Failed to parse JSON: ${e.message}` });
      }
    });
  });
}



// -------------------- Upload Route (Async with Job Tracking) --------------------
router.post(
  "/upload",
  (req, res, next) => {
    upload.any()(req, res, (err) => {
      if (err) {
        const msg = err.name === "MulterError" ? `Upload error: ${err.message}` : err.message
        console.error("[Multer Error]", msg)
        return res.status(400).json({ success: false, error: msg })
      }
      next()
    })
  },
  async (req, res) => {
    try {
      console.log("Incoming file fields:", (req.files || []).map(f => f.fieldname))

      if (!req.files || req.files.length === 0) {
        return res.status(400).json({ success: false, error: "No files uploaded" })
      }

      const jobId = uuidv4()
      jobs.set(jobId, { status: 'processing', progress: 0, message: 'Upload received, starting processing...' })

      // Respond immediately with job ID
      res.json({
        success: true,
        jobId,
        message: "Upload initiated. Processing in background.",
        filesProcessed: req.files.length,
      })

      // Process in background
      processUpload(jobId, req.files)

    } catch (error) {
      console.error("[Backend] Upload error:", error)
      res.status(500).json({ success: false, error: error.message })
    }
  }
)

// -------------------- Job Status Route --------------------
router.get('/job/:jobId', (req, res) => {
  const { jobId } = req.params
  const job = jobs.get(jobId)

  if (!job) {
    return res.status(404).json({ success: false, error: 'Job not found' })
  }

  // Set no-cache headers to prevent browser caching
  res.set({
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
  })

  res.json({ success: true, ...job })
})

// -------------------- Background Processing Function --------------------
async function processUpload(jobId, files) {
  try {
    jobs.set(jobId, { status: 'processing', progress: 10, message: 'Files uploaded, starting data cleaning...' })

    const cleanedDir = path.join(__dirname, "../../cleaned")
    if (!fs.existsSync(cleanedDir)) fs.mkdirSync(cleanedDir, { recursive: true })

    const uploadedFiles = files.map((file) => file.path)
    const timestamp = Date.now()
    const outputPath = path.join(cleanedDir, `combined_cleaned_${timestamp}.csv`)
    const cleanScriptPath = path.join(__dirname, "../../scripts/clean_data.py")

    console.log("[Backend] Starting data cleaning and loading...")
    jobs.set(jobId, { status: 'processing', progress: 20, message: 'Validating file headers...' })

    // Simulate header validation progress
    await new Promise(resolve => setTimeout(resolve, 500))
    jobs.set(jobId, { status: 'processing', progress: 40, message: 'Normalizing units and extracting dates...' })

    // Simulate unit normalization progress
    await new Promise(resolve => setTimeout(resolve, 500))
    jobs.set(jobId, { status: 'processing', progress: 60, message: 'Filtering data quality...' })

    // Simulate data quality filtering progress
    await new Promise(resolve => setTimeout(resolve, 500))
    jobs.set(jobId, { status: 'processing', progress: 80, message: 'Running data cleaning script...' })

    let cleanResult
    try {
      console.log("[Backend] Starting Python script execution...")
      cleanResult = await runPythonScript(cleanScriptPath, [outputPath, ...uploadedFiles])
      console.log("[Backend] Python script finished. Result:", cleanResult)
    } catch (error) {
      console.error("[Backend] Error running cleaner/loader:", error.message)
      jobs.set(jobId, {
        status: 'failed',
        progress: 100,
        error: `Cleaning/Loading failed: ${error.message}`,
        filesProcessed: files.length,
      })
      return
    }

    if (!cleanResult.success) {
      console.log("[Backend] Clean result indicates failure:", cleanResult)
      jobs.set(jobId, {
        status: 'failed',
        progress: 100,
        error: cleanResult.error || "Cleaning or loading failed",
        filesProcessed: files.length,
        cleaningResults: cleanResult,
      })
      return
    }

    // Update progress to show cleaning completed
    jobs.set(jobId, { status: 'processing', progress: 90, message: 'Data cleaning completed, finalizing results...' })

    // Brief pause to show finalization step
    await new Promise(resolve => setTimeout(resolve, 500))

    console.log("[Backend] Setting job to completed")
    jobs.set(jobId, {
      status: 'completed',
      progress: 100,
      message: "Data processing completed successfully! Files have been cleaned and loaded into the database.",
      filesProcessed: files.length,
      cleaningResults: cleanResult,
    })

  } catch (error) {
    console.error("[Backend] Background processing error:", error)
    jobs.set(jobId, {
      status: 'failed',
      progress: 100,
      error: error.message,
    })
  }
}



export default router
