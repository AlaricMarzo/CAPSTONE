import express from "express"
import multer from "multer"
import path from "path"
import fs from "fs"
import { spawn } from "child_process"
import { fileURLToPath } from "url"
import { dirname } from "path"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const router = express.Router()

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
    const python = spawn(pythonCmd, [scriptPath, ...args]);

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
    }, 5 * 60 * 1000); // 5 minutes
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



// -------------------- Upload Route --------------------
router.post("/upload", upload.array("files", 20), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({
        success: false,
        error: "No files uploaded",
      });
    }

    console.log(`[Backend] Received ${req.files.length} files`);

    const cleanedDir = path.join(__dirname, "../../cleaned");
    if (!fs.existsSync(cleanedDir)) {
      fs.mkdirSync(cleanedDir, { recursive: true });
    }

    const uploadedFiles = req.files.map((file) => file.path); // Collect file paths

    const timestamp = Date.now();
    const outputPath = path.join(cleanedDir, `combined_cleaned_${timestamp}.csv`); // Output path

    const cleanScriptPath = path.join(__dirname, "../../scripts/clean_data.py");

    console.log("[Backend] Starting data cleaning and loading...");

    let cleanResult;
    try {
      cleanResult = await runPythonScript(cleanScriptPath, [outputPath, ...uploadedFiles]); // Pass all files to Python script
      console.log("[Backend] Cleaning and loading complete:", cleanResult);
    } catch (error) {
      console.error("[Backend] Error running cleaner/loader:", error.message);
      return res.status(500).json({
        success: false,
        error: `Cleaning/Loading failed: ${error.message}`,
        filesProcessed: req.files.length,
      });
    }

    if (!cleanResult.success) {
      return res.json({
        success: false,
        error: cleanResult.error || "Cleaning or loading failed",
        filesProcessed: req.files.length,
        cleaningResults: cleanResult,
      });
    }

    res.json({
      success: true,
      message: "Files processed and loaded successfully",
      filesProcessed: req.files.length,
      cleaningResults: cleanResult,
    });
  } catch (error) {
    console.error("[Backend] Upload error:", error);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});


export default router
