import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { spawn } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const router = express.Router();

// Path to prescriptive output directory
const prescriptiveOutputDir = path.join(__dirname, "../analytics/prescriptive/prescriptive_output");

// Helper function to read CSV and convert to JSON
function csvToJson(csvPath) {
  if (!fs.existsSync(csvPath)) {
    return [];
  }

  const csvData = fs.readFileSync(csvPath, "utf8");
  const lines = csvData.split("\n").filter(line => line.trim() !== "");
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map(h => h.trim());
  const rows = lines.slice(1).map(line => {
    const values = line.split(",").map(v => v.trim());
    const obj = {};
    headers.forEach((header, index) => {
      const value = values[index] || "";
      // Try to parse numbers
      if (!isNaN(value) && value !== "") {
        obj[header] = parseFloat(value);
      } else {
        obj[header] = value;
      }
    });
    return obj;
  });

  return rows;
}

// Route to get all prescriptive analytics data
router.get("/prescriptive", async (req, res) => {
  try {
    const data = {};

    // Model 1: Reorder Point
    data.reorderPoint = csvToJson(path.join(prescriptiveOutputDir, "model_1_reorder_point.csv"));

    // Model 2: EOQ
    data.eoq = csvToJson(path.join(prescriptiveOutputDir, "model_2_eoq.csv"));

    // Model 3: Inventory Allocation
    data.inventoryAllocation = csvToJson(path.join(prescriptiveOutputDir, "model_3_inventory_allocation.csv"));

    // Model 4: What-If Analysis
    data.whatIfAnalysis = csvToJson(path.join(prescriptiveOutputDir, "model_4_whatif_analysis.csv"));

    // Model 5: Discount Optimization (Group)
    data.discountByGroup = csvToJson(path.join(prescriptiveOutputDir, "model_5_discount_by_group.csv"));

    // Model 5: Discount Optimization (Product)
    data.discountByProduct = csvToJson(path.join(prescriptiveOutputDir, "model_5_discount_by_product.csv"));

    // Model 6: Resource Planning
    data.resourcePlanning = csvToJson(path.join(prescriptiveOutputDir, "model_6_resource_planning.csv"));

    // Model 7: Anomaly Detection
    data.anomalyDetection = csvToJson(path.join(prescriptiveOutputDir, "model_7_anomaly_detection.csv"));

    // Model 7: Anomalies Only
    data.anomaliesOnly = csvToJson(path.join(prescriptiveOutputDir, "model_7_anomalies_only.csv"));

    // Summary
    data.summary = csvToJson(path.join(prescriptiveOutputDir, "SUMMARY_all_models.csv"));

    // Read recommendations text file
    const recommendationsPath = path.join(prescriptiveOutputDir, "model_8_prescriptive_recommendations.txt");
    if (fs.existsSync(recommendationsPath)) {
      data.recommendations = fs.readFileSync(recommendationsPath, "utf8");
    } else {
      data.recommendations = "Recommendations file not found.";
    }

    res.json({
      success: true,
      data: data
    });

  } catch (error) {
    console.error("Error fetching prescriptive analytics:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch prescriptive analytics data"
    });
  }
});

// Route to run prescriptive analytics
router.post("/run-prescriptive", async (req, res) => {
  try {
    const prescriptiveDir = path.join(__dirname, "../analytics/prescriptive");

    // Check if prescriptive.py exists
    const scriptPath = path.join(prescriptiveDir, "prescriptive.py");
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({
        success: false,
        error: "Prescriptive analytics script not found"
      });
    }

    // Run the Python script
    const pythonProcess = spawn("python", ["prescriptive.py"], {
      cwd: prescriptiveDir,
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        res.json({
          success: true,
          message: "Prescriptive analytics completed successfully",
          output: stdout
        });
      } else {
        console.error("Python script error:", stderr);
        res.status(500).json({
          success: false,
          error: "Prescriptive analytics failed",
          details: stderr
        });
      }
    });

    pythonProcess.on("error", (error) => {
      console.error("Failed to start Python process:", error);
      res.status(500).json({
        success: false,
        error: "Failed to execute prescriptive analytics",
        details: error.message
      });
    });

  } catch (error) {
    console.error("Error running prescriptive analytics:", error);
    res.status(500).json({
      success: false,
      error: "Failed to run prescriptive analytics"
    });
  }
});

export default router;
