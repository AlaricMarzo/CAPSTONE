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

// Path to descriptive output directory
const descriptiveOutputDir = path.join(__dirname, "../analytics/Descriptive/descriptive_output");

// Helper function to read CSV and convert to JSON
function csvToJson(csvPath) {
  if (!fs.existsSync(csvPath)) {
    return [];
  }

  const csvData = fs.readFileSync(csvPath, "utf8");
  const lines = csvData.split("\n").filter((line) => line.trim() !== "");
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const values = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((header, index) => {
      const value = values[index] || "";
      // Try to parse numbers
      if (!isNaN(value) && value !== "") {
        obj[header] = Number.parseFloat(value);
      } else {
        obj[header] = value;
      }
    });
    return obj;
  });

  return rows;
}

// Helper functions to generate summary JSONs for each analytics type
function generatePrescriptiveSummary(data) {
  return {
    totalModels: 7,
    modelCounts: {
      reorderPoint: data.reorderPoint.length,
      eoq: data.eoq.length,
      inventoryAllocation: data.inventoryAllocation.length,
      whatIfAnalysis: data.whatIfAnalysis.length,
      discountByGroup: data.discountByGroup.length,
      discountByProduct: data.discountByProduct.length,
      resourcePlanning: data.resourcePlanning.length,
      anomalyDetection: data.anomalyDetection.length,
      anomaliesOnly: data.anomaliesOnly.length,
    },
    summaryRecordsCount: data.summary.length,
    hasRecommendations: !!data.recommendations && data.recommendations !== "Recommendations file not found.",
  };
}

function generateDescriptiveSummary(data) {
  return {
    kpiCount: data.kpis.length,
    associationRulesCount: data.mbaRules.length,
    clusterCount: data.clustering.length,
    totalRecords: (data.kpis.length || 0) + (data.mbaRules.length || 0) + (data.clustering.length || 0),
  };
}

function generatePredictiveSummary(data) {
  return {
    models: {
      randomForest: {
        forecastsCount: data.randomForest.forecasts.length,
        metricsCount: data.randomForest.metrics.length,
        featuresCount: data.randomForest.features.length,
      },
      xgboost: {
        forecastsCount: data.xgboost.forecasts.length,
        metricsCount: data.xgboost.metrics.length,
        featuresCount: data.xgboost.features.length,
      },
      sarima: {
        forecastsCount: data.sarima.forecasts.length,
        metricsCount: data.sarima.metrics.length,
      },
    },
    totalForecasts:
      (data.randomForest.forecasts.length || 0) +
      (data.xgboost.forecasts.length || 0) +
      (data.sarima.forecasts.length || 0),
  };
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

    const summary = generatePrescriptiveSummary(data);

    res.json({
      success: true,
      summary: summary,
      data: data,
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

// Path to predictive output directories
const predictiveOutputDirs = {
  randomForest: path.join(__dirname, "../analytics/Predictive/ml_random_forest"),
  xgboost: path.join(__dirname, "../analytics/Predictive/ml_xgboost_model"),
  sarima: path.join(__dirname, "../analytics/Predictive/ts_sarima-ets-sarimax(2,1,2)")
};

// Route to get all descriptive analytics data
router.get("/descriptive", async (req, res) => {
  try {
    const data = {};

    // KPIs
    data.kpis = csvToJson(path.join(descriptiveOutputDir, "kpi_output", "kpi_summary.csv"));

    // MBA Rules
    data.mbaRules = csvToJson(path.join(descriptiveOutputDir, "mba_output", "association_rules.csv"));

    // Clustering
    data.clustering = csvToJson(path.join(descriptiveOutputDir, "clustering_output", "clustering_results.csv"));

    const summary = generateDescriptiveSummary(data);

    res.json({
      success: true,
      summary: summary,
      data: data,
    });

  } catch (error) {
    console.error("Error fetching descriptive analytics:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch descriptive analytics data"
    });
  }
});

// Route to get all predictive analytics data
router.get("/predictive", async (req, res) => {
  try {
    const data = {};

    // Random Forest
    data.randomForest = {
      forecasts: csvToJson(path.join(predictiveOutputDirs.randomForest, "forecasts.csv")),
      metrics: csvToJson(path.join(predictiveOutputDirs.randomForest, "metrics.csv")),
      features: csvToJson(path.join(predictiveOutputDirs.randomForest, "feature_importance.csv"))
    };

    // XGBoost
    data.xgboost = {
      forecasts: csvToJson(path.join(predictiveOutputDirs.xgboost, "forecasts.csv")),
      metrics: csvToJson(path.join(predictiveOutputDirs.xgboost, "metrics.csv")),
      features: csvToJson(path.join(predictiveOutputDirs.xgboost, "feature_importance.csv"))
    };

    // SARIMA/ETS
    data.sarima = {
      forecasts: csvToJson(path.join(predictiveOutputDirs.sarima, "forecasts.csv")),
      metrics: csvToJson(path.join(predictiveOutputDirs.sarima, "metrics.csv"))
    };

    const summary = generatePredictiveSummary(data);

    res.json({
      success: true,
      summary: summary,
      data: data,
    });

  } catch (error) {
    console.error("Error fetching predictive analytics:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch predictive analytics data"
    });
  }
});

// Route to run descriptive analytics
router.post("/run-descriptive", async (req, res) => {
  try {
    const descriptiveDir = path.join(__dirname, "../analytics/Descriptive");

    // Check if descriptive.py exists
    const scriptPath = path.join(descriptiveDir, "descriptive.py");
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({
        success: false,
        error: "Descriptive analytics script not found"
      });
    }

    // Run the Python script
    const pythonProcess = spawn("python", ["descriptive.py"], {
      cwd: descriptiveDir,
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
          message: "Descriptive analytics completed successfully",
          output: stdout
        });
      } else {
        console.error("Python script error:", stderr);
        res.status(500).json({
          success: false,
          error: "Descriptive analytics failed",
          details: stderr
        });
      }
    });

    pythonProcess.on("error", (error) => {
      console.error("Failed to start Python process:", error);
      res.status(500).json({
        success: false,
        error: "Failed to execute descriptive analytics",
        details: error.message
      });
    });

  } catch (error) {
    console.error("Error running descriptive analytics:", error);
    res.status(500).json({
      success: false,
      error: "Failed to run descriptive analytics"
    });
  }
});

router.get("/descriptive", (req, res) => {
  try {
    const summaryPath = path.join(descriptiveOutputDir, "summary.json");
    if (!fs.existsSync(summaryPath)) {
      return res.status(404).json({
        success: false,
        error: "Descriptive analytics data not found. Please run the analysis first."
      });
    }

    const summaryData = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
    res.json({
      success: true,
      data: summaryData
    });
  } catch (error) {
    console.error("Error reading descriptive analytics data:", error);
    res.status(500).json({
      success: false,
      error: "Failed to load descriptive analytics data",
      details: error.message
    });
  }
});

router.get("/prescriptive", (req, res) => {
  try {
    const summaryPath = path.join(prescriptiveOutputDir, "summary.json");
    if (!fs.existsSync(summaryPath)) {
      return res.status(404).json({
        success: false,
        error: "Prescriptive analytics data not found. Please run the analysis first."
      });
    }

    const summaryData = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
    res.json({
      success: true,
      data: summaryData
    });
  } catch (error) {
    console.error("Error reading prescriptive analytics data:", error);
    res.status(500).json({
      success: false,
      error: "Failed to load prescriptive analytics data",
      details: error.message
    });
  }
});

router.get("/predictive", (req, res) => {
  try {
    const rfDir = path.join(__dirname, "../analytics/Predictive/ml_random_forest");
    const xgbDir = path.join(__dirname, "../analytics/Predictive/ml_xgboost_model");

    let rfSummary = null;
    let xgbSummary = null;

    if (fs.existsSync(path.join(rfDir, "summary.json"))) {
      rfSummary = JSON.parse(fs.readFileSync(path.join(rfDir, "summary.json"), "utf8"));
    }

    if (fs.existsSync(path.join(xgbDir, "summary.json"))) {
      xgbSummary = JSON.parse(fs.readFileSync(path.join(xgbDir, "summary.json"), "utf8"));
    }

    if (!rfSummary && !xgbSummary) {
      return res.status(404).json({
        success: false,
        error: "Predictive analytics data not found. Please run the analysis first."
      });
    }

    res.json({
      success: true,
      data: {
        randomForest: rfSummary,
        xgboost: xgbSummary
      }
    });
  } catch (error) {
    console.error("Error reading predictive analytics data:", error);
    res.status(500).json({
      success: false,
      error: "Failed to load predictive analytics data",
      details: error.message
    });
  }
});

// Route to run predictive analytics
router.post("/run-predictive", async (req, res) => {
  try {
    const predictiveDir = path.join(__dirname, "../analytics/Predictive");

    // Check if models.py exists in parent directory
    const scriptPath = path.join(__dirname, "../analytics/models.py");
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({
        success: false,
        error: "Predictive analytics script not found"
      });
    }

    // Run the Python script
    const pythonProcess = spawn("python", [scriptPath], {
      cwd: path.join(__dirname, "../analytics"),
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
          message: "Predictive analytics completed successfully",
          output: stdout
        });
      } else {
        console.error("Python script error:", stderr);
        res.status(500).json({
          success: false,
          error: "Predictive analytics failed",
          details: stderr
        });
      }
    });

    pythonProcess.on("error", (error) => {
      console.error("Failed to start Python process:", error);
      res.status(500).json({
        success: false,
        error: "Failed to execute predictive analytics",
        details: error.message
      });
    });

  } catch (error) {
    console.error("Error running predictive analytics:", error);
    res.status(500).json({
      success: false,
      error: "Failed to run predictive analytics"
    });
  }
});

export default router;
