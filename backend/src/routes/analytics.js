import express from "express"
import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"
import { dirname } from "path"
import { spawn } from "child_process"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const router = express.Router()

const prescriptiveOutputDir = path.join(__dirname, "../analytics/prescriptive/prescriptive_output")
const descriptiveOutputDir = path.join(__dirname, "../analytics/Descriptive/descriptive_output")
const predictiveOutputDir = path.join(__dirname, "../analytics/Predictive")

// Helper function to read CSV and convert to JSON
function csvToJson(csvPath) {
  if (!fs.existsSync(csvPath)) {
    console.log("[v0] CSV file not found:", csvPath)
    return []
  }

  try {
    const csvData = fs.readFileSync(csvPath, "utf8")
    const lines = csvData.split("\n").filter((line) => line.trim() !== "")
    if (lines.length < 2) return []

    const headers = lines[0].split(",").map((h) => h.trim())
    const rows = lines.slice(1).map((line) => {
      const values = line.split(",").map((v) => v.trim())
      const obj = {}
      headers.forEach((header, index) => {
        const value = values[index] || ""
        if (!isNaN(value) && value !== "") {
          obj[header] = Number.parseFloat(value)
        } else {
          obj[header] = value
        }
      })
      return obj
    })

    return rows
  } catch (error) {
    console.error("[v0] Error parsing CSV:", error)
    return []
  }
}

// Helper function to read JSON
function jsonToArray(jsonPath) {
  if (!fs.existsSync(jsonPath)) {
    console.log("[v0] JSON file not found:", jsonPath)
    return []
  }
  try {
    const jsonData = fs.readFileSync(jsonPath, "utf8")
    return JSON.parse(jsonData)
  } catch (error) {
    console.error("[v0] Error parsing JSON:", error)
    return []
  }
}

router.get("/descriptive", async (req, res) => {
  try {
    const kpiDir = path.join(descriptiveOutputDir, "kpi_output")
    const mbaDir = path.join(descriptiveOutputDir, "mba_output")
    const clusterDir = path.join(descriptiveOutputDir, "clustering_output")

    const monthlySalesData = jsonToArray(path.join(kpiDir, "kpi_monthly_sales_qty.json"))
    const growthData = jsonToArray(path.join(kpiDir, "kpi_monthly_sales_growth_rate.json"))
    const categoryData = jsonToArray(path.join(kpiDir, "kpi_category_split_monthly.json"))
    const top10Sales = jsonToArray(path.join(kpiDir, "kpi_top10_by_sales.json"))
    const top10Qty = jsonToArray(path.join(kpiDir, "kpi_top10_by_qty.json"))
    const activeSkus = jsonToArray(path.join(kpiDir, "kpi_active_skus_monthly.json"))

    const totalSales = monthlySalesData.reduce((sum, d) => sum + (d.total_sales || 0), 0)
    const totalQty = monthlySalesData.reduce((sum, d) => sum + (d.total_qty || 0), 0)
    const latestGrowth = growthData.length > 0 ? growthData[growthData.length - 1].growth_rate : 0
    const latestSkus = activeSkus.length > 0 ? activeSkus[activeSkus.length - 1].active_skus : 0

    const monthlyFormattedData = monthlySalesData.map((d) => ({
      month: d.month || "",
      sales: d.total_sales || 0,
      quantity: d.total_qty || 0,
    }))

    const categoryLatest = categoryData.length > 0 ? categoryData[categoryData.length - 1] : {}
    const categoryDistribution = Object.entries(categoryLatest)
      .filter(([key]) => key !== "month")
      .map(([name, value]) => ({ name, value: Number.parseFloat(value) || 0 }))

    const topProducts = (top10Sales || []).slice(0, 10).map((d) => ({
      name: d.sku_description || d.name || "Product",
      sales: d.total_sales || 0,
      quantity: d.qty || 0,
    }))

    const clusteringData = jsonToArray(path.join(clusterDir, "cluster_summaries", "global.json"))
    const clusteringSummary = (clusteringData || []).map((d, idx) => ({
      cluster: idx,
      count: d.customer_count || d.count || 0,
      avg_value: d.avg_value || 0,
    }))

    const mbaRulesData = jsonToArray(path.join(mbaDir, "mba_rules.json"))
    const mbaRules = (mbaRulesData || []).slice(0, 10).map((d) => ({
      item_a: d.antecedents || d.item_a || "",
      item_b: d.consequents || d.item_b || "",
      lift: d.lift || 0,
      confidence_ab_pct: (d.confidence || d.confidence_ab_pct) * 100 || 0,
      support_pct: (d.support || d.support_pct) * 100 || 0,
    }))

    const formattedData = {
      kpi_summary: {
        total_sales: totalSales,
        total_quantity: totalQty,
        active_skus: latestSkus,
        growth_rate: latestGrowth,
      },
      monthly_sales: monthlyFormattedData,
      category_distribution: categoryDistribution,
      top_products: topProducts,
      clustering_summary: clusteringSummary,
      mba_rules: mbaRules,
    }

    res.json({ success: true, data: formattedData })
  } catch (error) {
    console.error("[v0] Error fetching descriptive analytics:", error)
    res.status(500).json({ success: false, error: error.message })
  }
})

router.get("/predictive", async (req, res) => {
  try {
    const sarimaDirPath = path.join(predictiveOutputDir, "ts_sarima-ets-sarimax(2,1,2)")

    const forecastData = csvToJson(path.join(sarimaDirPath, "forecasts.csv"))
    const metricsData = csvToJson(path.join(sarimaDirPath, "metrics.csv"))

    const formattedForecasts = (forecastData || []).map((d) => ({
      date: d.date || "",
      actual: Number.parseFloat(d.actual) || 0,
      rf_predicted: 0,
      xgb_predicted: 0,
      sarima_predicted: Number.parseFloat(d.predicted) || 0,
      confidence_lower: Number.parseFloat(d.lower_bound) || 0,
      confidence_upper: Number.parseFloat(d.upper_bound) || 0,
    }))

    const metrics = (metricsData || [])[0] || {}
    const modelPerformance = {
      random_forest: {
        mae: 0,
        rmse: 0,
        r_squared: 0,
      },
      xgboost: {
        mae: 0,
        rmse: 0,
        r_squared: 0,
      },
      sarima: {
        mae: Number.parseFloat(metrics.mae) || 0,
        rmse: Number.parseFloat(metrics.rmse) || 0,
        r_squared: 0,
      },
    }

    const formattedData = {
      models_summary: {
        total_models: 3,
        avg_accuracy:
          (modelPerformance.random_forest.r_squared +
            modelPerformance.xgboost.r_squared +
            modelPerformance.sarima.r_squared) /
          3,
        total_forecasts: formattedForecasts.length,
      },
      forecast_data: formattedForecasts,
      model_performance: modelPerformance,
    }

    res.json({ success: true, data: formattedData })
  } catch (error) {
    console.error("[v0] Error fetching predictive analytics:", error)
    res.status(500).json({ success: false, error: error.message })
  }
})

router.get("/prescriptive", async (req, res) => {
  try {
    const reorderData = csvToJson(path.join(prescriptiveOutputDir, "model_1_reorder_point.csv"))
    const eoqData = csvToJson(path.join(prescriptiveOutputDir, "model_2_eoq.csv"))
    const allocationData = csvToJson(path.join(prescriptiveOutputDir, "model_3_inventory_allocation.csv"))
    const whatIfData = csvToJson(path.join(prescriptiveOutputDir, "model_4_whatif_analysis.csv"))
    const discountData = csvToJson(path.join(prescriptiveOutputDir, "model_5_discount_by_product.csv"))
    const resourceData = csvToJson(path.join(prescriptiveOutputDir, "model_6_resource_planning.csv"))
    const anomalyData = csvToJson(path.join(prescriptiveOutputDir, "model_7_anomaly_detection.csv"))
    const summaryData = csvToJson(path.join(prescriptiveOutputDir, "SUMMARY_all_models.csv"))

    const reorder_points = (reorderData || []).map((d) => ({
      medicine: d.medicine || d.sku_description || d.name || d.product || "Product",
      avg_daily_demand: Number.parseFloat(d.avg_daily_demand) || Number.parseFloat(d.daily_demand) || 0,
      safety_stock: Number.parseFloat(d.safety_stock) || 0,
      reorder_point: Number.parseFloat(d.reorder_point) || 0,
      forecast_30day: Number.parseFloat(d.forecast_30day) || Number.parseFloat(d.forecast) || 0,
    }))

    const eoq_data = (eoqData || []).map((d) => ({
      medicine: d.medicine || d.sku_description || d.name || d.product || "Product",
      annual_demand: Number.parseFloat(d.annual_demand) || 0,
      eoq: Number.parseFloat(d.eoq) || 0,
      orders_per_year: Number.parseFloat(d.orders_per_year) || 0,
      days_between_orders:
        Number.parseFloat(d.days_between_orders) || 365 / (Number.parseFloat(d.orders_per_year) || 1),
      total_annual_cost: Number.parseFloat(d.total_annual_cost) || 0,
    }))

    const allocations = (allocationData || []).map((d) => ({
      medicine: d.medicine || d.sku_description || d.name || d.product || "Product",
      optimal_allocation: Number.parseFloat(d.optimal_allocation) || Number.parseFloat(d.units_allocated) || 0,
      allocated_value: Number.parseFloat(d.allocated_value) || Number.parseFloat(d.value) || 0,
      profit_margin: Number.parseFloat(d.profit_margin) || Number.parseFloat(d.margin_pct) || 0,
      expected_profit: Number.parseFloat(d.expected_profit) || Number.parseFloat(d.profit) || 0,
    }))

    const discountGroupMap = new Map()
    ;(discountData || []).forEach((d) => {
      const group = d.customer_group || d.group || "Default"
      if (!discountGroupMap.has(group)) {
        discountGroupMap.set(group, {
          customer_group: group,
          qty: 0,
          sales: 0,
          profit: 0,
          avg_discount_pct: 0,
          profit_margin: 0,
        })
      }
      const current = discountGroupMap.get(group)
      current.qty += Number.parseFloat(d.qty) || 0
      current.sales += Number.parseFloat(d.sales) || 0
      current.profit += Number.parseFloat(d.profit) || 0
      current.avg_discount_pct = Number.parseFloat(d.discount_pct) || current.avg_discount_pct
      current.profit_margin = Number.parseFloat(d.profit_margin) || current.profit_margin
    })
    const discount_groups = Array.from(discountGroupMap.values())

    const resource_planning = (resourceData || []).map((d) => ({
      medicine: d.medicine || d.sku_description || d.name || d.product || "Product",
      daily_demand: Number.parseFloat(d.daily_demand) || Number.parseFloat(d.demand) || 0,
      projected_demand: Number.parseFloat(d.projected_demand) || Number.parseFloat(d.forecast) || 0,
      storage_needed: Number.parseFloat(d.storage_needed) || Number.parseFloat(d.storage) || 0,
      capital_needed: Number.parseFloat(d.capital_needed) || Number.parseFloat(d.capital) || 0,
    }))

    const anomalies = (anomalyData || []).map((d) => ({
      date: d.date || new Date().toISOString().split("T")[0],
      sales: Number.parseFloat(d.sales) || 0,
      qty: Number.parseFloat(d.qty) || 0,
      profit: Number.parseFloat(d.profit) || 0,
      anomaly_score: Number.parseFloat(d.anomaly_score) || 0,
    }))

    const total_sales = discount_groups.reduce((sum, g) => sum + g.sales, 0)
    const total_profit = discount_groups.reduce((sum, g) => sum + g.profit, 0)
    const total_cost = total_sales - total_profit
    const overall_profit_margin_pct = total_sales > 0 ? (total_profit / total_sales) * 100 : 0
    const total_quantity_sold = discount_groups.reduce((sum, g) => sum + g.qty, 0)

    const financial_summary = {
      total_sales,
      total_cost,
      total_profit,
      overall_profit_margin_pct,
      total_quantity_sold,
    }

    const key_metrics = {
      total_products_optimized: reorder_points.length,
      total_models: 7,
      total_cost_savings: summaryData.reduce((sum, d) => sum + (Number.parseFloat(d.total_savings) || 0), 0),
    }

    const formattedData = {
      reorder_points,
      eoq_data,
      allocations,
      discount_groups,
      resource_planning,
      anomalies,
      financial_summary,
      key_metrics,
    }

    console.log("[v0] Prescriptive data formatted successfully")
    res.json({ success: true, data: formattedData })
  } catch (error) {
    console.error("[v0] Error fetching prescriptive analytics:", error)
    res.status(500).json({ success: false, error: error.message })
  }
})

// Route to run descriptive analytics
router.post("/run-descriptive", async (req, res) => {
  try {
    const descriptiveDir = path.join(__dirname, "../analytics/Descriptive")
    const scriptPath = path.join(descriptiveDir, "descriptive.py")
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({ success: false, error: "Descriptive analytics script not found" })
    }

    const pythonProcess = spawn("python", ["descriptive.py"], {
      cwd: descriptiveDir,
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

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        res.json({ success: true, message: "Descriptive analytics completed successfully", output: stdout })
      } else {
        console.error("[v0] Python script error:", stderr)
        res.status(500).json({ success: false, error: "Descriptive analytics failed", details: stderr })
      }
    })

    pythonProcess.on("error", (error) => {
      console.error("[v0] Failed to start Python process:", error)
      res.status(500).json({ success: false, error: "Failed to execute descriptive analytics", details: error.message })
    })
  } catch (error) {
    console.error("[v0] Error running descriptive analytics:", error)
    res.status(500).json({ success: false, error: "Failed to run descriptive analytics" })
  }
})

// Route to run predictive analytics
router.post("/run-predictive", async (req, res) => {
  try {
    const scriptPath = path.join(__dirname, "../analytics/models.py")
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({ success: false, error: "Predictive analytics script not found" })
    }

    const pythonProcess = spawn("python", [scriptPath], {
      cwd: path.join(__dirname, "../analytics"),
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

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        res.json({ success: true, message: "Predictive analytics completed successfully", output: stdout })
      } else {
        console.error("[v0] Python script error:", stderr)
        res.status(500).json({ success: false, error: "Predictive analytics failed", details: stderr })
      }
    })

    pythonProcess.on("error", (error) => {
      console.error("[v0] Failed to start Python process:", error)
      res.status(500).json({ success: false, error: "Failed to execute predictive analytics", details: error.message })
    })
  } catch (error) {
    console.error("[v0] Error running predictive analytics:", error)
    res.status(500).json({ success: false, error: "Failed to run predictive analytics" })
  }
})

// Route to run prescriptive analytics
router.post("/run-prescriptive", async (req, res) => {
  try {
    const prescriptiveDir = path.join(__dirname, "../analytics/prescriptive")
    const scriptPath = path.join(prescriptiveDir, "prescriptive.py")
    if (!fs.existsSync(scriptPath)) {
      return res.status(404).json({ success: false, error: "Prescriptive analytics script not found" })
    }

    const pythonProcess = spawn("python", ["prescriptive.py"], {
      cwd: prescriptiveDir,
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

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        res.json({ success: true, message: "Prescriptive analytics completed successfully", output: stdout })
      } else {
        console.error("[v0] Python script error:", stderr)
        res.status(500).json({ success: false, error: "Prescriptive analytics failed", details: stderr })
      }
    })

    pythonProcess.on("error", (error) => {
      console.error("[v0] Failed to start Python process:", error)
      res
        .status(500)
        .json({ success: false, error: "Failed to execute prescriptive analytics", details: error.message })
    })
  } catch (error) {
    console.error("[v0] Error running prescriptive analytics:", error)
    res.status(500).json({ success: false, error: "Failed to run prescriptive analytics" })
  }
})

// Route to list available files for download
router.get("/files", async (req, res) => {
  try {
    const files = {}

    // Descriptive analytics files
    const descriptiveDir = path.join(descriptiveOutputDir, "kpi_output")
    if (fs.existsSync(descriptiveDir)) {
      const descriptiveFiles = fs.readdirSync(descriptiveDir)
        .filter(file => file.endsWith('.csv') || file.endsWith('.png'))
        .map(file => ({
          name: file,
          path: path.join(descriptiveDir, file),
          type: file.endsWith('.csv') ? 'csv' : 'png',
          category: 'descriptive'
        }))
      files.descriptive = descriptiveFiles
    }

    // Prescriptive analytics files
    if (fs.existsSync(prescriptiveOutputDir)) {
      const prescriptiveFiles = fs.readdirSync(prescriptiveOutputDir)
        .filter(file => file.endsWith('.csv') || file.endsWith('.png'))
        .map(file => ({
          name: file,
          path: path.join(prescriptiveOutputDir, file),
          type: file.endsWith('.csv') ? 'csv' : 'png',
          category: 'prescriptive'
        }))
      files.prescriptive = prescriptiveFiles
    }

    // Predictive analytics files
    const sarimaDir = path.join(predictiveOutputDir, "ts_sarima-ets-sarimax(2,1,2)")
    if (fs.existsSync(sarimaDir)) {
      const predictiveFiles = fs.readdirSync(sarimaDir)
        .filter(file => file.endsWith('.csv') || file.endsWith('.png'))
        .map(file => ({
          name: file,
          path: path.join(sarimaDir, file),
          type: file.endsWith('.csv') ? 'csv' : 'png',
          category: 'predictive'
        }))
      files.predictive = predictiveFiles
    }

    res.json({ success: true, files })
  } catch (error) {
    console.error("[v0] Error listing files:", error)
    res.status(500).json({ success: false, error: error.message })
  }
})

// Route to download specific files
router.get("/download/:category/:filename", async (req, res) => {
  try {
    const { category, filename } = req.params
    let filePath = ""

    switch (category) {
      case "descriptive":
        filePath = path.join(descriptiveOutputDir, "kpi_output", filename)
        break
      case "prescriptive":
        filePath = path.join(prescriptiveOutputDir, filename)
        break
      case "predictive":
        filePath = path.join(predictiveOutputDir, "ts_sarima-ets-sarimax(2,1,2)", filename)
        break
      default:
        return res.status(400).json({ success: false, error: "Invalid category" })
    }

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ success: false, error: "File not found" })
    }

    const fileExtension = path.extname(filename).toLowerCase()
    const mimeType = fileExtension === '.csv' ? 'text/csv' : 'image/png'

    res.setHeader('Content-Type', mimeType)
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`)

    const fileStream = fs.createReadStream(filePath)
    fileStream.pipe(res)

    fileStream.on('error', (error) => {
      console.error("[v0] Error streaming file:", error)
      res.status(500).json({ success: false, error: "Error downloading file" })
    })
  } catch (error) {
    console.error("[v0] Error downloading file:", error)
    res.status(500).json({ success: false, error: error.message })
  }
})

export default router
