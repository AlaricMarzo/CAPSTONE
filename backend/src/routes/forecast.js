import { Router } from "express";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const router = Router();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PY = process.env.PYTHON_BIN || "python"; // or "python3"

function runPython({ model, sku, horizon = 6 }) {
  return new Promise((resolve, reject) => {
    const csvPath = path.join(__dirname, "..", "..", "cleaned", "complete_data.csv");
    const scriptPath = path.join(__dirname, "..", "analytics", "arima_ets.py");

    const args = [
      scriptPath,
      "--csv", csvPath,
      "--model", model,
      "--horizon", String(horizon)
    ];
    if (sku) args.push("--sku", sku);

    const proc = spawn(PY, args, { shell: true });
    let out = "", err = "";
    proc.stdout.on("data", d => (out += d.toString()));
    proc.stderr.on("data", d => (err += d.toString()));
    proc.on("close", code => {
      if (code !== 0) return reject(new Error(err || `python exited ${code}`));
      try { resolve(JSON.parse(out)); }
      catch (e) { reject(new Error("Invalid JSON from python: " + e.message + "\n" + out)); }
    });
  });
}

// GET /api/forecast/arima?sku=Paracetamol&horizon=6
router.get("/arima", async (req, res) => {
  try {
    const { sku, horizon } = req.query;
    const data = await runPython({ model: "arima", sku, horizon: Number(horizon || 6) });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/forecast/ets?sku=Paracetamol&horizon=6
router.get("/ets", async (req, res) => {
  try {
    const { sku, horizon } = req.query;
    const data = await runPython({ model: "ets", sku, horizon: Number(horizon || 6) });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

export default router;
