// backend/src/utils/runPythonScript.js
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use python from env if provided, else default to "python"
const PYTHON_CMD = process.env.PYTHON_CMD || "python";

/**
 * Run a Python script located under backend/src.
 *
 * @param {string} relativeScriptPath - path relative to backend/src
 *   e.g. "analytics/Descriptive/descriptive.py"
 *   or   "analytics/prescriptive/prescriptive.py"
 * @param {string[]} args - CLI args to pass to the script
 * @returns {Promise<{ stdout: string, stderr: string, code: number }>}
 */
export default function runPythonScript(relativeScriptPath, args = []) {
  return new Promise((resolve, reject) => {
    try {
      // __dirname is .../backend/src/utils
      // We want to resolve relative to .../backend/src
      const scriptsRoot = path.resolve(__dirname, "..");
      const scriptFullPath = path.resolve(scriptsRoot, relativeScriptPath);

      const child = spawn(PYTHON_CMD, [scriptFullPath, ...args], {
        env: { ...process.env },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("error", (err) => {
        reject(err);
      });

      child.on("close", (code) => {
        if (code !== 0) {
          const err = new Error(
            `Python script exited with code ${code}. stderr:\n${stderr}`
          );
          // Attach for logging if you want
          err.code = code;
          err.stderr = stderr;
          err.stdout = stdout;
          reject(err);
        } else {
          resolve({ stdout, stderr, code });
        }
      });
    } catch (err) {
      reject(err);
    }
  });
}
