import { spawn } from "child_process";
import path from "path";

export function runPythonScript(relativeScriptPath, args = []) {
  return new Promise((resolve) => {
    const scriptPath = path.join(__dirname, "..", relativeScriptPath);

    const child = spawn("python", [scriptPath, ...args], {
      cwd: path.join(__dirname, ".."),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("close", (code) => {
      if (code === 0) {
        // If your Python scripts print JSON, parse it here
        try {
          const parsed = stdout.trim() ? JSON.parse(stdout) : {};
          resolve({
            success: true,
            ...parsed,
          });
        } catch {
          console.warn("[Python] Non-JSON stdout:", stdout);
          resolve({
            success: true,
            rawOutput: stdout,
          });
        }
      } else {
        console.error("[Python] script failed", {
          exitCode: code,
          stderr,
          stdout,
        });
        resolve({
          success: false,
          error: stderr || `Python script exited with code ${code}`,
        });
      }
    });

    child.on("error", (err) => {
      console.error("[Python] spawn error", err);
      resolve({
        success: false,
        error: err.message || String(err),
      });
    });
  });
}
