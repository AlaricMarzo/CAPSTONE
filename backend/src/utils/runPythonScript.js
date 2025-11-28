// backend/src/utils/runPythonScript.js
import { spawn } from "child_process";
import path from "path";

export default function runPythonScript(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    const absScriptPath = path.resolve(scriptPath);

    console.log("[runPythonScript] Running python:", absScriptPath, "args:", args);

    // adjust "python" -> "python3" if your container uses that name
    const child = spawn("python", [absScriptPath, ...args], {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => {
      const text = data.toString();
      stdout += text;
      console.log("[PYTHON STDOUT]", text.trim());
    });

    child.stderr.on("data", (data) => {
      const text = data.toString();
      stderr += text;
      console.error("[PYTHON STDERR]", text.trim());
    });

    child.on("error", (err) => {
      console.error("[runPythonScript] Failed to start python:", err);
      reject(
        new Error(
          `Failed to start Python process for ${absScriptPath}: ${err.message}`
        )
      );
    });

    child.on("exit", (code, signal) => {
      console.log("[runPythonScript] exit:", { code, signal });

      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(
          new Error(
            `Python script exited abnormally. code=${code}, signal=${signal}\n` +
            `STDERR:\n${stderr || "(empty)"}\n` +
            `STDOUT:\n${stdout || "(empty)"}`
          )
        );
      }
    });
  });
}
