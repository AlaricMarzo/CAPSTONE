import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import morgan from "morgan";

dotenv.config();
const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

app.get("/health", (_, res) => res.json({ status: "OK" }));

app.post("/api/clean", upload.single("file"), async (req, res) => {
  try {
    res.json({ message: "File received", file: req.file.originalname });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`✅ Server running on http://localhost:${PORT}`));
