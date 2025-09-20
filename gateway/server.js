import express from "express";
import multer from "multer";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import FormData from "form-data";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const rawUploadDir = process.env.UPLOAD_DIR || "uploads";
const uploadDir = path.isAbsolute(rawUploadDir)
  ? rawUploadDir
  : path.join(process.cwd(), rawUploadDir);

fs.mkdirSync(uploadDir, { recursive: true });

const upload = multer({ dest: uploadDir });

const safeUnlink = async (filePath) => {
  if (!filePath) {
    return;
  }

  try {
    await fs.promises.unlink(filePath);
  } catch (error) {
    if (error && error.code !== "ENOENT") {
      console.error(`Failed to remove temporary file ${filePath}`, error);
    }
  }
};

const joinUrl = (base, endpoint) => {
  const normalizedBase = base.replace(/\/+$/, "");
  const normalizedEndpoint = endpoint
    ? `/${endpoint.replace(/^\/+/, "")}`
    : "";
  return `${normalizedBase}${normalizedEndpoint}`;
};

const getPythonUrl = (endpoint) => {
  const base = process.env.PYTHON_URL;
  if (!base) {
    throw new Error("Python service URL is not configured");
  }

  return joinUrl(base, endpoint);
};

const getIngestionUrl = () => {
  if (process.env.INGESTION_URL) {
    return process.env.INGESTION_URL;
  }

  const base = process.env.PYTHON_URL;
  if (!base) {
    throw new Error("Ingestion service URL is not configured");
  }

  return joinUrl(base, "/ingest");
};

const handleRequestError = (res, error, defaultMessage) => {
  let status = 500;
  let message = defaultMessage;

  if (axios.isAxiosError(error)) {
    status = error.response?.status ?? 502;
    const data = error.response?.data;
    if (data) {
      if (typeof data === "string") {
        message = data;
      } else if (typeof data === "object") {
        if (typeof data.error === "string") {
          message = data.error;
        } else if (typeof data.message === "string") {
          message = data.message;
        } else if (typeof data.detail === "string") {
          message = data.detail;
        }
      }
    }
  }

  console.error(defaultMessage, error);
  res.status(status).json({ error: message });
};

app.post("/api/recognize", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: "Image file is required" });
      return;
    }

    const imageName = req.file.originalname || req.file.filename;
    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path), imageName);
    const response = await axios.post(getPythonUrl("/recognize"), form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });
    res.status(response.status ?? 200).json(response.data);
  } catch (err) {
    handleRequestError(res, err, "Recognition failed");
  } finally {
    await safeUnlink(req.file?.path);
  }
});

app.get("/api/movies", async (req, res) => {
  try {
    const response = await axios.get(getPythonUrl("/movies"));
    res.status(response.status ?? 200).json(response.data);
  } catch (error) {
    handleRequestError(res, error, "Failed to fetch movies");
  }
});

app.post("/api/scene", async (req, res) => {
  try {
    const response = await axios.post(getPythonUrl("/scene"), req.body);
    res.status(response.status ?? 200).json(response.data);
  } catch (error) {
    handleRequestError(res, error, "Failed to fetch scene");
  }
});

app.post("/api/upload", upload.single("video"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "Video file is required" });
    return;
  }

  const tempPath = req.file.path;

  try {
    const videoName = req.file.originalname || req.file.filename;
    const form = new FormData();
    form.append("video", fs.createReadStream(tempPath), videoName);

    Object.entries(req.body || {}).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach((item) => form.append(key, item));
      } else if (value !== undefined && value !== null) {
        form.append(key, value);
      }
    });

    const response = await axios.post(getIngestionUrl(), form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });

    res.status(response.status ?? 200).json(response.data);
  } catch (error) {
    handleRequestError(res, error, "Failed to start ingestion job");
  } finally {
    await safeUnlink(tempPath);
  }
});

app.listen(3000, () => console.log("Gateway running on port 3000"));