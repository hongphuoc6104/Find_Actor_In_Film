import express from "express";
import multer from "multer";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();
const app = express();
app.use(cors());
const upload = multer({ dest: "uploads/" });

app.post("/api/recognize", upload.single("image"), async (req, res) => {
  try {
    const form = new FormData();
    form.append("image", fs.createReadStream(req.file.path));
    const r = await axios.post(process.env.PYTHON_URL + "/recognize", form, {
      headers: form.getHeaders(),
    });
    res.json(r.data);
  } catch (err) {
    res.status(500).json({ error: "Recognition failed" });
  } finally {
    fs.unlinkSync(req.file.path);
  }
});

app.listen(3000, () => console.log("Gateway running on port 3000"));
