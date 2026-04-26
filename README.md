# Face Clustering & Actor Search System

## Giới Thiệu Dự Án

Đây là hệ thống AI giúp tìm diễn viên/nhân vật trong phim bằng ảnh khuôn mặt. Người dùng có thể tải video, chạy pipeline xử lý, sau đó upload một ảnh để hệ thống trả về các phim có khuôn mặt tương tự, kèm các đoạn thời gian xuất hiện.

### Bài Toán
- Tìm một diễn viên/nhân vật trong kho phim lớn bằng ảnh đầu vào.
- Tránh việc phải xem thủ công toàn bộ video để dò cảnh xuất hiện.
- Tự động hóa quá trình trích xuất frame, nhận diện mặt, gom cụm và truy vấn.

### Hướng Giải Quyết
- Tách video thành frame theo pipeline tự động.
- Phát hiện khuôn mặt bằng InsightFace.
- Tạo embedding và gom cụm khuôn mặt thành từng nhân vật.
- Lưu kết quả vào warehouse để phục vụ tìm kiếm nhanh.
- Cho phép upload ảnh truy vấn và trả về phim, score và timestamp.

### Điểm Nổi Bật
- Pipeline xử lý video nhiều giai đoạn, có thể chạy full hoặc chạy lại phần clustering.
- Có giao diện web 3 tab: tải video, xử lý video, tìm kiếm khuôn mặt.
- Hỗ trợ tinh chỉnh tham số cho detection, clustering, filtering và post-merge.
- Tự động tạo preview ảnh nhân vật và manifest kết quả.

### Luồng Hoạt Động
`Video/Input -> Frame Extraction -> Face Detection -> Embedding -> Clustering -> Character Manifest -> Image Search -> Timestamp Results`

### Tech Stack
- Backend: `FastAPI`, `Python`
- AI/CV: `InsightFace`, `ONNX Runtime`, `OpenCV`, `scikit-learn`
- Orchestration: `Prefect`
- Frontend: `Vue 3`, `Vite`
- Download/Video: `yt-dlp`, `FFmpeg`
- Storage: `Parquet`, `JSON`


## Cài Đặt & Chạy

### Cách Nhanh Nhất: Chạy Bằng Docker

Nếu chỉ muốn chạy dự án nhanh, dùng Docker Compose để khởi động toàn bộ hệ thống gồm backend, frontend, Redis và worker.

**Yêu cầu:**
- Docker Desktop hoặc Docker Engine
- Docker Compose v2
- RAM khuyến nghị từ 16GB vì model nhận diện khuôn mặt khá nặng

**Clone và chạy:**
```bash
git clone https://github.com/hongphuoc6104/Find_Actor_In_Film.git
cd Find_Actor_In_Film
docker compose up --build
```

**Mở ứng dụng:**
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

**Dừng ứng dụng:**
```bash
docker compose down
```

**Chạy nền:**
```bash
docker compose up --build -d
```

**Xem log:**
```bash
docker compose logs -f backend
docker compose logs -f worker
docker compose logs -f frontend
```

**Dữ liệu được lưu ở máy thật:**
- Video: `Data/video/`
- Frames/crops/embeddings: `Data/frames/`, `Data/face_crops/`, `Data/embeddings/`
- Kết quả xử lý: `warehouse/`
- Config: `configs/`

**Model InsightFace:**
- Docker bỏ preload model lúc startup để giao diện mở nhanh hơn.
- Lần đầu dùng chức năng tìm kiếm/xử lý khuôn mặt, container sẽ tải model `buffalo_l`.
- Model được cache trong Docker volume `insightface-cache`, nên các lần chạy sau không cần tải lại.

> Lưu ý: Docker mặc định chạy ONNX Runtime CPU. Nếu muốn dùng NVIDIA GPU trong container, cần cài NVIDIA Container Toolkit và chỉnh thêm cấu hình GPU cho service backend/worker.

---

### Cài Đặt Thủ Công

### Yêu Cầu Hệ Thống

| Yêu cầu | Tối thiểu | Khuyến nghị |
|---------|-----------|-------------|
| **Hệ điều hành** | Ubuntu 20.04+ / Windows 10+ / macOS 12+ | Ubuntu 22.04 |
| **Python** | 3.10+ | 3.10 |
| **Node.js** | 20.19+ | 20.19+ |
| **RAM** | 8GB | 16GB+ |
| **GPU** | Không bắt buộc | NVIDIA GPU với CUDA |

---

### Bước 1: Cài Đặt Git và Python (nếu chưa có)

**Windows:**
- **Git:** Tải từ https://git-scm.com/ → chạy installer → chọn mặc định
- **Python:** Tải từ https://python.org/ → **tick "Add Python to PATH"** khi cài

**Linux/macOS:** Thường đã có sẵn. Kiểm tra: `git --version` và `python3 --version`

---

### Bước 2: Clone Repository

```bash
git clone https://github.com/hongphuoc6104/Find_Actor_In_Film.git
cd Find_Actor_In_Film
```

---

### Bước 3: Cài Đặt FFmpeg (bắt buộc)

FFmpeg dùng để xử lý video. **Bắt buộc phải cài.**

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows (hướng dẫn chi tiết):**

1. Truy cập https://github.com/BtbN/FFmpeg-Builds/releases
2. Tải file `ffmpeg-master-latest-win64-gpl.zip`
3. Giải nén vào thư mục, ví dụ: `C:\ffmpeg`
4. Thêm vào PATH:
   - Nhấn `Win + R`, gõ `sysdm.cpl`, nhấn Enter
   - Chọn tab **Advanced** → **Environment Variables**
   - Trong **System variables**, tìm **Path** → **Edit**
   - Nhấn **New** → nhập `C:\ffmpeg\bin`
   - Nhấn **OK** để lưu
5. Mở **CMD mới** và kiểm tra: `ffmpeg -version`

**Kiểm tra cài đặt thành công:**
```bash
ffmpeg -version
# Nếu hiện thông tin version → OK!
```

---

### Bước 4: Tạo Virtual Environment

| Hệ điều hành | Lệnh |
|--------------|------|
| **Linux/macOS** | `python3 -m venv .venv && source .venv/bin/activate` |
| **Windows (PowerShell)** | `python -m venv .venv; .venv\Scripts\Activate.ps1` |
| **Windows (CMD)** | `python -m venv .venv && .venv\Scripts\activate.bat` |

---

### Bước 5: Cài Đặt Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý:** File `requirements.txt` đã bao gồm tất cả thư viện cần thiết:
> - `uvicorn` - chạy API server
> - `yt-dlp` - tải video từ YouTube  
> - `insightface` - nhận diện khuôn mặt
> - `fastapi` - API framework
> - Và các thư viện khác...

#### 🪟 Cho Windows: Nếu gặp lỗi khi cài đặt

**1. Lỗi `Microsoft Visual C++ 14.0 or greater is required` (khi cài insightface):**
```powershell
# Cài insightface từ wheel có sẵn thay vì build từ source
pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl
```

**2. Lỗi `numpy.dtype size changed`:**
```powershell
pip install "numpy<2.0.0" --force-reinstall
```

**3. Lỗi `DLL load failed` (khi import onnxruntime):**
- Tải và cài Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

**4. Lỗi PowerShell: `running scripts is disabled`:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Gõ Y và Enter để xác nhận
```

---

### Bước 6: GPU Acceleration (Tùy chọn)

Chỉ áp dụng nếu bạn có **GPU NVIDIA**:

```bash
# Kiểm tra GPU
nvidia-smi

# Nếu có GPU, cài onnxruntime-gpu để tăng tốc 5-10x
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

> **macOS:** Không hỗ trợ CUDA, sẽ tự động chạy trên CPU.

---

### Bước 7: Cài Đặt Frontend

**Cài Node.js** (nếu chưa có):

| Hệ điều hành | Cách cài |
|--------------|----------|
| **Ubuntu/Debian** | `curl -fsSL https://deb.nodesource.com/setup_20.x \| sudo -E bash - && sudo apt install -y nodejs` |
| **macOS** | `brew install node@20` |
| **Windows** | Tải từ https://nodejs.org/ và chạy installer |

**Kiểm tra:**
```bash
node --version  # v20.x.x
npm --version   # 10.x.x
```

**Cài dependencies:**
```bash
cd frontend-client
npm install
cd ..
```

---

### Bước 8: Chạy Ứng Dụng

Mở 2 terminal:

**Terminal 1 - Backend:**
```bash
# Linux/macOS
source .venv/bin/activate
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Windows
.venv\Scripts\activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend-client
npm run dev
```

**Mở trình duyệt:**
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

**Dừng:** Nhấn `Ctrl+C` trong mỗi terminal

---

### 📂 Thêm Video Để Xử Lý

#### Cách 1: Qua Web Interface (Khuyến nghị)

Giao diện có 3 tabs:

| Tab | Chức năng |
|-----|----------|
| 📥 **Tải Video** | Download video từ YouTube |
| ⚙️ **Xử Lý** | Chạy pipeline với các tham số tùy chỉnh |
| 🔍 **Tìm Kiếm** | Upload ảnh để tìm diễn viên |

**Quy trình:**
1. Mở `http://localhost:5173`
2. Tab **Tải Video** → Paste URL → Click **"Tải Video"**
3. Tab **⚙️ Xử Lý** → Chọn video → Điều chỉnh tham số (tùy chọn) → **"Bắt Đầu Xử Lý"**
4. Tab **🔍 Tìm Kiếm** → Upload ảnh → Xem kết quả

#### Cách 2: Tải video thủ công

```bash
# Cài yt-dlp nếu chưa có
pip install yt-dlp
```

**Linux/macOS:**
```bash
yt-dlp -f "bestvideo[height<=1080]+bestaudio" --merge-output-format mp4 -o "Data/video/MOVIE_NAME.mp4" "https://youtube.com/watch?v=VIDEO_ID"
```

**Windows (CMD - một dòng):**
```cmd
yt-dlp -f "bestvideo[height<=1080]+bestaudio" --merge-output-format mp4 -o "Data\video\MOVIE_NAME.mp4" "https://youtube.com/watch?v=VIDEO_ID"
```

> **Lưu ý:** Nếu bị lỗi 403/rate limit, thêm `--cookies-from-browser` trước `-f`:

| Browser | Lệnh thêm vào |
|---------|--------------|
| Chrome | `--cookies-from-browser chrome` |
| Firefox | `--cookies-from-browser firefox` |
| Edge | `--cookies-from-browser edge` |

**Ví dụ với Chrome cookies:**
```cmd
yt-dlp --cookies-from-browser chrome -f "bestvideo[height<=1080]+bestaudio" --merge-output-format mp4 -o "Data\video\MOVIE_NAME.mp4" "URL"
```

**Chạy pipeline:**

Linux/macOS:
```bash
source .venv/bin/activate
python -m flows.pipeline --movie "MOVIE_NAME"
```

Windows:
```cmd
.venv\Scripts\activate
python -m flows.pipeline --movie "MOVIE_NAME"
```


---

## 📖 Hướng Dẫn Sử Dụng

### 1. Tìm Kiếm Diễn Viên

1. Mở giao diện web tại `http://localhost:5173`
2. Click **"Upload Ảnh"** và chọn ảnh khuôn mặt diễn viên
3. Hệ thống sẽ trả về:
   - Các phim mà diễn viên xuất hiện
   - Các cảnh (timestamp) trong từng phim
   - Độ tương đồng (similarity score)

### 2. Xử Lý Video Mới

```bash
# 1. Thêm video vào Data/video/
cp new_movie.mp4 Data/video/NEW_MOVIE.mp4

# 2. Chạy pipeline
python -m flows.pipeline --movie "NEW_MOVIE"

# 3. Rebuild search index (nếu cần)
python -c "from utils.indexer import build_character_index; build_character_index(force_rebuild=True)"
```

### 3. Xem Kết Quả Clustering

Sau khi chạy pipeline, kết quả được lưu tại:

```
warehouse/
├── parquet/
│   └── MOVIE_NAME_clusters.parquet  # Data clusters
├── cluster_previews/
│   └── MOVIE_NAME/
│       ├── character_1/             # Ảnh preview nhân vật 1
│       ├── character_2/             # Ảnh preview nhân vật 2
│       └── ...
└── characters.json                   # Manifest tất cả nhân vật
```

---

## 📁 Cấu Trúc Dự Án

```
Find_Actor_In_Film/
├── api/                      # FastAPI backend
│   └── main.py              # API endpoints
├── configs/                  # Configuration files
│   ├── config.yaml          # Main config
│   └── videos/              # Per-video configs
├── flows/                    # Pipeline orchestration
│   └── pipeline.py          # Main 12-stage pipeline
├── tasks/                    # Pipeline tasks
│   ├── ingestion_task.py    # Video → Frames
│   ├── embedding_task.py    # Frames → Embeddings
│   ├── cluster_task.py      # Clustering
│   └── ...
├── services/                 # Business logic
│   └── recognition.py       # Face search service
├── utils/                    # Utilities
│   ├── indexer.py           # Search index
│   └── search_actor.py      # Search logic
├── frontend-client/          # Vue.js frontend
│   ├── src/
│   └── package.json
├── Data/                     # Video & processed data (gitignored)
├── warehouse/                # Output data (gitignored)
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Cấu Hình

File cấu hình chính: `configs/config.yaml`

```yaml
# Clustering parameters
clustering:
  distance_threshold:
    default: 1.15        # Ngưỡng khoảng cách

# Merge parameters
merge:
  within_movie_threshold: 0.55

# Quality filters
quality_filters:
  min_det_score: 0.45    # Min detection confidence
  min_face_size: 50      # Min face size in pixels
```

**Tùy chỉnh cho từng video:** Tạo file `configs/videos/MOVIE_NAME.yaml`

📖 **[Xem hướng dẫn chi tiết tất cả tham số →](PARAMETERS.md)**

---

## 🔧 Troubleshooting

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `ModuleNotFoundError: insightface` | Chưa cài insightface | `pip install insightface` |
| `CUDA out of memory` | GPU không đủ VRAM | Sử dụng `onnxruntime` (CPU) thay vì `onnxruntime-gpu` |
| `No faces detected` | Chất lượng video thấp | Giảm `min_det_score` trong config |
| `Port 8000 already in use` | Port đang bị chiếm | `fuser -k 8000/tcp` |
| `ParameterTypeError` | Prefect version issue | Đã fix trong code, không cần xử lý |

### 🎬 Lỗi tải video YouTube

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `Sign in to confirm you're not a bot` | YouTube yêu cầu xác thực | Mở YouTube trong Edge/Chrome/Firefox trước, sau đó thử lại |
| `No supported JavaScript runtime` | Thiếu JS runtime | Hệ thống tự động fallback - thử lại |
| `Video riêng tư` | Video private | Chọn video public khác |
| `Video không khả dụng` | Video bị xóa/chặn | Thử video khác |
| `HTTP Error 403` | Bị chặn | Đợi vài phút rồi thử lại |

> 💡 **Tip:** Hệ thống tự động thử lấy cookies từ Edge → Firefox → Chrome. Hãy đảm bảo ít nhất 1 trình duyệt đã mở YouTube trước đó.

### 🖥️ Lỗi GPU

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `CUDAExecutionProvider not available` | Chạy trên CPU | Bình thường nếu không có GPU |
| `libcublasLt.so not found` | Thiếu CUDA libraries | `sudo apt install nvidia-cuda-toolkit` |

### 🪟 Lỗi trên Windows

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `error: Microsoft Visual C++ 14.0 or greater is required` khi cài `insightface` | Thiếu build tools | Tải wheel có sẵn: `pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl` |
| `ValueError: numpy.dtype size changed` | numpy 2.x không tương thích với InsightFace | Hạ cấp numpy: `pip install "numpy<2.0.0" --force-reinstall` |
| `DLL load failed while importing onnxruntime_pybind11_state` | Thiếu Visual C++ Redistributable | Tải và cài từ: https://aka.ms/vs/17/release/vc_redist.x64.exe |
| `UnicodeEncodeError: 'charmap' codec can't encode characters` | Windows console không hỗ trợ Unicode/tiếng Việt | Đã fix trong code (v1.2.1+). Nếu gặp lỗi, thêm vào đầu file Python: `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` |
| `Vite requires Node.js version 20.19+` | Node.js quá cũ | Cài Node.js >= 20.19.0 từ https://nodejs.org/ |
| `FileNotFoundError` khi tải YouTube | yt-dlp không trong PATH | Đã fix trong code - nếu gặp lỗi, chạy: `.venv\Scripts\yt-dlp.exe` thay vì `yt-dlp` |
| `ModuleNotFoundError: No module named 'pandas'` khi xử lý video | Pipeline dùng Python hệ thống thay vì venv | Đã fix trong code (v1.2.1+). Đảm bảo chạy Backend từ venv đã activated |

> ⚠️ **Lưu ý cho Windows:**
> - Nên cài **Visual C++ Redistributable 2015-2022** trước khi bắt đầu
> - Dùng **Node.js >= 20.19.0** (không dùng phiên bản cũ hơn)
> - Nếu gặp lỗi build InsightFace, dùng wheel có sẵn thay vì build từ source

### ⚠️ Lỗi khi xử lý video

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `[Stop] No embeddings found` hoặc không có nhóm nào được tạo | Video quá ngắn, chất lượng thấp, hoặc không có khuôn mặt rõ | Thử video khác có nhiều khuôn mặt hơn, hoặc giảm `min_det_score` trong config |
| Pipeline chạy xong nhưng không có kết quả tìm kiếm | Clustering không tạo được nhóm nào (video quá ngắn hoặc ít mặt) | Đây là hành vi bình thường với video ngắn (<1 phút). Thử video dài hơn |

### Kiểm tra cài đặt

```bash
# Test Python environment
python -c "import insightface; print('InsightFace OK')"
python -c "import fastapi; print('FastAPI OK')"

# Kiểm tra GPU
nvidia-smi  # Xem thông tin GPU
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"

# Test face detection
python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
print('Face Analysis ready!')
"
```

---

## 📊 Pipeline Stages

| Stage | Tên | Mô tả |
|-------|-----|-------|
| 1 | **Ingestion** | Trích xuất frames từ video |
| 1.5 | **Analysis** | Phân tích đặc điểm video (độ sáng, độ rõ) |
| 2 | **Embedding** | Tạo face embeddings với ArcFace |
| 3 | **Warehouse** | Gom embeddings vào data warehouse |
| 4 | **Clustering** | Phân cụm ban đầu (Agglomerative) |
| 5 | **Merge** | Gộp các cụm giống nhau |
| 6 | **Filter** | Lọc bỏ cụm nhiễu/nhỏ |
| 7 | **Post-Merge** | Hấp thụ cụm vệ tinh vào cụm chính |
| 8 | **Preview** | Tạo ảnh preview cho mỗi nhân vật |
| 9 | **Manifest** | Tạo file characters.json |
| 10 | **Labeling** | Gán nhãn tự động (nếu có labeled_faces) |
| 11 | **Validation** | Tạo báo cáo chất lượng |
| 12 | **Evaluation** | Đánh giá hiệu quả clustering |

---

## 📜 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 👤 Tác Giả

- **Hong Phuoc** - [GitHub](https://github.com/hongphuoc6104)

---

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis library
- [Prefect](https://prefect.io/) - Workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework

---

<div align="center">

** Nếu dự án hữu ích, hãy cho một star nhé! **

</div>
