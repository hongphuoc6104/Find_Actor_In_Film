# 🎬 Face Clustering & Actor Search System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Vue.js](https://img.shields.io/badge/Vue.js-3.5-green?logo=vue.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Hệ thống nhận diện và phân cụm khuôn mặt diễn viên trong video sử dụng Deep Learning**

[English](#english) | [Tiếng Việt](#tiếng-việt)

</div>

---

## 📖 Mô Tả Dự Án

Đây là hệ thống **tự động nhận diện và phân cụm khuôn mặt** diễn viên trong video phim, cho phép:

- 🔍 **Tìm kiếm diễn viên** bằng ảnh: Upload ảnh diễn viên → Hệ thống trả về các cảnh họ xuất hiện
- 🎭 **Phân cụm tự động**: Gom nhóm các khuôn mặt giống nhau thành từng nhân vật mà không cần gán nhãn trước
- 📊 **12-Stage Pipeline**: Quy trình xử lý từ video thô đến cơ sở dữ liệu nhân vật hoàn chỉnh
- 🌐 **Web Interface**: Giao diện trực quan để tìm kiếm và xem kết quả

### 🛠️ Công Nghệ Sử Dụng

| Component | Technology |
|-----------|------------|
| **Face Detection** | RetinaFace |
| **Face Recognition** | ArcFace (InsightFace buffalo_l) |
| **Clustering** | Agglomerative Hierarchical Clustering |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Vue.js 3 + Vite + TailwindCSS |
| **Pipeline Orchestration** | Prefect |

---

## 🚀 Cài Đặt Nhanh (One-Click)

### Yêu Cầu Hệ Thống

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Ubuntu 20.04+ / Windows 10+ (WSL) / macOS | Ubuntu 22.04 |
| **Python** | 3.10+ | 3.10 |
| **Node.js** | 18+ | 20+ |
| **RAM** | 8GB | 16GB+ |
| **GPU** | Không bắt buộc | NVIDIA GPU với CUDA |

---

### ⚡ Cài Đặt (1 Lệnh)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Find_Actor_In_Film.git
cd Find_Actor_In_Film

# Chạy setup (tự động cài tất cả)
bash setup.sh
```

Script `setup.sh` sẽ tự động:
- ✅ Kiểm tra Python và Node.js
- ✅ Tạo virtual environment
- ✅ Cài đặt Python dependencies
- ✅ Cài đặt Frontend dependencies
- ✅ Tạo các thư mục cần thiết

---

### ▶️ Chạy Dự Án (1 Lệnh)

```bash
# Chạy cả Backend + Frontend
bash start.sh
```

Sau đó mở trình duyệt:
- 📺 **Frontend:** http://localhost:5173
- � **Backend API:** http://localhost:8000
- 📚 **API Docs:** http://localhost:8000/docs

**Dừng dự án:** Nhấn `Ctrl+C`

---

### 📂 Thêm Video Để Xử Lý

```bash
# 1. Copy video vào thư mục Data/video
cp /path/to/movie.mp4 Data/video/MOVIE_NAME.mp4

# 2. Kích hoạt môi trường
source .venv/bin/activate

# 3. Chạy pipeline xử lý
python -m flows.pipeline --movie "MOVIE_NAME"

# Hoặc chạy nhanh (bỏ qua các bước debug)
python -m flows.pipeline --movie "MOVIE_NAME" --fast
```

---

## 📖 Hướng Dẫn Chi Tiết (Manual Setup)

<details>
<summary>Click để xem hướng dẫn cài đặt thủ công</summary>

### Bước 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Find_Actor_In_Film.git
cd Find_Actor_In_Film
```

### Bước 2: Tạo Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### Bước 3: Cài Đặt Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Cài Đặt Frontend

```bash
cd frontend-client
npm install
cd ..
```

### Bước 5: Tạo Thư Mục Data

```bash
mkdir -p Data/video Data/frames Data/face_crops Data/embeddings
mkdir -p warehouse/parquet warehouse/cluster_previews
```

### Bước 6: Chạy Server

**Terminal 1 (Backend):**
```bash
source .venv/bin/activate
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend-client
npm run dev
```

</details>

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
├── requirements.txt          # Python dependencies
└── run_server.sh            # Server startup script
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

---

## 🔧 Troubleshooting

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `ModuleNotFoundError: insightface` | Chưa cài insightface | `pip install insightface` |
| `CUDA out of memory` | GPU không đủ VRAM | Sử dụng `onnxruntime` (CPU) thay vì `onnxruntime-gpu` |
| `No faces detected` | Chất lượng video thấp | Giảm `min_det_score` trong config |
| `Port 8000 already in use` | Port đang bị chiếm | `fuser -k 8000/tcp` |

### Kiểm tra cài đặt

```bash
# Test Python environment
python -c "import insightface; print('InsightFace OK')"
python -c "import fastapi; print('FastAPI OK')"
python -c "import pandas; print('Pandas OK')"

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

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis library
- [Prefect](https://prefect.io/) - Workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy cho một star nhé! ⭐**

</div>
