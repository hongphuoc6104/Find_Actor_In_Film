# 🏛️ PROJECT FACELOOK - SYSTEM CONTEXT

## 1. Mục tiêu

Hệ thống Video Analytics tự động: Ingestion -> Face Detection -> Clustering -> Search & Retrieval.

## 2. Tech Stack & Architecture

- **Core AI:** Python 3.10, InsightFace, PyTorch, Pandas, Scikit-learn (Agglomerative Clustering).
- **Orchestration:** Prefect (Workflow Management).
- **Backend:** FastAPI, Celery (Async Workers), Redis (Broker/Cache).
- **Frontend:** ReactJS (hoặc Streamlit cho MVP).
- **Storage:** Local Filesystem (Video, Parquet, JSON).

## 3. Cấu trúc thư mục chuẩn

- `/tasks`: Scripts xử lý AI (Embedding, Cluster, Merge).
- `/flows`: Prefect pipeline flows.
- `/api`: FastAPI source code & Celery workers.
- `/warehouse`:
  - `characters.json`: Database file (Core ghi -> Backend đọc).
  - `cluster_previews/`: Ảnh đại diện nhân vật.
  - `parquet/`: Dữ liệu trung gian.

## 4. QUY TẮC TỐI THƯỢNG (Golden Rules)

1. **Isolation:** AI Core, Backend, Frontend hoạt động độc lập.
2. **Data Contracts:** Giao tiếp qua file JSON/Parquet quy định trước.
3. **Mocking:** Backend/Frontend phải dùng Mock Data nếu Core chưa chạy xong.
4. **State Management:** Luôn cập nhật file STATUS\_\*.md sau mỗi task.
