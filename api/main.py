# api/main.py
import sys
import os
from pathlib import Path
import json
import uuid

# Thêm thư mục gốc vào sys.path để có thể import các module khác
sys.path.insert(0, os.getcwd())

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Import worker và client redis
from api.celery_worker import run_pipeline_task, redis_client

# Import các service và công cụ khác
from services.recognition import recognize
from utils.config_loader import load_config
# (Bạn có thể tạo một file api/helpers.py và chuyển các hàm helper này vào đó cho gọn)
from services.scene_loader import _read_metadata

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Face Recognition API",
    description="API cho hệ thống nhận diện và xử lý video.",
    version="1.2.0"
)

# --- Phục vụ các file tĩnh (ảnh preview) ---
app.mount("/static/previews", StaticFiles(directory="warehouse/cluster_previews"), name="previews")


# ==========================================================
# API ENDPOINTS CHO LUỒNG "TRAIN" (XỬ LÝ VIDEO)
# ==========================================================

@app.post("/api/v1/jobs/submit", status_code=202, tags=["Training Jobs"])
async def submit_job(
        video_file: UploadFile = File(...),
        movie_title: str = Form(...),
        min_det_score: float = Form(None),  # Tham số tùy chọn
        min_size: int = Form(None)  # Tham số tùy chọn
):
    """
    Tải lên một video và các thông số để bắt đầu một tác vụ xử lý (train).

    - **movie_title**: Tên định danh cho video này (không có đuôi file).
    - **min_det_score**: (Tùy chọn) Độ nhạy phát hiện khuôn mặt (0.0 -> 1.0).
    - **min_size**: (Tùy chọn) Ngưỡng để xác định diễn viên chính.
    """
    video_dir = Path("Data/video")
    video_dir.mkdir(exist_ok=True)

    file_extension = Path(video_file.filename).suffix or ".mp4"
    final_video_path = video_dir / f"{movie_title}{file_extension}"

    if final_video_path.exists():
        print(f"Video '{movie_title}' đã tồn tại. Sẽ kiểm tra và chạy lại các bước cần thiết.")
    else:
        with open(final_video_path, "wb") as buffer:
            buffer.write(await video_file.read())
        print(f"Đã lưu video mới: {final_video_path}")

    # Tạo cấu trúc params để gửi cho worker
    user_params = {}
    if min_det_score is not None:
        user_params = deep_merge(user_params, {"quality_filters": {"min_det_score": min_det_score}})
    if min_size is not None:
        user_params = deep_merge(user_params, {"filter_clusters": {"min_size": min_size}})

    job_id = str(uuid.uuid4())
    run_pipeline_task.delay(
        job_id=job_id,
        movie_title=movie_title,
        user_params=user_params
    )

    redis_client.hset(f"job:{job_id}", mapping={"status": "QUEUED"})

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "message": "Đã nhận yêu cầu. Quá trình xử lý đã được đưa vào hàng đợi."
    }


@app.get("/api/v1/jobs/status/{job_id}", tags=["Training Jobs"])
async def get_job_status(job_id: str):
    """
    Kiểm tra trạng thái của một tác vụ xử lý bằng Job ID.
    """
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Không tìm thấy Job ID.")

    return job_data


# ==========================================================
# API ENDPOINTS CHO LUỒNG NHẬN DIỆN VÀ TRUY VẤN
# ==========================================================

# (Các endpoints khác như /recognize, /movies, /movies/{title}/characters, /videos/{title}
#  sẽ được thêm vào đây. Chúng ta sẽ hoàn thiện chúng ở bước sau.)

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the API. Go to /docs to see the endpoints."}