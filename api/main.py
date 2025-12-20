# api/main.py
import sys
import os
from pathlib import Path
import json
import uuid
from typing import Optional
from enum import Enum

# Thêm thư mục gốc vào sys.path để có thể import các module khác
sys.path.insert(0, os.getcwd())

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import worker và client redis
from api.celery_worker import run_pipeline_task, redis_client

# Import các service và công cụ khác
from services.recognition import recognize
from utils.config_loader import load_config, deep_merge
from services.scene_loader import _read_metadata


# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Face Recognition API",
    description="API cho hệ thống nhận diện và xử lý video.",
    version="1.2.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Phục vụ các file tĩnh (ảnh preview và videos) ---
app.mount("/static/previews", StaticFiles(directory="warehouse/cluster_previews"), name="previews")
try:
    app.mount("/videos", StaticFiles(directory="Data/video"), name="videos")
except Exception:
    pass  # Directory might not exist yet


# ==========================================================
# API ENDPOINTS CHO LUỒNG "TRAIN" (XỬ LÝ VIDEO)
# ==========================================================

@app.post("/api/v1/jobs/submit", status_code=202, tags=["Training Jobs"])
async def submit_job(
        video_file: UploadFile = File(...),
        movie_title: str = Form(...),
        min_det_score: Optional[float] = Form(None, description="(Ghi đè) Độ nhạy phát hiện khuôn mặt."),
        min_size: Optional[int] = Form(None, description="(Ghi đè) Ngưỡng tùy chỉnh cho min_size.")
):
    """
    Tải lên một video và các thông số để bắt đầu một tác vụ xử lý (train).
    """
    video_dir = Path("Data/video")
    video_dir.mkdir(exist_ok=True)

    file_extension = Path(video_file.filename).suffix or ".mp4"
    final_video_path = video_dir / f"{movie_title}{file_extension}"

    if not final_video_path.exists():
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
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Không tìm thấy Job ID.")
    return {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in job_data.items()}


# ==========================================================
# API ENDPOINTS CHO LUỒNG NHẬN DIỆN VÀ TRUY VẤN
# ==========================================================

@app.get("/api/v1/movies", tags=["Search"])
async def get_movies():
    """
    Lấy danh sách tất cả các phim có sẵn trong warehouse.
    """
    try:
        cfg = load_config()
        meta_path = Path(cfg.get("storage", {}).get("metadata_json", "Data/metadata.json"))
        
        if not meta_path.exists():
            return {"movies": []}
        
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        movies = []
        for movie_name, movie_data in metadata.items():
            if movie_name == "_generated":  # Skip internal metadata
                continue
            
            # Tìm file video
            video_path = None
            video_dir = Path("Data/video")
            if video_dir.exists():
                for ext in [".mp4", ".avi", ".mkv", ".mov"]:
                    candidate = video_dir / f"{movie_name}{ext}"
                    if candidate.exists():
                        video_path = f"/videos/{movie_name}{ext}"
                        break
            
            if not video_path:
                continue  # Skip movies without video files
            
            movies.append({
                "movie_name": movie_name,
                "video_url": video_path,
                "duration": movie_data.get("duration", "N/A"),
                "fps": movie_data.get("fps", 0)
            })
        
        return {"movies": movies}
    
    except Exception as e:
        print(f"[Error] Failed to load movies: {e}")
        return {"movies": []}


@app.post("/api/v1/search", tags=["Search"])
async def search_face(file: UploadFile = File(...)):
    """
    Tìm kiếm khuôn mặt trong tất cả các phim.
    Trả về danh sách phim có chứa khuôn mặt tương tự.
    """
    import tempfile
    from services.scene_loader import get_scenes_for_character
    
    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    temp_path = temp_dir / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Use recognition service
        results = recognize(str(temp_path))
        
        if results.get("is_unknown", True):
            return {
                "is_unknown": True,
                "matches": []
            }
        
        # Format response for frontend
        cfg = load_config()
        matches = []
        
        for movie_data in results.get("movies", []):
            movie_title = movie_data.get("movie")
            characters = movie_data.get("characters", [])
            
            if not characters:
                continue
            
            # Get video URL
            video_url = None
            video_dir = Path("Data/video")
            for ext in [".mp4", ".avi", ".mkv", ".mov"]:
                candidate = video_dir / f"{movie_title}{ext}"
                if candidate.exists():
                    video_url = f"/videos/{movie_title}{ext}"
                    break
            
            if not video_url:
                continue
            
            # Format characters with scenes
            formatted_chars = []
            for char in characters:
                char_id = char.get("character_id")
                scenes = char.get("scenes", [])
                
                # If scenes are not provided, try to load from scene_loader
                if not scenes and char_id:
                    scenes = get_scenes_for_character(
                        cfg=cfg,
                        movie_title=movie_title,
                        char_id=char_id
                    )
                
                formatted_chars.append({
                    "character_id": char_id,
                    "name": char.get("name", "Diễn viên"),  # Changed from character_name to name
                    "score": char.get("score", 0.0),
                    "score_display": f"{int(char.get('score', 0) * 100)}%",
                    "match_status": char.get("match_status", "SUGGESTION"),
                    "match_label": char.get("match_label", "Gợi ý"),
                    "scenes": scenes
                })
            
            matches.append({
                "movie": movie_title,
                "video_url": video_url,
                "characters": formatted_chars
            })
        
        return {
            "is_unknown": len(matches) == 0,
            "matches": matches
        }
    
    except Exception as e:
        print(f"[Error] Search failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the API. Go to /docs to see the endpoints."}