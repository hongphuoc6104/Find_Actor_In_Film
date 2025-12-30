# api/main.py
import sys
import os

# Fix Windows console encoding for Vietnamese characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
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
from contextlib import asynccontextmanager

# Import worker và client redis
from api.celery_worker import run_pipeline_task, redis_client

# Import các service và công cụ khác
from services.recognition import recognize
from utils.config_loader import load_config, deep_merge
from services.scene_loader import _read_metadata

# Path to yt-dlp in venv (fix for Windows PATH issues)
YT_DLP_PATH = str(Path(sys.executable).parent / "yt-dlp.exe")

# Path to Python in venv (fix for subprocess using system Python)
VENV_PYTHON = sys.executable


# --- Startup Event: Preload models ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload InsightFace model khi server khởi động"""
    print("[Startup] Preloading InsightFace model...")
    try:
        from utils.search_actor import _get_app
        _get_app()  # Load model vào RAM
        print("[Startup] InsightFace model loaded successfully!")
    except Exception as e:
        print(f"[Startup] Could not preload model: {e}")
    yield
    print("[Shutdown] Server stopping...")


# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(
    title="Face Recognition API",
    description="API cho hệ thống nhận diện và xử lý video.",
    version="1.2.0",
    lifespan=lifespan  # Attach startup handler
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

# Mount evaluation folder for UMAP charts
try:
    os.makedirs("warehouse/evaluation", exist_ok=True)
    app.mount("/static/evaluation", StaticFiles(directory="warehouse/evaluation"), name="evaluation")
except Exception:
    pass


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
# API ENDPOINTS CHO KHO VIDEO (DOWNLOAD ONLY)
# ==========================================================

@app.get("/api/v1/videos", tags=["Video Library"])
async def get_video_library():
    """
    Lấy danh sách TẤT CẢ video đã tải trong Data/video/.
    Bao gồm cả video đã xử lý và chưa xử lý.
    """
    video_dir = Path("Data/video")
    if not video_dir.exists():
        return {"videos": []}
    
    # Load metadata to check processed status
    cfg = load_config()
    meta_path = Path(cfg.get("storage", {}).get("metadata_json", "Data/metadata.json"))
    metadata = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except:
            pass
    
    videos = []
    for video_file in video_dir.iterdir():
        if video_file.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
            movie_name = video_file.stem
            is_processed = movie_name in metadata
            
            # Get file info
            file_size = video_file.stat().st_size
            file_size_mb = round(file_size / (1024 * 1024), 1)
            
            videos.append({
                "movie_name": movie_name,
                "video_url": f"/videos/{video_file.name}",
                "file_size_mb": file_size_mb,
                "is_processed": is_processed,
                "duration": metadata.get(movie_name, {}).get("duration", "N/A") if is_processed else None
            })
    
    # Sort: unprocessed first, then by name
    videos.sort(key=lambda x: (x["is_processed"], x["movie_name"]))
    
    return {"videos": videos}


# ==========================================================
# API ENDPOINTS CHO XỬ LÝ VIDEO (PROCESSING)
# ==========================================================

# In-memory process jobs (similar to youtube_jobs)
process_jobs = {}

def run_process_pipeline(job_id: str, movie_title: str, mode: str, params: dict):
    """Run pipeline in background with custom params"""
    import subprocess
    
    try:
        process_jobs[job_id]["status"] = "PROCESSING"
        process_jobs[job_id]["stage"] = "Đang khởi động pipeline..."
        
        # Build command based on mode
        cmd = [VENV_PYTHON, "-m", "flows.pipeline", "--movie", movie_title]
        
        if mode == "recluster":
            # Skip ingestion and embedding, only run clustering stages
            cmd.extend(["--skip-ingestion", "--skip-embedding"])
            process_jobs[job_id]["stage"] = "Đang gom nhóm lại (skip ingestion+embedding)..."
        else:
            process_jobs[job_id]["stage"] = "Đang xử lý từ đầu..."
        
        # TODO: Apply custom params (future enhancement)
        # For now, params are ignored - would need to write to temp config
        
        env = os.environ.copy()
        env["PREFECT_LOGGING_LEVEL"] = "WARNING"
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, 
            timeout=7200, cwd=os.getcwd(), env=env
        )
        
        output_combined = result.stdout + result.stderr
        has_error = 'Traceback' in output_combined and 'CancelledError' not in output_combined
        
        if not has_error:
            # Clear search cache
            try:
                from utils.search_actor import _load_characters_json
                _load_characters_json.cache_clear()
            except:
                pass
            process_jobs[job_id]["status"] = "COMPLETED"
            process_jobs[job_id]["stage"] = "Hoàn tất!"
        else:
            error_lines = [l for l in output_combined.split('\n') if 'Error' in l]
            error_msg = error_lines[-1][:100] if error_lines else "Unknown error"
            process_jobs[job_id]["status"] = "FAILED"
            process_jobs[job_id]["stage"] = f"Lỗi: {error_msg}"
            
    except subprocess.TimeoutExpired:
        process_jobs[job_id]["status"] = "FAILED"
        process_jobs[job_id]["stage"] = "Timeout - Quá thời gian"
    except Exception as e:
        process_jobs[job_id]["status"] = "FAILED"
        process_jobs[job_id]["stage"] = str(e)[:100]


@app.post("/api/v1/process/start", tags=["Processing"])
async def start_processing(
    movie_name: str = Form(..., description="Tên video cần xử lý"),
    mode: str = Form("full", description="Chế độ: 'full' (từ đầu) hoặc 'recluster' (gom nhóm lại)"),
    # Stage 1-2 params (require full reprocess)
    min_det_score: Optional[float] = Form(None, description="Độ tin cậy nhận diện (0.3-0.8)"),
    min_face_size: Optional[int] = Form(None, description="Kích thước mặt tối thiểu (30-100)"),
    min_blur_clarity: Optional[float] = Form(None, description="Độ rõ nét tối thiểu (25-60)"),
    landmark_hard_cutoff: Optional[float] = Form(None, description="Ngưỡng landmark cứng (0.3-0.8)"),
    landmark_core: Optional[float] = Form(None, description="Ngưỡng landmark core (0.5-0.9)"),
    # Stage 4+ params (can recluster)
    distance_threshold: Optional[float] = Form(None, description="Ngưỡng clustering (0.6-1.3)"),
    merge_threshold: Optional[float] = Form(None, description="Ngưỡng merge (0.45-0.65)"),
    min_track_size: Optional[int] = Form(None, description="Số frame tối thiểu/track (1-10)"),
    min_cluster_size: Optional[int] = Form(None, description="Kích thước cụm tối thiểu (2-30)"),
    post_merge_threshold: Optional[float] = Form(None, description="Ngưỡng post-merge (0.5-0.75)")
):
    """
    Bắt đầu xử lý video với custom parameters.
    
    Chế độ:
    - full: Xử lý từ đầu (ingestion → embedding → clustering)
    - recluster: Bỏ qua trích xuất, chỉ gom nhóm lại
    """
    import threading
    
    # Check video exists
    video_dir = Path("Data/video")
    video_exists = False
    for ext in [".mp4", ".avi", ".mkv", ".mov"]:
        if (video_dir / f"{movie_name}{ext}").exists():
            video_exists = True
            break
    
    if not video_exists:
        raise HTTPException(status_code=404, detail=f"Video '{movie_name}' không tồn tại")
    
    # Validate mode
    if mode not in ["full", "recluster"]:
        raise HTTPException(status_code=400, detail="Mode phải là 'full' hoặc 'recluster'")
    
    # For recluster, check if embeddings exist
    if mode == "recluster":
        emb_path = Path("Data/embeddings") / f"{movie_name}.parquet"
        if not emb_path.exists():
            raise HTTPException(
                status_code=400, 
                detail="Chưa có embeddings. Cần chạy 'Xử lý từ đầu' trước."
            )
    
    # Collect params
    params = {
        "quality_filters": {"landmark_quality_filter": {}},
        "clustering": {"distance_threshold": {}},
        "merge": {},
        "filter_clusters": {},
        "filter": {},
        "post_merge": {}
    }
    
    if min_det_score is not None:
        params["quality_filters"]["min_det_score"] = min_det_score
    if min_face_size is not None:
        params["quality_filters"]["min_face_size"] = min_face_size
    if min_blur_clarity is not None:
        params["quality_filters"]["min_blur_clarity"] = min_blur_clarity
    if landmark_hard_cutoff is not None:
        params["quality_filters"]["landmark_quality_filter"]["min_score_hard_cutoff"] = landmark_hard_cutoff
    if landmark_core is not None:
        params["quality_filters"]["landmark_quality_filter"]["min_score_for_core"] = landmark_core
    if distance_threshold is not None:
        params["clustering"]["distance_threshold"]["default"] = distance_threshold
    if merge_threshold is not None:
        params["merge"]["within_movie_threshold"] = merge_threshold
    if min_track_size is not None:
        params["filter"]["min_track_size"] = min_track_size
    if min_cluster_size is not None:
        params["filter_clusters"]["min_size"] = min_cluster_size
        params["filter"]["min_cluster_size"] = min_cluster_size
    if post_merge_threshold is not None:
        params["post_merge"]["distance_threshold"] = post_merge_threshold
    
    job_id = str(uuid.uuid4())
    process_jobs[job_id] = {
        "status": "QUEUED",
        "stage": "Đang khởi động...",
        "movie_name": movie_name,
        "mode": mode,
        "params": params
    }
    
    # Start in background
    thread = threading.Thread(
        target=run_process_pipeline, 
        args=(job_id, movie_name, mode, params)
    )
    thread.start()
    
    return {
        "job_id": job_id,
        "movie_name": movie_name,
        "mode": mode,
        "status": "QUEUED",
        "message": f"Đã bắt đầu xử lý video '{movie_name}' với chế độ '{mode}'"
    }


@app.get("/api/v1/process/status/{job_id}", tags=["Processing"])
async def get_process_status(job_id: str):
    """Lấy trạng thái job xử lý"""
    if job_id not in process_jobs:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    return process_jobs[job_id]


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


# ==========================================================
# API ENDPOINT CHO DOWNLOAD YOUTUBE + AUTO TRAIN
# ==========================================================

# In-memory job storage (for demo - no Redis needed)
youtube_jobs = {}

import threading

def run_pipeline_background(job_id: str, movie_title: str):
    """Run pipeline in background thread"""
    import subprocess
    try:
        youtube_jobs[job_id]["status"] = "PROCESSING"
        youtube_jobs[job_id]["stage"] = "Đang chạy pipeline xử lý video..."
        
        # Run pipeline with fast mode
        result = subprocess.run(
            [VENV_PYTHON, "-m", "flows.pipeline", "--movie", movie_title, "--skip-ingestion"],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            youtube_jobs[job_id]["status"] = "COMPLETED"
            youtube_jobs[job_id]["stage"] = "Hoàn tất! Bạn có thể tìm kiếm diễn viên."
        else:
            youtube_jobs[job_id]["status"] = "FAILED"
            youtube_jobs[job_id]["stage"] = f"Pipeline thất bại: {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        youtube_jobs[job_id]["status"] = "FAILED"
        youtube_jobs[job_id]["stage"] = "Timeout - Video quá lớn"
    except Exception as e:
        youtube_jobs[job_id]["status"] = "FAILED"
        youtube_jobs[job_id]["stage"] = str(e)[:200]


def clean_youtube_url(url: str) -> str:
    """Clean YouTube URL - remove playlist, radio, and other extra parameters"""
    import re
    from urllib.parse import urlparse, parse_qs, urlencode
    
    # Extract video ID
    if 'youtu.be/' in url:
        # Short URL format: https://youtu.be/VIDEO_ID
        match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url)
        if match:
            return f'https://www.youtube.com/watch?v={match.group(1)}'
    
    # Standard URL format
    match = re.search(r'[?&]v=([a-zA-Z0-9_-]{11})', url)
    if match:
        return f'https://www.youtube.com/watch?v={match.group(1)}'
    
    return url


def run_download_and_pipeline(job_id: str, url: str, movie_title: str, output_path: Path):
    """Run download and pipeline in background - FULLY ASYNC"""
    import subprocess
    
    try:
        youtube_jobs[job_id]["stage"] = "Đang tải video..."
        
        # Download video with cookies to avoid 403 error
        download_cmd = [
            YT_DLP_PATH, "--no-playlist",
            "--cookies-from-browser", "chrome",  # Try chrome first, fallback to edge
            "--extractor-args", "youtube:player_client=android",
            "-f", "bestvideo[vcodec^=avc][height<=1080]+bestaudio/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            url
        ]
        
        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=600)
        
        # If failed, try with Edge
        if result.returncode != 0:
            youtube_jobs[job_id]["stage"] = "Thử lại với Edge..."
            download_cmd[3] = "edge"
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=600)
        
        # If failed, try with Firefox
        if result.returncode != 0:
            youtube_jobs[job_id]["stage"] = "Thử lại với Firefox..."
            download_cmd[3] = "firefox"
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=600)
        
        # If still failed, try without cookies (for public videos)
        if result.returncode != 0:
            youtube_jobs[job_id]["stage"] = "Thử không dùng cookies..."
            download_cmd_no_cookie = [
                YT_DLP_PATH, "--no-playlist",
                "--extractor-args", "youtube:player_client=android",
                "-f", "bestvideo[vcodec^=avc][height<=1080]+bestaudio/best[height<=1080]/best",
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                url
            ]
            result = subprocess.run(download_cmd_no_cookie, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0 or not output_path.exists():
            youtube_jobs[job_id]["status"] = "FAILED"
            youtube_jobs[job_id]["stage"] = f"Download thất bại: {result.stderr[:150]}"
            return
        
        youtube_jobs[job_id]["status"] = "PROCESSING"
        youtube_jobs[job_id]["stage"] = "Download xong! Đang xử lý video..."
        
        # Run pipeline - redirect stderr to avoid Prefect INFO logs
        env = os.environ.copy()
        env["PREFECT_LOGGING_LEVEL"] = "WARNING"  # Suppress INFO logs
        
        # Run FULL pipeline with ALL stages (preview, labeling, evaluation)
        # No skip flags - run everything A-Z for new videos
        result = subprocess.run(
            [VENV_PYTHON, "-m", "flows.pipeline", "--movie", movie_title],
            capture_output=True, text=True, timeout=3600, cwd=os.getcwd(), env=env
        )
        
        # Check for success - look for completion indicators
        output_combined = result.stdout + result.stderr
        # Exclude Prefect internal errors (CancelledError is not a real failure)
        is_cancelled = 'CancelledError' in output_combined or 'cancel scope' in output_combined
        has_real_error = ('Traceback' in output_combined or 'Exception' in output_combined) and not is_cancelled
        
        if not has_real_error:
            # Rebuild search index to include new video
            youtube_jobs[job_id]["stage"] = "Đang cập nhật search index..."
            try:
                from utils.search_actor import _load_characters_json
                _load_characters_json.cache_clear()  # Clear cache to reload new data
            except Exception:
                pass
            youtube_jobs[job_id]["status"] = "COMPLETED"
            youtube_jobs[job_id]["stage"] = "Hoàn tất! Bạn có thể tìm kiếm diễn viên."
        elif 'Traceback' in output_combined or 'Exception' in output_combined:
            # Real error
            error_lines = [l for l in output_combined.split('\n') if 'Error' in l or 'Exception' in l]
            error_msg = error_lines[-1][:100] if error_lines else "Unknown error"
            youtube_jobs[job_id]["status"] = "FAILED"
            youtube_jobs[job_id]["stage"] = f"Lỗi: {error_msg}"
        else:
            # Probably succeeded even if returncode != 0 (Prefect quirk)
            try:
                from utils.search_actor import _load_characters_json
                _load_characters_json.cache_clear()
            except Exception:
                pass
            youtube_jobs[job_id]["status"] = "COMPLETED"
            youtube_jobs[job_id]["stage"] = "Hoàn tất! Bạn có thể tìm kiếm diễn viên."
            
    except subprocess.TimeoutExpired:
        youtube_jobs[job_id]["status"] = "FAILED"
        youtube_jobs[job_id]["stage"] = "Timeout"
    except Exception as e:
        youtube_jobs[job_id]["status"] = "FAILED"
        youtube_jobs[job_id]["stage"] = str(e)[:100]


@app.post("/api/v1/youtube/download-and-process", tags=["YouTube"])
async def download_and_process_youtube(url: str = Form(...), movie_title: str = Form(None)):
    """
    Download video từ YouTube và TỰ ĐỘNG chạy pipeline xử lý.
    - Giới hạn video dưới 30 phút
    - URL được tự động làm sạch (bỏ playlist, radio params)
    - Trả về job_id để theo dõi tiến trình
    """
    import subprocess
    import re
    
    # Clean URL first
    url = clean_youtube_url(url)
    
    # Validate URL
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    if not re.match(youtube_regex, url):
        raise HTTPException(status_code=400, detail="URL không hợp lệ. Vui lòng nhập URL YouTube.")
    
    job_id = str(uuid.uuid4())
    youtube_jobs[job_id] = {"status": "DOWNLOADING", "stage": "Đang lấy thông tin video...", "url": url}
    
    try:
        # Get video info QUICKLY (non-blocking for basic info)
        info_cmd = [YT_DLP_PATH, "--dump-json", "--no-download", "--no-playlist", url]
        result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            youtube_jobs[job_id]["status"] = "FAILED"
            youtube_jobs[job_id]["stage"] = "Không thể lấy thông tin video. Kiểm tra lại URL."
            raise HTTPException(status_code=400, detail="Không thể lấy thông tin video. Thử URL ngắn hơn (bỏ &list=...)")
        
        video_info = json.loads(result.stdout)
        duration = video_info.get("duration", 0)
        title = video_info.get("title", "video")
        
        # Duration info (no limit)
        duration_display = f"{duration//60}:{duration%60:02d}"
        
        # Clean title for filename - use timestamp format for clarity
        if not movie_title:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            movie_title = f"YT_{timestamp}"
        
        youtube_jobs[job_id]["movie_title"] = movie_title
        youtube_jobs[job_id]["duration"] = duration
        youtube_jobs[job_id]["stage"] = f"Đang tải video ({duration//60}:{duration%60:02d})..."
        
        video_dir = Path("Data/video")
        video_dir.mkdir(parents=True, exist_ok=True)
        output_path = video_dir / f"{movie_title}.mp4"
        
        # If video exists, skip download and run pipeline
        if output_path.exists():
            youtube_jobs[job_id]["status"] = "PROCESSING"
            youtube_jobs[job_id]["stage"] = "Video đã tồn tại, đang xử lý..."
            thread = threading.Thread(target=run_pipeline_background, args=(job_id, movie_title))
            thread.start()
        else:
            # Start download + pipeline in background - FULLY ASYNC
            thread = threading.Thread(target=run_download_and_pipeline, args=(job_id, url, movie_title, output_path))
            thread.start()
        
        return {
            "job_id": job_id,
            "movie_title": movie_title,
            "duration": duration,
            "duration_display": f"{duration//60}:{duration%60:02d}",
            "status": "DOWNLOADING",
            "message": "Đang tải video trong nền..."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        youtube_jobs[job_id]["status"] = "FAILED"
        youtube_jobs[job_id]["stage"] = str(e)[:100]
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/youtube/cancel/{job_id}", tags=["YouTube"])
async def cancel_youtube_job(job_id: str):
    """Hủy job đang chạy VÀ xóa tất cả dữ liệu đã tạo"""
    if job_id not in youtube_jobs:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    # Mark as cancelled
    youtube_jobs[job_id]["status"] = "CANCELLED"
    youtube_jobs[job_id]["stage"] = "Đã hủy - đang dọn dữ liệu..."
    
    # Delete all data for this movie to avoid garbage
    movie_title = youtube_jobs[job_id].get("movie_title")
    deleted_info = {"files": 0, "folders": 0}
    
    if movie_title:
        try:
            from delete_movie import delete_movie_data
            result = delete_movie_data(movie_title, dry_run=False)
            deleted_info = {
                "files": len(result.get("files", [])),
                "folders": len(result.get("folders", []))
            }
        except Exception as e:
            print(f"[Cancel] Error cleaning up: {e}")
    
    youtube_jobs[job_id]["stage"] = "Đã hủy và dọn sạch dữ liệu"
    
    return {
        "status": "CANCELLED",
        "message": "Job đã hủy và dữ liệu đã được dọn sạch",
        "deleted": deleted_info
    }


@app.delete("/api/v1/movies/{movie_name}", tags=["Movies"])
async def delete_movie(movie_name: str):
    """
    Xóa HOÀN TOÀN dữ liệu của 1 video.
    Bao gồm: video, frames, crops, embeddings, clusters, previews, và JSON entries.
    """
    from delete_movie import delete_movie_data
    
    result = delete_movie_data(movie_name, dry_run=False)
    
    if not result["files"] and not result["folders"]:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho video: {movie_name}")
    
    return {
        "status": "DELETED",
        "movie_name": movie_name,
        "deleted_files": len(result["files"]),
        "deleted_folders": len(result["folders"]),
        "details": result
    }



@app.get("/api/v1/youtube/status/{job_id}", tags=["YouTube"])
async def get_youtube_job_status(job_id: str):
    """Lấy trạng thái của job download+process YouTube"""
    if job_id not in youtube_jobs:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    return youtube_jobs[job_id]


# Keep old download-only endpoint for backward compatibility
@app.post("/api/v1/youtube/download", tags=["YouTube"])
async def download_youtube(url: str = Form(...), movie_title: str = Form(None)):
    """Download video từ YouTube (KHÔNG tự động xử lý)"""
    import subprocess
    import re
    from urllib.parse import urlparse, parse_qs, urlencode
    
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    if not re.match(youtube_regex, url):
        raise HTTPException(status_code=400, detail="URL không hợp lệ. Vui lòng nhập URL YouTube đầy đủ.")
    
    # Clean URL - remove timestamp and other unnecessary params
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # Keep only video ID (v parameter)
    clean_params = {}
    if 'v' in query_params:
        clean_params['v'] = query_params['v'][0]
        url = f"https://www.youtube.com/watch?{urlencode(clean_params)}"
    elif 'youtu.be' in url:
        # Short URL format: https://youtu.be/VIDEO_ID
        pass  # yt-dlp handles this
    
    try:
        # Multiple methods to try - YouTube bot detection changes frequently
        methods = [
            # Method 1: Edge browser (user's default)
            ["yt-dlp", "--dump-json", "--no-download", "--no-playlist", 
             "--cookies-from-browser", "edge", url],
            # Method 2: Firefox cookies
            ["yt-dlp", "--dump-json", "--no-download", "--no-playlist", 
             "--cookies-from-browser", "firefox", url],
            # Method 3: Chrome cookies
            ["yt-dlp", "--dump-json", "--no-download", "--no-playlist", 
             "--cookies-from-browser", "chrome", url],
            # Method 4: Android client (sometimes avoids bot check)
            ["yt-dlp", "--dump-json", "--no-download", "--no-playlist",
             "--extractor-args", "youtube:player_client=android", url],
            # Method 5: Web client with user agent
            ["yt-dlp", "--dump-json", "--no-download", "--no-playlist",
             "--extractor-args", "youtube:player_client=web",
             "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", url],
        ]
        
        result = None
        last_error = ""
        for cmd in methods:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and result.stdout.strip():
                break
            last_error = result.stderr[:300] if result.stderr else "Unknown error"
        
        if result.returncode != 0 or not result.stdout.strip():
            # Parse common errors
            if "Private video" in last_error:
                raise HTTPException(status_code=400, detail="Video riêng tư, không thể tải.")
            elif "age" in last_error.lower():
                raise HTTPException(status_code=400, detail="Video yêu cầu đăng nhập (giới hạn độ tuổi).")
            elif "unavailable" in last_error.lower():
                raise HTTPException(status_code=400, detail="Video không khả dụng.")
            elif "bot" in last_error.lower() or "Sign in" in last_error:
                raise HTTPException(status_code=400, detail="YouTube yêu cầu xác thực. Hãy mở YouTube trong Firefox/Chrome và thử lại.")
            else:
                raise HTTPException(status_code=400, detail="Không thể lấy thông tin video. Thử lại sau.")
        
        video_info = json.loads(result.stdout)
        duration = video_info.get("duration", 0)
        title = video_info.get("title", "video")
        
        # Generate movie title if not provided
        if not movie_title:
            # Clean title: remove Vietnamese diacritics and special chars
            clean_title = title
            # Remove special characters, keep alphanumeric and spaces
            clean_title = re.sub(r'[^\w\s-]', '', clean_title, flags=re.UNICODE)
            clean_title = clean_title[:30].strip().replace(' ', '_').upper()
            movie_title = clean_title if clean_title else "VIDEO"
        else:
            # Clean user-provided title
            movie_title = re.sub(r'[^a-zA-Z0-9_-]', '', movie_title.upper())
        
        video_dir = Path("Data/video")
        video_dir.mkdir(parents=True, exist_ok=True)
        output_path = video_dir / f"{movie_title}.mp4"
        
        if output_path.exists():
            return {"status": "EXISTS", "movie_title": movie_title, "duration": duration}
        
        # Download video with cookies (same as info extraction)
        download_cmd = [
            "yt-dlp",
            "-f", "bestvideo[vcodec^=avc][height<=1080]+bestaudio/best[height<=1080]/best",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--retries", "3",
            "--fragment-retries", "3",
            "--cookies-from-browser", "edge",  # Use Edge browser cookies
            "--no-playlist",
            url
        ]
        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        # If failed with edge, try firefox then chrome
        if result.returncode != 0:
            download_cmd[12] = "firefox"
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            download_cmd[12] = "chrome"
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=1800)
        
        # Fallback: try without cookies (for public videos)
        if result.returncode != 0:
            download_cmd_no_cookie = [
                "yt-dlp",
                "-f", "bestvideo[vcodec^=avc][height<=1080]+bestaudio/best[height<=1080]/best",
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                "--retries", "3",
                "--no-playlist",
                url
            ]
            result = subprocess.run(download_cmd_no_cookie, capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            # Cleanup partial download
            if output_path.exists():
                output_path.unlink()
            part_files = list(video_dir.glob(f"{movie_title}.*"))
            for f in part_files:
                f.unlink()
            
            error_msg = result.stderr[-300:] if result.stderr else "Unknown error"
            if "HTTP Error 403" in error_msg:
                raise HTTPException(status_code=500, detail="Video bị chặn tải. Thử lại sau.")
            else:
                raise HTTPException(status_code=500, detail=f"Download thất bại. Thử video khác.")
        
        return {"status": "SUCCESS", "movie_title": movie_title, "duration": duration}
        
    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Timeout - Video quá lớn hoặc mạng chậm")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Lỗi xử lý dữ liệu video")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)[:100]}")


@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the API. Go to /docs to see the endpoints."}