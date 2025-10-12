# api/celery_worker.py
import sys
import os
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path
sys.path.insert(0, os.getcwd())

import json
import redis
from celery import Celery

from flows.pipeline import face_clustering_pipeline
from utils.config_loader import load_config, deep_merge

# Cấu hình Celery
celery_app = Celery('worker', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
celery_app.conf.task_track_started = True

# Kết nối đến Redis để lưu trạng thái
redis_client = redis.StrictRedis(host='localhost', port=6379, db=1, decode_responses=True)


@celery_app.task(bind=True)
def run_pipeline_task(self, job_id: str, movie_title: str, user_params: dict):
    """
    Tác vụ chạy nền, chứa logic thông minh để quyết định chạy lại từ đâu.
    """
    try:
        redis_client.hset(f"job:{job_id}", mapping={
            "status": "PROCESSING",
            "stage": "Analyzing parameters..."
        })

        # ... (Toàn bộ logic so sánh tham số giữ nguyên) ...
        cfg = load_config()
        metadata_filepath = cfg["storage"]["metadata_json"]

        old_params = {}
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
                old_params = all_metadata.get(movie_title, {}).get("last_run_params", {})

        new_det_score = user_params.get("quality_filters", {}).get("min_det_score")
        new_min_size = user_params.get("filter_clusters", {}).get("min_size")

        old_det_score = old_params.get("min_det_score")
        old_min_size = old_params.get("min_size")

        skip_ingestion = True
        skip_embedding = False

        embedding_file = Path(cfg["storage"]["embeddings_folder_per_movie"]) / f"{movie_title}.parquet"

        if new_det_score is not None and new_det_score != old_det_score:
            print(f"Job {job_id}: `min_det_score` thay đổi. Chạy lại từ Embedding.")
            redis_client.hset(f"job:{job_id}", "stage", "Re-running from Embedding...")
            if embedding_file.exists():
                embedding_file.unlink()
            skip_embedding = False

        elif new_min_size is not None and new_min_size != old_min_size:
            print(f"Job {job_id}: `min_size` thay đổi. Chạy lại từ Clustering.")
            redis_client.hset(f"job:{job_id}", "stage", "Re-running from Clustering...")
            skip_embedding = True

        elif not embedding_file.exists():
            print(f"Job {job_id}: File embedding không tồn tại. Chạy lại từ Embedding.")
            skip_embedding = False

        else:
            print(f"Job {job_id}: Không có tham số nào thay đổi đáng kể. Bỏ qua xử lý.")
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "COMPLETED",
                "stage": "No changes needed."
            })
            return {"status": "SKIPPED", "message": "Parameters unchanged."}

        print(f"Bắt đầu xử lý job {job_id} cho phim '{movie_title}'")
        face_clustering_pipeline(
            movie=movie_title,
            params_override=user_params,
            skip_ingestion=skip_ingestion,
            skip_embedding=skip_embedding
        )

        # --- SỬA LỖI CÚ PHÁP Ở ĐÂY ---
        redis_client.hset(f"job:{job_id}", mapping={"status": "COMPLETED"})
        print(f"Hoàn thành job {job_id}")

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        # --- VÀ SỬA LỖI CÚ PHÁP Ở ĐÂY ---
        redis_client.hset(f"job:{job_id}", mapping={
            "status": "FAILED",
            "error": error_message
        })
        print(f"Job {job_id} thất bại: {e}")