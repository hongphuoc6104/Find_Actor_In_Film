# api/celery_worker.py
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

import json
import redis
from celery import Celery
from typing import Dict, Any

from flows.pipeline import face_clustering_pipeline
# Chỉ cần import load_config và deep_merge
from utils.config_loader import load_config, deep_merge

celery_app = Celery('worker', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
celery_app.conf.task_track_started = True
redis_client = redis.StrictRedis(host='localhost', port=6379, db=1, decode_responses=True)


@celery_app.task(bind=True)
def run_pipeline_task(self, job_id: str, movie_title: str, user_params: Dict[str, Any]):
    """
    Tác vụ chạy nền. Nhiệm vụ chính là chuyển tiếp thông tin xuống pipeline.
    """
    try:
        redis_client.hset(f"job:{job_id}", mapping={"status": "PROCESSING", "stage": "Starting pipeline..."})

        # Logic so sánh tham số cũ/mới để quyết định skip vẫn giữ nguyên
        cfg = load_config()
        metadata_filepath = cfg["storage"]["metadata_json"]
        old_params = {}
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
                old_params = all_metadata.get(movie_title, {}).get("last_run_params", {})

        # Worker không cần biết min_size là bao nhiêu, nó chỉ so sánh nếu người dùng có ghi đè hay không.
        new_det_score = user_params.get("quality_filters", {}).get("min_det_score")
        new_min_size = user_params.get("filter_clusters", {}).get("min_size")

        old_det_score = old_params.get("min_det_score")
        old_min_size = old_params.get("min_size")

        skip_ingestion = True
        skip_embedding = True  # Mặc định bỏ qua

        embedding_file = Path(cfg["storage"]["embeddings_folder_per_movie"]) / f"{movie_title}.parquet"

        if not embedding_file.exists():
            skip_ingestion = False
            skip_embedding = False
        elif new_det_score is not None and new_det_score != old_det_score:
            skip_embedding = False
        # Nếu min_size do người dùng ghi đè khác với lần trước thì cũng chạy lại
        elif new_min_size is not None and new_min_size != old_min_size:
            pass  # skip_embedding đã là True, chỉ cần chạy lại từ sau embedding

        # Nếu người dùng không ghi đè min_size, chúng ta vẫn chạy pipeline để logic tự động quyết định.
        # Logic "skip nếu không có gì thay đổi" sẽ được xử lý sâu hơn nếu cần.

        print(f"Starting job {job_id} for movie '{movie_title}'")
        face_clustering_pipeline(
            movie=movie_title,
            params_override=user_params,
            skip_ingestion=skip_ingestion,
            skip_embedding=skip_embedding
        )

        redis_client.hset(f"job:{job_id}", mapping={"status": "COMPLETED", "stage": "Finished"})
        print(f"Finished job {job_id}")

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        redis_client.hset(f"job:{job_id}", mapping={"status": "FAILED", "error": error_message})
        print(f"Job {job_id} failed: {e}")