# tasks/embedding_task.py
import os
import time
import hashlib
import json
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from prefect import task
from insightface.app import FaceAnalysis

from utils.config_loader import load_config
from utils.image_utils import calculate_blur_score, check_brightness, check_contrast
from utils.vector_utils import l2_normalize
from .tracklet_task import link_tracklets


# -------------------- Hàm lọc chất lượng mới -------------------- #

def _calculate_face_quality_score(face: Any) -> float:
    """
    Tính điểm chất lượng dựa trên sự hiện diện của các landmarks chính.
    Điểm 1.0 là hoàn hảo (thấy rõ 2 mắt, mũi, miệng).
    """
    if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
        return 0.0

    landmarks = face.landmark_2d_106

    # Các chỉ số của các bộ phận chính trên khuôn mặt
    key_indices = {
        "left_eye": [35, 36, 33, 39, 42, 40],
        "right_eye": [89, 90, 87, 93, 96, 94],
        "nose": [47, 51, 52, 53],
        "mouth": [52, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68, 70]
    }

    score = 0.0
    # Mắt: mỗi mắt đóng góp 0.3 điểm
    # Kiểm tra xem có bất kỳ điểm nào của mắt được phát hiện không
    if any(landmarks[i, 0] > 0 for i in key_indices["left_eye"]): score += 0.3
    if any(landmarks[i, 0] > 0 for i in key_indices["right_eye"]): score += 0.3
    # Mũi: đóng góp 0.2 điểm
    if any(landmarks[i, 0] > 0 for i in key_indices["nose"]): score += 0.2
    # Miệng: đóng góp 0.2 điểm
    if any(landmarks[i, 0] > 0 for i in key_indices["mouth"]): score += 0.2

    return round(score, 2)


def make_global_id(movie: str, frame: str, bbox: np.ndarray) -> str:
    s = f"{movie}|{frame}|{int(bbox[0])}|{int(bbox[1])}|{int(bbox[2])}|{int(bbox[3])}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def process_single_movie(movie_name: str, movie_frames_path: str, app: FaceAnalysis, config: dict):
    image_files = sorted(
        os.path.join(movie_frames_path, f)
        for f in os.listdir(movie_frames_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    movie_specific_rows = []
    q_filters = config.get("quality_filters", {})
    lq_filter_cfg = q_filters.get("landmark_quality_filter", {})
    landmark_filter_enabled = lq_filter_cfg.get("enable", False)
    min_score_hard_cutoff = lq_filter_cfg.get("min_score_hard_cutoff", 0.0)

    stats = {"removed_det_score": 0, "removed_face_ratio": 0, "removed_blur": 0, "removed_quality_score": 0}

    storage_cfg = config["storage"]
    face_crops_root = storage_cfg["face_crops_root"]
    movie_face_crop_dir = os.path.join(face_crops_root, movie_name)
    os.makedirs(movie_face_crop_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc=f"Scanning {movie_name}"):
        try:
            img = cv2.imread(img_path)
            if img is None: continue

            faces = app.get(img) or []
            if not faces: continue

            for face in faces:
                # BƯỚC 1: Tính điểm chất lượng cho mọi khuôn mặt
                quality_score = _calculate_face_quality_score(face)

                # BƯỚC 2: Áp dụng bộ lọc cứng (loại bỏ rác tuyệt đối)
                if landmark_filter_enabled and quality_score < min_score_hard_cutoff:
                    stats["removed_quality_score"] += 1
                    continue

                # Các bộ lọc khác giữ nguyên
                if face.det_score < q_filters.get("min_det_score", 0.4):
                    stats["removed_det_score"] += 1
                    continue

                if face.embedding is None: continue

                # ... các xử lý khác ...
                original_bbox = face.bbox.astype(np.int32)
                final_face_crop = img[original_bbox[1]:original_bbox[3], original_bbox[0]:original_bbox[2]]
                if final_face_crop.size == 0: continue

                global_id = make_global_id(movie_name, os.path.basename(img_path), original_bbox)
                crop_path = os.path.join(movie_face_crop_dir, f"{global_id}.jpg")
                cv2.imwrite(crop_path, final_face_crop)

                emb = l2_normalize(face.embedding)

                # BƯỚC 3: Thêm cột quality_score vào kết quả
                row = {
                    "global_id": global_id,
                    "movie": movie_name,
                    "frame": os.path.basename(img_path),
                    "bbox": original_bbox.tolist(),
                    "det_score": float(face.det_score),
                    "emb": emb.tolist(),
                    "face_crop_path": crop_path,
                    "quality_score": quality_score  # Cột mới quan trọng
                }
                movie_specific_rows.append(row)

        except Exception as e:
            print(f"\n[Error] failed to process {img_path}: {e}")
            continue

    print(f"\n[Quality Stats for {movie_name}]: Removed by hard_cutoff_quality_score: {stats['removed_quality_score']}")
    return movie_specific_rows, stats


@task(name="Embedding Task")
def embedding_task():
    cfg = load_config()

    print("Initializing InsightFace model...")
    app = FaceAnalysis(name=cfg["embedding"]["model"], providers=cfg["embedding"]["providers"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model ready.")

    storage_cfg = cfg["storage"]
    metadata_filepath = storage_cfg["metadata_json"]
    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    except FileNotFoundError:
        all_metadata = {}

    frames_root = storage_cfg["frames_root"]
    embeddings_folder = storage_cfg["embeddings_folder_per_movie"]
    os.makedirs(embeddings_folder, exist_ok=True)

    movie_folders = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        if active_movie in movie_folders:
            movie_folders = [active_movie]
        else:
            print(f"[Embedding] Không tìm thấy thư mục frames cho '{active_movie}'. Bỏ qua.")
            return

    new_data_generated = False
    for movie_name in movie_folders:
        expected_parquet_path = os.path.join(embeddings_folder, f"{movie_name}.parquet")

        # Bắt buộc chạy lại nếu file cũ không có cột 'quality_score' (để cập nhật)
        force_rerun = True
        if os.path.exists(expected_parquet_path):
            try:
                # Đọc chỉ 1 cột để kiểm tra cho nhanh
                df_check = pd.read_parquet(expected_parquet_path, columns=['quality_score'])
                if 'quality_score' in df_check.columns:
                    force_rerun = False
            except Exception:
                force_rerun = True

        if not force_rerun:
            print(f"File embedding cho phim '{movie_name}' đã có cột quality_score. Bỏ qua.")
            continue

        new_data_generated = True
        print(f"\nProcessing movie: {movie_name}")
        movie_frames_path = os.path.join(frames_root, movie_name)
        movie_rows, q_stats = process_single_movie(movie_name, movie_frames_path, app, cfg)

        if not movie_rows:
            print(f"⚠️ Không tìm thấy khuôn mặt nào đủ điều kiện cho phim '{movie_name}'.")
            all_metadata.setdefault(movie_name, {})["num_faces_detected"] = 0
            continue

        df_movie = pd.DataFrame(movie_rows)
        df_movie = link_tracklets(df_movie)
        centroids = df_movie.groupby("track_id")["emb"].apply(
            lambda e: l2_normalize(np.median(np.stack(e.to_list()), axis=0))).rename("track_centroid")
        df_movie = df_movie.merge(centroids, on="track_id")
        df_movie["track_centroid"] = df_movie["track_centroid"].apply(lambda x: x.tolist())
        df_movie.to_parquet(expected_parquet_path, index=False)
        print(f"✅ Đã lưu {len(movie_rows)} embeddings cho phim '{movie_name}' tại: {expected_parquet_path}")

        all_metadata.setdefault(movie_name, {})["num_faces_detected"] = len(movie_rows)

    if new_data_generated:
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Cập nhật thành công file {metadata_filepath}")
    else:
        print("\nKhông có phim mới nào để xử lý.")

