# tasks/embedding_task.py
import os
import time
import hashlib
import json
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


# -------------------- Tiện ích -------------------- #

def make_global_id(movie: str, frame: str, bbox: np.ndarray) -> str:
    s = f"{movie}|{frame}|{int(bbox[0])}|{int(bbox[1])}|{int(bbox[2])}|{int(bbox[3])}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ---------------- Xử lý từng phim ---------------- #

def process_single_movie(movie_name: str, movie_frames_path: str, app: FaceAnalysis, config: dict):
    image_files = sorted(
        os.path.join(movie_frames_path, f)
        for f in os.listdir(movie_frames_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    movie_specific_rows = []
    q_filters = config.get("quality_filters", {})
    stats = {
        "removed_det_score": 0,
        "removed_face_ratio": 0,
        "removed_blur": 0,
    }

    storage_cfg = config["storage"]
    face_crops_root = storage_cfg["face_crops_root"]
    movie_face_crop_dir = os.path.join(face_crops_root, movie_name)
    os.makedirs(movie_face_crop_dir, exist_ok=True)

    for img_path in tqdm(image_files, desc=f"Scanning {movie_name}"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize trước khi detect để tăng tốc
            scale = 1.0
            processing_img = img
            h, w, _ = processing_img.shape
            if h > config["pre_resize_dim"] or w > config["pre_resize_dim"]:
                scale = config["pre_resize_dim"] / max(h, w)
                processing_img = cv2.resize(img, (int(w * scale), int(h * scale)))

            faces = app.get(processing_img) or []
            if not faces:
                continue

            frame_area = processing_img.shape[0] * processing_img.shape[1]
            min_face_area = q_filters.get("min_face_ratio", 0.003) * frame_area

            good_quality_faces = []
            for face in faces:
                # Lọc tầng 1: điểm detect + diện tích khuôn mặt
                if face.det_score < q_filters.get("min_det_score", 0.4):
                    stats["removed_det_score"] += 1
                    continue
                x1, y1, x2, y2 = face.bbox
                if (x2 - x1) * (y2 - y1) < min_face_area:
                    stats["removed_face_ratio"] += 1
                    continue

                # Crop theo bbox ở toạ độ gốc
                orig_x1, orig_y1, orig_x2, orig_y2 = np.round(face.bbox / scale).astype(int)
                face_crop = img[orig_y1:orig_y2, orig_x1:orig_x2]
                if face_crop.size == 0:
                    continue

                # Lọc tầng 2: sáng + tương phản
                gray_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                if q_filters.get("brightness", {}).get("enable") and not check_brightness(
                    gray_face_crop, q_filters["brightness"]["value_range"]
                ):
                    continue
                if q_filters.get("contrast", {}).get("enable") and not check_contrast(
                    gray_face_crop, q_filters["contrast"]["min_value"]
                ):
                    continue

                # Lọc tầng 3: độ nét
                blur_score = calculate_blur_score(face_crop)
                if blur_score < q_filters.get("min_blur_clarity", 60.0):
                    stats["removed_blur"] += 1
                    continue

                good_quality_faces.append(face)

            # Chọn top N gương mặt tốt nhất trên frame
            good_quality_faces.sort(key=lambda f: f.det_score, reverse=True)
            selected_faces = good_quality_faces[: config["max_faces_per_frame"]]

            for face in selected_faces:
                if face.embedding is None:
                    continue

                original_bbox = np.round(face.bbox / scale).astype(np.int32)
                face_crop = img[original_bbox[1]: original_bbox[3], original_bbox[0]: original_bbox[2]]
                if face_crop.size == 0:
                    continue

                global_id = make_global_id(movie_name, os.path.basename(img_path), original_bbox)
                crop_path = os.path.join(movie_face_crop_dir, f"{global_id}.jpg")
                cv2.imwrite(crop_path, face_crop)

                emb = face.embedding
                if config["embedding"].get("l2_normalize", True):
                    emb = l2_normalize(emb)

                row = {
                    "global_id": global_id,
                    "movie": movie_name,
                    "frame": os.path.basename(img_path),
                    "bbox": original_bbox.tolist(),
                    "det_score": float(face.det_score),
                    "emb": emb.tolist(),
                    "face_crop_path": crop_path,
                    "ts_created": int(time.time()),
                    "version": 1,
                }
                movie_specific_rows.append(row)

        except Exception as e:
            print(f"\n[Error] failed to process {img_path}: {e}")
            continue

    return movie_specific_rows, stats


# -------------------- Task chính -------------------- #

@task(name="Embedding Task")
def embedding_task():
    """
    Chế độ mặc định: xử lý tất cả thư mục frames.
    Chế độ đơn-phim: nếu set ENV FS_ACTIVE_MOVIE, chỉ xử lý đúng phim đó.
    - Sinh embeddings + face crops
    - Link tracklet, tính track_centroid (median + L2)
    - Ghi parquet per-movie và cập nhật metadata
    """
    cfg = load_config()
    q_cfg = cfg.get("quality_filters", {})
    config = {
        "embedding": cfg["embedding"],
        "storage": cfg["storage"],
        "quality_filters": {
            "min_det_score": q_cfg.get("min_det_score", 0.4),
            "min_face_ratio": q_cfg.get("min_face_ratio", 0.003),
            "min_blur_clarity": q_cfg.get("min_blur_clarity", 60.0),
            "brightness": q_cfg.get("brightness", {}),
            "contrast": q_cfg.get("contrast", {}),
        },
        "max_faces_per_frame": cfg.get("search", {}).get("max_faces_per_frame", 5),
        "pre_resize_dim": cfg.get("pre_resize_dim", 1280),
    }
    storage_cfg = config["storage"]

    # InsightFace init
    print("Initializing InsightFace model...")
    app = FaceAnalysis(
        name=config["embedding"]["model"],
        providers=config["embedding"]["providers"],
    )
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        # fallback CPU nếu không có GPU
        app.prepare(ctx_id=-1, det_size=(640, 640))
    print("Model ready.")

    # Đọc metadata cũ (nếu có)
    metadata_filepath = storage_cfg["metadata_json"]
    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    except FileNotFoundError:
        all_metadata = {}

    frames_root = storage_cfg["frames_root"]
    embeddings_folder = storage_cfg["embeddings_folder_per_movie"]
    os.makedirs(embeddings_folder, exist_ok=True)

    # Lấy danh sách phim theo thư mục frames
    movie_folders = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]

    # Chế độ đơn-phim
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        if active_movie in movie_folders:
            movie_folders = [active_movie]
            print(f"[Embedding] Chế độ đơn-phim: chỉ xử lý '{active_movie}'")
        else:
            print(f"[Embedding] Không tìm thấy thư mục frames cho '{active_movie}' tại '{frames_root}'. Bỏ qua embedding.")
            return True

    new_data_generated = False
    quality_logs = []

    for movie_name in movie_folders:
        expected_parquet_path = os.path.join(embeddings_folder, f"{movie_name}.parquet")

        # Nếu đã có file và đã có track_centroid -> bỏ qua
        if os.path.exists(expected_parquet_path):
            try:
                df_movie = pd.read_parquet(expected_parquet_path)
            except Exception:
                df_movie = pd.DataFrame()
            if not df_movie.empty and "track_centroid" in df_movie.columns:
                print(f"File embedding cho phim '{movie_name}' đã tồn tại. Bỏ qua.")
                # đảm bảo metadata tối thiểu
                if movie_name not in all_metadata:
                    all_metadata[movie_name] = {}
                all_metadata[movie_name]["num_faces_detected"] = len(df_movie)
                all_metadata[movie_name]["embedding_file_path"] = expected_parquet_path
                continue

            # Nếu thiếu track_centroid -> bổ sung
            if not df_movie.empty:
                print(f"File embedding cho phim '{movie_name}' thiếu track_centroid. Đang cập nhật...")
                df_movie = link_tracklets(df_movie)
                centroids = (
                    df_movie.groupby("track_id")["emb"]
                    .apply(lambda e: l2_normalize(np.median(np.stack(e.to_list()), axis=0)))
                    .rename("track_centroid")
                )
                df_movie = df_movie.merge(centroids, on="track_id")
                df_movie["track_centroid"] = df_movie["track_centroid"].apply(lambda x: x.tolist())
                df_movie.to_parquet(expected_parquet_path, index=False)
                new_data_generated = True
                if movie_name not in all_metadata:
                    all_metadata[movie_name] = {}
                all_metadata[movie_name]["num_faces_detected"] = len(df_movie)
                all_metadata[movie_name]["embedding_file_path"] = expected_parquet_path
                continue
            # nếu đọc lỗi/empty -> xử lý lại từ đầu như bên dưới

        # Sinh mới embeddings cho phim
        new_data_generated = True
        print(f"\nProcessing movie: {movie_name}")
        movie_frames_path = os.path.join(frames_root, movie_name)

        movie_rows, q_stats = process_single_movie(movie_name, movie_frames_path, app, config)
        quality_logs.append({"movie": movie_name, **q_stats})

        if movie_name not in all_metadata:
            all_metadata[movie_name] = {}

        if not movie_rows:
            print(f"⚠️ Không tìm thấy khuôn mặt nào đủ điều kiện cho phim '{movie_name}'.")
            all_metadata[movie_name]["num_faces_detected"] = 0
            all_metadata[movie_name]["embedding_file_path"] = None
            continue

        df_movie = pd.DataFrame(movie_rows)

        # Liên kết liên tiếp thành tracklet + gán track_id
        df_movie = link_tracklets(df_movie)

        # Tính track_centroid (median + L2) theo track
        centroids = (
            df_movie.groupby("track_id")["emb"]
            .apply(lambda e: l2_normalize(np.median(np.stack(e.to_list()), axis=0)))
            .rename("track_centroid")
        )
        df_movie = df_movie.merge(centroids, on="track_id")
        df_movie["track_centroid"] = df_movie["track_centroid"].apply(lambda x: x.tolist())

        df_movie.to_parquet(expected_parquet_path, index=False)
        print(f"✅ Đã lưu {len(movie_rows)} embeddings cho phim '{movie_name}' tại: {expected_parquet_path}")

        all_metadata[movie_name]["num_faces_detected"] = len(movie_rows)
        all_metadata[movie_name]["embedding_file_path"] = expected_parquet_path

    # Ghi log chất lượng track (toàn cục)
    if quality_logs:
        os.makedirs("logs", exist_ok=True)
        pd.DataFrame(quality_logs).to_csv("logs/track_quality.csv", index=False)
        print("[INFO] Saved track quality log -> logs/track_quality.csv")

    # Cập nhật metadata nếu có thay đổi
    if new_data_generated:
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Cập nhật thành công file {metadata_filepath}")
    else:
        print("\nKhông có phim mới nào để xử lý.")

    return True


if __name__ == "__main__":
    embedding_task()
