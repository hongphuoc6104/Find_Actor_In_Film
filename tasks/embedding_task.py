# tasks/embedding_task.py
import os

import hashlib
import json
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from prefect import task
from insightface.app import FaceAnalysis

from utils.config_loader import load_config
from utils.image_utils import calculate_blur_score
from utils.vector_utils import l2_normalize
from .tracklet_task import link_tracklets

def _calculate_face_quality_score(face: Any, image_shape: tuple) -> float:
    """
    Tính điểm chất lượng bằng cách giải bài toán PnP từ landmark 3D để có góc quay chính xác.
    Đây là phương pháp đáng tin cậy nhất.
    """
    try:
        # Lấy landmark 3D (68 điểm) từ đối tượng face
        landmarks_3d = getattr(face, 'landmark_3d_68', None)
        if landmarks_3d is None or landmarks_3d.shape[0] < 68:
            return 0.0  # Không có đủ landmark 3D để tính

        # Chiều cao và chiều rộng của ảnh
        h, w = image_shape[:2]

        # Mô hình 3D của khuôn mặt (các điểm chính)
        # Đây là các điểm tham chiếu trên một khuôn mặt 3D chuẩn
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Đầu mũi (Nose tip)
            (0.0, -330.0, -65.0),        # Cằm (Chin)
            (-225.0, 170.0, -135.0),     # Khóe mắt trái (Left eye left corner)
            (225.0, 170.0, -135.0),      # Khóe mắt phải (Right eye right corner)
            (-150.0, -150.0, -125.0),    # Khóe miệng trái (Left Mouth corner)
            (150.0, -150.0, -125.0)      # Khóe miệng phải (Right mouth corner)
        ])

        # Các điểm 2D tương ứng trên ảnh
        # Lấy ra các điểm landmark 2D khớp với mô hình 3D
        image_points = np.array([
            landmarks_3d[30, :2],    # Đầu mũi
            landmarks_3d[8, :2],     # Cằm
            landmarks_3d[36, :2],    # Khóe mắt trái
            landmarks_3d[45, :2],    # Khóe mắt phải
            landmarks_3d[48, :2],    # Khóe miệng trái
            landmarks_3d[54, :2]     # Khóe miệng phải
        ], dtype="double")

        # Cấu hình camera giả định
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # Giải bài toán PnP để tìm vector quay và dịch chuyển
        dist_coeffs = np.zeros((4, 1))  # Giả định không có méo ảnh
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0

        # Chuyển đổi vector quay thành ma trận quay, rồi thành các góc Euler
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, translation_vector))
        )

        # Lấy góc pitch và yaw (độ)
        pitch, yaw, _ = euler_angles.flatten()[:3]

        # --- Logic tính điểm từ góc quay ---
        yaw_penalty = min((abs(yaw) / 50.0)**2, 1.0) # Phạt nặng hơn khi yaw > 50 độ
        pitch_penalty = min((abs(pitch) / 40.0)**2, 1.0) # Phạt nặng hơn khi pitch > 40 độ

        total_penalty = 0.7 * yaw_penalty + 0.3 * pitch_penalty
        final_score = max(0.0, 1.0 - total_penalty)

        return round(final_score, 2)

    except Exception:
        # Nếu có bất kỳ lỗi nào trong quá trình tính toán, trả về điểm thấp
        return 0.0

# --- KẾT THÚC THAY ĐỔI ---

def make_global_id(movie: str, frame: str, bbox: np.ndarray) -> str:
    s = f"{movie}|{frame}|{int(bbox[0])}|{int(bbox[1])}|{int(bbox[2])}|{int(bbox[3])}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def process_single_movie(movie_name: str, movie_frames_path: str, app: FaceAnalysis, config: dict):
    # ... (Toàn bộ phần còn lại của file giữ nguyên y hệt như phiên bản trước) ...
    image_files = sorted(
        os.path.join(movie_frames_path, f)
        for f in os.listdir(movie_frames_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    movie_specific_rows = []

    q_filters = config.get("quality_filters", {})
    lq_filter_cfg = q_filters.get("landmark_quality_filter", {})

    min_det_score = q_filters.get("min_det_score", 0.4)
    min_blur_clarity = q_filters.get("min_blur_clarity", 0.0)
    landmark_filter_enabled = lq_filter_cfg.get("enable", False)
    min_score_hard_cutoff = lq_filter_cfg.get("min_score_hard_cutoff", 0.0)

    print("\n" + "=" * 25 + " DEBUGGING FILTER PARAMETERS " + "=" * 25)
    print(f"  Movie: {movie_name}")
    print(f"  > Will use min_det_score:          {min_det_score}")
    print(f"  > Will use min_blur_clarity:         {min_blur_clarity}")
    print(f"  > Will use landmark_filter_enabled: {landmark_filter_enabled}")
    print(f"  > Will use min_score_hard_cutoff:   {min_score_hard_cutoff}")
    print("=" * 72 + "\n")

    stats = {"removed_det_score": 0, "removed_blur": 0, "removed_quality_score": 0, "passed": 0}
    debug_counter = 0

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
                if face.embedding is None: continue

                if face.det_score < min_det_score:
                    stats["removed_det_score"] += 1
                    continue

                quality_score = _calculate_face_quality_score(face, img.shape)

                if debug_counter < 10:
                    # print(
                    #     f"[DEBUG] Face in {os.path.basename(img_path)} - Quality Score: {quality_score} | Threshold: {min_score_hard_cutoff}")
                    debug_counter += 1

                if landmark_filter_enabled and quality_score < min_score_hard_cutoff:
                    stats["removed_quality_score"] += 1
                    continue

                original_bbox = face.bbox.astype(np.int32)
                face_crop_img = img[original_bbox[1]:original_bbox[3], original_bbox[0]:original_bbox[2]]
                if face_crop_img.size == 0: continue

                blur_score = calculate_blur_score(face_crop_img)
                if blur_score < min_blur_clarity:
                    stats["removed_blur"] += 1
                    continue

                stats["passed"] += 1

                global_id = make_global_id(movie_name, os.path.basename(img_path), original_bbox)
                crop_path = os.path.join(movie_face_crop_dir, f"{global_id}.jpg")
                cv2.imwrite(crop_path, face_crop_img)
                emb = l2_normalize(face.embedding)
                row = {
                    "global_id": global_id, "movie": movie_name, "frame": os.path.basename(img_path),
                    "bbox": original_bbox.tolist(), "det_score": float(face.det_score),
                    "emb": emb.tolist(), "face_crop_path": crop_path, "quality_score": quality_score
                }
                movie_specific_rows.append(row)

        except Exception as e:
            print(f"\n[Error] failed to process {img_path}: {e}")
            continue

    print(f"\n[Quality Stats for {movie_name}]:")
    print(f"  - Passed all filters: {stats['passed']}")
    print(f"  - Removed by det_score: {stats['removed_det_score']}")
    print(f"  - Removed by quality_score: {stats['removed_quality_score']}")
    print(f"  - Removed by blur_clarity: {stats['removed_blur']}")
    return movie_specific_rows, stats


@task(name="Embedding Task")
def embedding_task():
    # ... (Toàn bộ nội dung hàm này giữ nguyên y hệt) ...
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
        print(f"\nProcessing movie: {movie_name}")
        movie_frames_path = os.path.join(frames_root, movie_name)
        movie_rows, q_stats = process_single_movie(movie_name, movie_frames_path, app, cfg)
        new_data_generated = True
        if not movie_rows:
            print(f"⚠️ Không tìm thấy khuôn mặt nào đủ điều kiện cho phim '{movie_name}'.")
            all_metadata.setdefault(movie_name, {})["num_faces_detected"] = 0
            continue
        df_movie = pd.DataFrame(movie_rows)
        df_movie = link_tracklets(df_movie)
        if 'track_id' not in df_movie.columns or df_movie['track_id'].isna().all():
            print(f"⚠️ Không thể tạo tracklet cho phim '{movie_name}'.")
            df_movie.to_parquet(expected_parquet_path, index=False)
            print(
                f"✅ Đã lưu {len(movie_rows)} embeddings (không có tracklet) cho phim '{movie_name}' tại: {expected_parquet_path}")
            continue
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