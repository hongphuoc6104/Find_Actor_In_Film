# tasks/analyze_video_task.py
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from prefect import task

from utils.config_loader import load_config
from utils.image_utils import calculate_blur_score
# Tái sử dụng hàm đếm khuôn mặt từ ingestion_task
from tasks.ingestion_task import count_faces_retina

# =====================================================================
# CÁC NGƯỠNG PHÂN TÍCH - CÓ THỂ TINH CHỈNH TẠI ĐÂY
# =====================================================================
ANALYSIS_THRESHOLDS = {
    # Phân loại độ sáng (giá trị pixel trung bình của ảnh xám, 0-255)
    "LIGHTING_DARK": 75,  # Dưới ngưỡng này => "Tối"
    "LIGHTING_BRIGHT": 160,  # Trên ngưỡng này => "Sáng", ở giữa => "Trung bình"

    # Phân loại độ nét (điểm Variance of Laplacian)
    "CLARITY_BLURRY": 180,  # Dưới ngưỡng này => "Mờ" (do rung lắc, out-focus)
    "CLARITY_SHARP": 380,  # Trên ngưỡng này => "Nét", ở giữa => "Trung bình"

    # Phân loại độ phức tạp (số khuôn mặt trung bình trên mỗi frame)
    "COMPLEXITY_CROWDED": 2.5  # Từ ngưỡng này trở lên => "Đông đúc"
}


# =====================================================================


@task(name="Analyze Video Task")
def analyze_video_task(movie_title: str) -> Dict[str, str]:
    """
    Phân tích các frames đã được trích xuất để tạo ra một "hồ sơ video" (Video Profile).
    Hồ sơ này bao gồm các đặc tính về độ dài, ánh sáng, độ nét và độ phức tạp.
    SỬ DỤNG LẤY MẪU CÓ HỆ THỐNG ĐỂ ĐẢM BẢO KẾT QUẢ NHẤT QUÁN.
    """
    print(f"\n--- Analyzing video profile for '{movie_title}' ---")
    cfg = load_config()
    frames_root = Path(cfg["storage"]["frames_root"])
    movie_frames_dir = frames_root / movie_title

    # --- 1. Lấy danh sách frames và kiểm tra (Không thay đổi) ---
    if not movie_frames_dir.is_dir():
        print(f"[Analyze] Cảnh báo: Không tìm thấy thư mục frames cho '{movie_title}'. Trả về profile mặc định.")
        return {"duration": "Unknown", "lighting": "Unknown", "clarity": "Unknown", "complexity": "Unknown"}

    all_frame_paths = sorted([p for p in movie_frames_dir.glob("*.jpg")])
    n_extracted_frames = len(all_frame_paths)

    if n_extracted_frames == 0:
        print(f"[Analyze] Cảnh báo: Thư mục frames của '{movie_title}' rỗng. Trả về profile mặc định.")
        return {"duration": "Unknown", "lighting": "Unknown", "clarity": "Unknown", "complexity": "Unknown"}

    # --- 2. Logic lấy mẫu thích ứng (Adaptive Sampling 2.0 - Không thay đổi) ---
    if n_extracted_frames < 500:
        sample_size = min(max(int(n_extracted_frames * 0.2), 1), 70)
    elif 500 <= n_extracted_frames < 3000:
        sample_size = min(max(int(n_extracted_frames * 0.1), 50), 150)
    else:
        sample_size = min(max(int(n_extracted_frames * 0.05), 150), 250)

    # --- CẬP NHẬT: Thay thế lấy mẫu ngẫu nhiên bằng lấy mẫu có hệ thống ---
    # Logic cũ: sample_paths = random.sample(all_frame_paths, sample_size)

    if sample_size <= 0:
        sample_paths = []
    else:
        # Tính khoảng cách giữa các mẫu để rải đều trên toàn bộ video
        step = max(1, n_extracted_frames // sample_size)
        # Sử dụng slicing để lấy các phần tử cách đều nhau
        sample_paths = all_frame_paths[::step]

    print(f"[Analyze] Tổng số frames: {n_extracted_frames}. Lấy {len(sample_paths)} frames theo hệ thống để phân tích.")
    # --- KẾT THÚC CẬP NHẬT ---

    # --- 3. Phân tích các frame mẫu (Không thay đổi) ---
    brightness_scores, blur_scores, face_counts = [], [], []

    for frame_path in sample_paths:
        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness_scores.append(gray_img.mean())
            blur_scores.append(calculate_blur_score(img))
            face_counts.append(count_faces_retina(img))

        except Exception as e:
            print(f"[Analyze] Lỗi khi xử lý frame {frame_path.name}: {e}")
            continue

    if not brightness_scores:
        print("[Analyze] Cảnh báo: Không phân tích được frame nào. Trả về profile mặc định.")
        return {"duration": "Unknown", "lighting": "Unknown", "clarity": "Unknown", "complexity": "Unknown"}

    # --- 4. Tổng hợp và Phân loại (Không thay đổi) ---
    avg_brightness = np.mean(brightness_scores)
    avg_blur = np.mean(blur_scores)
    avg_faces = np.mean(face_counts)

    if n_extracted_frames < 600:
        duration_cat = "Short"
    elif 600 <= n_extracted_frames < 3000:
        duration_cat = "Medium"
    else:
        duration_cat = "Long"

    if avg_brightness < ANALYSIS_THRESHOLDS["LIGHTING_DARK"]:
        lighting_cat = "Dark"
    elif avg_brightness > ANALYSIS_THRESHOLDS["LIGHTING_BRIGHT"]:
        lighting_cat = "Bright"
    else:
        lighting_cat = "Normal"

    if avg_blur < ANALYSIS_THRESHOLDS["CLARITY_BLURRY"]:
        clarity_cat = "Blurry"
    elif avg_blur > ANALYSIS_THRESHOLDS["CLARITY_SHARP"]:
        clarity_cat = "Sharp"
    else:
        clarity_cat = "Normal"

    if avg_faces >= ANALYSIS_THRESHOLDS["COMPLEXITY_CROWDED"]:
        complexity_cat = "Crowded"
    else:
        complexity_cat = "Sparse"

    video_profile = {
        "duration": duration_cat,
        "lighting": lighting_cat,
        "clarity": clarity_cat,
        "complexity": complexity_cat
    }

    print(f"[Analyze] Hoàn thành phân tích. Video Profile: {video_profile}")

    return video_profile