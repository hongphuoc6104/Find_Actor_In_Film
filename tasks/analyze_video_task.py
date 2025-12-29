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
# [TUNING] CÁC NGƯỠNG PHÂN TÍCH ĐÃ ĐƯỢC TINH CHỈNH
# =====================================================================
ANALYSIS_THRESHOLDS = {
    # 1. Ánh sáng (0-255):
    # - Phim kinh dị/hành động đêm thường < 70
    # - Sitcom/Talkshow studio thường > 130
    "LIGHTING_DARK": 80,      # Đã nâng nhẹ ngưỡng để bắt được nhiều phim "hơi tối"
    "LIGHTING_BRIGHT": 140,   # Giảm nhẹ để dễ rơi vào nhóm Bright (studio)

    # 2. Độ nét (Variance of Laplacian):
    # - Action cam rung lắc mạnh thường < 150
    # - Phim 4K tĩnh, phỏng vấn thường > 300
    "CLARITY_BLURRY": 150,    # Ngưỡng mờ do chuyển động (motion blur)
    "CLARITY_SHARP": 400,     # Ngưỡng cực nét (thường là quay cận mặt tĩnh)

    # 3. Độ phức tạp (Số mặt trung bình/frame):
    # - Phim tâm lý 2 người nói chuyện ~ 1.0 - 1.5
    # - Phim chiến tranh/hành động > 2.5
    "COMPLEXITY_CROWDED": 2.2 # Giảm ngưỡng xuống để thắt chặt tiêu chuẩn cho phim đông người
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

    if sample_size <= 0:
        sample_paths = []
    else:
        # Tính khoảng cách giữa các mẫu để rải đều trên toàn bộ video
        step = max(1, n_extracted_frames // sample_size)
        # Sử dụng slicing để lấy các phần tử cách đều nhau
        sample_paths = all_frame_paths[::step]

    print(f"[Analyze] Tổng số frames: {n_extracted_frames}. Lấy {len(sample_paths)} frames theo hệ thống để phân tích.")

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

    # --- 4. Tổng hợp và Phân loại (Logic giữ nguyên, chỉ phụ thuộc vào Thresholds ở trên) ---
    avg_brightness = np.mean(brightness_scores)
    avg_blur = np.mean(blur_scores)
    avg_faces = np.mean(face_counts)

    # Thêm log chi tiết để debug xem phim rơi vào khoảng nào
    print(f"[Analyze] Metrics Raw -> Brightness: {avg_brightness:.1f}, Blur: {avg_blur:.1f}, AvgFaces: {avg_faces:.2f}")

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