# tasks/ingestion_task.py
import os
import json
from glob import glob
from typing import Dict, Any, Tuple, Optional

import cv2 as cv
from prefect import task

from utils.config_loader import load_config


# ========== RetinaFace via InsightFace ==========
# (Các hàm helper từ _make_retinaface_app đến extract_frames_with_adaptive_sampling giữ nguyên, không thay đổi)
def _make_retinaface_app(det_size=(640, 640)):
    """
    Tạo FaceAnalysis app của InsightFace.
    """
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError(
            "Chưa cài insightface. Cài: pip install insightface onnxruntime-gpu"
        ) from e

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
    except TypeError:
        app = FaceAnalysis(name="buffalo_l")
    try:
        app.prepare(ctx_id=0, det_size=det_size)
    except Exception:
        app.prepare(ctx_id=-1, det_size=det_size)
    return app


_retina_app = None


def count_faces_retina(frame) -> int:
    global _retina_app
    if _retina_app is None:
        _retina_app = _make_retinaface_app(det_size=(640, 640))
    faces = _retina_app.get(frame) or []
    return len(faces)


def _extract_video_metadata(cap: cv.VideoCapture, video_path: str) -> Dict[str, Any]:
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration_seconds = (total_frames / fps) if fps and fps > 0 else 0.0
    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": round(float(duration_seconds), 2),
    }


def extract_frames_with_adaptive_sampling(
        video_path: str, root_output_folder: str, *, min_faces: int = 2, jpeg_quality: int = 85,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Trích xuất frames từ video.
    Mặc định: 1 frame/giây (interval = FPS của video)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder_for_movie = os.path.join(root_output_folder, video_name)
    os.makedirs(output_folder_for_movie, exist_ok=True)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video '{video_path}'")
        return None, None

    video_metadata = _extract_video_metadata(cap, video_path)
    fps = video_metadata["fps"] or 0.0
    
    # Cố định lấy 1 frame/giây (interval = FPS)
    frame_interval = int(fps) if fps > 0 else 24
    
    print(f"[Ingestion] FPS={fps:.1f}, interval={frame_interval} (1 frame/giây)")

    try:
        cv.setNumThreads(max(1, (os.cpu_count() or 4) // 2))
    except Exception:
        pass

    frame_count = 0
    saved_frame_count = 0
    encode_params = [cv.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    print(f"Bắt đầu trích xuất khung hình cho '{video_name}'...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder_for_movie, f"frame_{frame_count:07d}.jpg")
            cv.imwrite(frame_filename, frame, encode_params)
            saved_frame_count += 1
        frame_count += 1
    cap.release()
    print(f"-> Hoàn tất! Đã lưu {saved_frame_count} khung hình tại '{output_folder_for_movie}'")
    return video_name, video_metadata


# =====================================================================
# TASK CHÍNH (Đã sửa lỗi đồng bộ)
# =====================================================================

@task(name="Ingestion Task: Extract Frames")
def ingestion_task(movie: Optional[str] = None) -> bool:
    """
    Nhiệm vụ ingest:
    - Tìm video tương ứng với `movie`.
    - Trích xuất frames nếu cần thiết.
    - Trả về True nếu video được tìm thấy (dù có trích xuất mới hay không).
    - Trả về False nếu không tìm thấy video nào khớp.
    """
    cfg = load_config()
    video_folder = cfg["storage"]["video_root"]
    frames_folder = cfg["storage"]["frames_root"]
    metadata_filepath = cfg["storage"]["metadata_json"]

    active_movie = movie or (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if not active_movie:
        print("[Ingestion] Cảnh báo: Không có tên phim nào được chỉ định.")
        return False

    print(f"[Ingestion] Chế độ đơn-phim: chỉ xử lý '{active_movie}'")

    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    except FileNotFoundError:
        all_metadata = {}

    patterns = ["*.mp4", "*.MP4", "*.mkv", "*.MKV", "*.mov", "*.MOV", "*.avi", "*.AVI"]
    video_paths = []
    for pat in patterns:
        video_paths.extend(glob(os.path.join(video_folder, pat)))
    video_paths = sorted(set(video_paths))

    if not video_paths:
        print(f"Không tìm thấy video nào trong thư mục '{video_folder}'")
        return False  # --- CẬP NHẬT 1: Trả về False nếu thư mục video rỗng

    # Lọc chính xác video đang cần xử lý
    filtered_paths = [p for p in video_paths if os.path.splitext(os.path.basename(p))[0] == active_movie]

    if not filtered_paths:
        print(f"[Ingestion] Không tìm thấy video tên '{active_movie}' trong '{video_folder}'. Bỏ qua ingestion.")
        return False  # --- CẬP NHẬT 2: Trả về False nếu phim cụ thể không tồn tại

    print(f"[Ingestion] Tìm thấy 1 video khớp: {os.path.basename(filtered_paths[0])}")

    # Từ đây trở đi, chúng ta chắc chắn đã tìm thấy video, nên cuối cùng sẽ trả về True
    video_paths = filtered_paths

    min_faces = int(cfg.get("ingestion", {}).get("min_faces_per_scene", 2))
    jpeg_quality = int(cfg.get("ingestion", {}).get("jpeg_quality", 85))

    new_videos_processed = False
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder_for_movie = os.path.join(frames_folder, video_name)

        if os.path.isdir(output_folder_for_movie) and len(os.listdir(output_folder_for_movie)) > 0:
            print(f"Video '{video_name}' có vẻ đã được trích xuất frames trước đó. Bỏ qua.")
            if video_name not in all_metadata:
                cap = cv.VideoCapture(video_path)
                if cap.isOpened():
                    all_metadata[video_name] = _extract_video_metadata(cap, video_path)
                    new_videos_processed = True
                cap.release()
            continue

        new_videos_processed = True
        name, info = extract_frames_with_adaptive_sampling(
            video_path, frames_folder, min_faces=min_faces, jpeg_quality=jpeg_quality,
        )
        if name is not None and info is not None:
            all_metadata[name] = info

    if new_videos_processed:
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Đã cập nhật thành công file metadata tại '{metadata_filepath}'")
    else:
        print("\nKhông có video mới nào để xử lý.")

    return True  # --- CẬP NHẬT 3: Trả về True vì đã tìm thấy video


if __name__ == "__main__":
    ingestion_task()