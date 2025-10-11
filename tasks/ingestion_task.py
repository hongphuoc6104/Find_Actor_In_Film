# tasks/ingestion_task.py
import os
import json
from glob import glob
from typing import Dict, Any, Tuple, Optional

import cv2 as cv
from prefect import task

from utils.config_loader import load_config


# ========== RetinaFace via InsightFace ==========
# Ưu tiên CUDA nếu khả dụng; fallback CPU.
def _make_retinaface_app(det_size=(640, 640)):
    """
    Tạo FaceAnalysis app của InsightFace:
    - Ưu tiên CUDAExecutionProvider (onnxruntime-gpu)
    - Fallback CPUExecutionProvider nếu không có GPU
    """
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError(
            "Chưa cài insightface. Cài: pip install insightface onnxruntime-gpu"
        ) from e

    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    # Một số phiên bản insightface hỗ trợ truyền providers trực tiếp
    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
    except TypeError:
        # Phiên bản cũ hơn không nhận providers -> để insightface tự chọn
        app = FaceAnalysis(name="buffalo_l")

    # ctx_id = 0 -> GPU; nếu không có GPU, InsightFace sẽ dùng CPU
    try:
        app.prepare(ctx_id=0, det_size=det_size)
    except Exception:
        # ép CPU
        app.prepare(ctx_id=-1, det_size=det_size)
    return app


# Dò số khuôn mặt với RetinaFace (bbox + landmarks sẵn)
_retina_app = None
def count_faces_retina(frame) -> int:
    global _retina_app
    if _retina_app is None:
        # Có thể điều chỉnh kích thước detector qua config nếu muốn
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
    video_path: str,
    root_output_folder: str,
    *,
    min_faces: int = 2,
    jpeg_quality: int = 85,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Trích xuất khung hình và trả về (video_name, video_metadata).
    - Dùng RetinaFace để đếm số khuôn mặt.
    - Nếu ít mặt (< min_faces) -> tăng tần suất lấy mẫu (nhanh hơn).
    - Lưu JPEG chất lượng q=85 để giảm IO.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder_for_movie = os.path.join(root_output_folder, video_name)
    os.makedirs(output_folder_for_movie, exist_ok=True)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video '{video_path}'")
        return None, None

    # Metadata
    video_metadata = _extract_video_metadata(cap, video_path)
    fps = video_metadata["fps"] or 0.0
    default_interval = int(fps) if fps > 0 else 30
    fast_interval = max(1, default_interval // 2)
    current_interval = default_interval

    # OpenCV threading (giảm tranh chấp CPU nếu môi trường nhiều luồng)
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
        if not ret:
            break

        if frame_count % current_interval == 0:
            # Đếm khuôn mặt bằng RetinaFace (GPU nếu có)
            try:
                n_faces = count_faces_retina(frame)
            except Exception as e:
                # Nếu detector lỗi, fallback "không đếm" để không chặn pipeline
                print(f"[CẢNH BÁO] RetinaFace lỗi: {e}. Bỏ qua đếm, giữ interval mặc định.")
                n_faces = min_faces

            # Lưu frame
            frame_filename = os.path.join(
                output_folder_for_movie, f"frame_{frame_count:07d}.jpg"
            )
            cv.imwrite(frame_filename, frame, encode_params)
            saved_frame_count += 1

            # Điều chỉnh khoảng lấy mẫu cho khung kế tiếp
            current_interval = fast_interval if n_faces < min_faces else default_interval

        frame_count += 1

    cap.release()
    print(f"-> Hoàn tất! Đã lưu {saved_frame_count} khung hình tại '{output_folder_for_movie}'")
    return video_name, video_metadata


# THAY ĐỔI 1: Đổi tên task cho rõ ràng hơn và thêm tham số 'movie' vào hàm
@task(name="Ingestion Task: Extract Frames")
def ingestion_task(movie: Optional[str] = None):
    """
    Nhiệm vụ ingest:
    - Nếu tham số 'movie' được truyền vào -> chỉ xử lý đúng 1 phim đó
    - Ngược lại: giữ hành vi cũ (quét toàn bộ thư mục video_root)
    - Với mỗi video: nếu chưa trích thì trích khung + metadata
    - Cập nhật metadata_json
    """
    cfg = load_config()
    video_folder = cfg["storage"]["video_root"]
    frames_folder = cfg["storage"]["frames_root"]
    metadata_filepath = cfg["storage"]["metadata_json"]

    # THAY ĐỔI 2: Ưu tiên sử dụng tham số 'movie' được truyền vào
    active_movie = movie or (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        print(f"[Ingestion] Chế độ đơn-phim: chỉ xử lý '{active_movie}'")

    # đọc metadata cũ (nếu có)
    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    except FileNotFoundError:
        all_metadata = {}

    # tìm video (mp4, mkv, mov phổ biến)
    patterns = ["*.mp4", "*.MP4", "*.mkv", "*.MKV", "*.mov", "*.MOV", "*.avi", "*.AVI"]
    video_paths = []
    for pat in patterns:
        video_paths.extend(glob(os.path.join(video_folder, pat)))
    video_paths = sorted(set(video_paths))

    if not video_paths:
        print(f"Không tìm thấy video nào trong thư mục '{video_folder}'")
        return

    # Lọc theo active_movie nếu có
    if active_movie:
        filtered = []
        for p in video_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            # so khớp tên tuyệt đối (case-sensitive), nếu cần bạn đổi sang lower()
            if name == active_movie:
                filtered.append(p)
        if not filtered:
            print(f"[Ingestion] Không tìm thấy video tên '{active_movie}' trong '{video_folder}'. Bỏ qua ingestion.")
            return
        video_paths = filtered
        print(f"[Ingestion] Tìm thấy 1 video khớp: {os.path.basename(video_paths[0])}")
    else:
        print(f"Tìm thấy tổng cộng {len(video_paths)} video.")

    # tham số lấy từ config (tùy chọn)
    min_faces = int(cfg.get("ingestion", {}).get("min_faces_per_scene", 2))
    jpeg_quality = int(cfg.get("ingestion", {}).get("jpeg_quality", 85))

    new_videos_processed = False
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder_for_movie = os.path.join(frames_folder, video_name)

        # Bỏ qua nếu đã có frames
        if os.path.isdir(output_folder_for_movie) and len(os.listdir(output_folder_for_movie)) > 0:
            print(f"Video '{video_name}' có vẻ đã được trích xuất frames trước đó. Bỏ qua.")
            if video_name not in all_metadata:
                # tái tạo metadata nếu thiếu
                cap = cv.VideoCapture(video_path)
                if cap.isOpened():
                    all_metadata[video_name] = _extract_video_metadata(cap, video_path)
                    new_videos_processed = True
                cap.release()
            continue

        # Trích mới
        new_videos_processed = True
        name, info = extract_frames_with_adaptive_sampling(
            video_path,
            frames_folder,
            min_faces=min_faces,
            jpeg_quality=jpeg_quality,
        )
        if name is not None and info is not None:
            all_metadata[name] = info

    # ghi metadata nếu có thay đổi
    if new_videos_processed:
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Đã cập nhật thành công file metadata tại '{metadata_filepath}'")
    else:
        print("\nKhông có video mới nào để xử lý.")


if __name__ == "__main__":
    ingestion_task()