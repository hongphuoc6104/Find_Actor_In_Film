# services/recognition.py
from __future__ import annotations
import os
import math  # Thêm import math
from typing import Any, Dict, List, Optional

# --- Các thành phần cốt lõi ---
from utils.config_loader import load_config, get_recognition_settings
from utils.indexer import build_character_index, search_by_embedding
from utils.search_actor import _get_app, _read_image, _detect_best_face

DEBUG = os.getenv("FS_DEBUG", "1") != "0"


# --- SỬA LỖI: Đưa các hàm helper vào đúng vị trí ---
def _as_float(x: Any, default: float = 0.0) -> float:
    """Chuyển đổi giá trị sang float một cách an toàn."""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (ValueError, TypeError, AttributeError):
        pass
    return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    """Chuyển đổi giá trị sang int một cách an toàn."""
    try:
        # Chuyển qua float trước để xử lý các chuỗi như "123.0"
        return int(float(x))
    except (ValueError, TypeError, AttributeError):
        return int(default)


# --- Import các hàm còn lại từ scene_loader ---
from .scene_loader import _read_metadata


def _ensure_scenes(char_entry: Dict[str, Any], max_scenes: int = 8) -> None:
    sc = char_entry.get("scenes")
    if isinstance(sc, list) and sc:
        char_entry["scenes"] = sc[:max_scenes]


def recognize(image_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    Hàm nhận diện chính, sử dụng indexer làm phương pháp tìm kiếm chính.
    """
    cfg = load_config()
    recognition_cfg = get_recognition_settings(cfg)

    # Tải các ngưỡng từ config
    present_threshold = _as_float(recognition_cfg.get("present_threshold", 0.55), 0.55)
    near_match_threshold = _as_float(recognition_cfg.get("SIM_THRESHOLD", 0.45), 0.45)

    # Đảm bảo index được xây dựng trước khi tìm kiếm
    try:
        build_character_index(force_rebuild=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"[Recognize][FATAL] Không thể xây dựng index: {e}")
        return {"is_unknown": True, "movies": [], "error": str(e)}

    # Luồng xử lý mới
    # 1. Trích xuất embedding từ ảnh truy vấn
    app = _get_app()
    query_image = _read_image(image_path)
    if query_image is None:
        return {"is_unknown": True, "movies": [], "error": "Không thể đọc file ảnh."}

    query_embedding = _detect_best_face(app, query_image)
    if query_embedding is None:
        if DEBUG: print("[Recognize] Không tìm thấy khuôn mặt trong ảnh truy vấn.")
        return {"is_unknown": True, "movies": [], "error": "Không tìm thấy khuôn mặt."}

    # 2. Thực hiện tìm kiếm bằng indexer
    max_results = _as_int((cfg.get("search") or {}).get("max_results", 20), 20)
    if top_k: max_results = max(max_results, int(top_k))

    raw_matches = search_by_embedding(
        query_vec=query_embedding,
        top_k=max_results,
        min_score=near_match_threshold,  # Lọc sơ bộ ở bước tìm kiếm
    )

    if DEBUG:
        best_score = raw_matches[0]['score'] if raw_matches else 0.0
        print(f"[Recognize][RAW] Indexer trả về {len(raw_matches)} kết quả, best_score={best_score:.4f}")

    if not raw_matches:
        return {"is_unknown": True, "movies": []}

    # 3. Gom nhóm kết quả theo phim và định dạng đầu ra
    matches_by_movie: Dict[str, List[Dict[str, Any]]] = {}
    for match in raw_matches:
        movie_title = match.get("movie", "Unknown Movie")
        matches_by_movie.setdefault(movie_title, []).append(match)

    output_movies: List[Dict[str, Any]] = []
    for movie_title, candidates in matches_by_movie.items():
        kept_chars: List[Dict[str, Any]] = []
        for c in candidates:
            score = _as_float(c.get("score"), 0.0)

            # Phân loại match_status
            status = "present" if score >= present_threshold else "near_match"
            label = "Xuất hiện" if status == "present" else "Gần giống"

            ent = {
                "character_id": str(c.get("character_id")),
                "score": score,
                "rep_image": c.get("rep_image"),
                "preview_paths": c.get("preview_paths") or [],
                "scenes": c.get("scenes") or [],
                "match_status": status,
                "match_label": label,
            }
            _ensure_scenes(ent, max_scenes=_as_int((cfg.get("search") or {}).get("max_scenes", 8), 8))
            kept_chars.append(ent)

        if kept_chars:
            output_movies.append({
                "movie": movie_title,
                "characters": kept_chars,
            })

    # Sắp xếp phim theo điểm số cao nhất
    output_movies.sort(key=lambda m: max([c.get('score', 0.0) for c in m['characters']]), reverse=True)

    return {
        "is_unknown": len(output_movies) == 0,
        "movies": output_movies,
    }