# services/recognition.py
from __future__ import annotations
import os
import math
from typing import Any, Dict, List, Optional

from utils.config_loader import load_config, get_recognition_settings
from utils.indexer import build_character_index, search_by_embedding
from utils.search_actor import _get_app, _read_image, _detect_best_face

DEBUG = os.getenv("FS_DEBUG", "1") != "0"


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except (ValueError, TypeError, AttributeError):
        return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except (ValueError, TypeError, AttributeError):
        return int(default)


def recognize(image_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    Hàm nhận diện chính, được nâng cấp với logic quyết định thông minh.
    """
    cfg = load_config()
    recognition_cfg = get_recognition_settings(cfg)
    search_cfg = cfg.get("search", {})

    # Tải các ngưỡng quyết định từ config (vay mượn từ find_actor.py)
    confident_threshold = _as_float(search_cfg.get("confident_threshold", 0.55), 0.55)
    suggestion_threshold = _as_float(recognition_cfg.get("SIM_THRESHOLD", 0.45), 0.45)
    margin_threshold = _as_float(search_cfg.get("margin_threshold", 0.1), 0.1)

    try:
        build_character_index()
    except Exception as e:
        print(f"[Recognize][FATAL] Không thể xây dựng index: {e}")
        return {"is_unknown": True, "movies": [], "error": str(e)}

    app = _get_app()
    query_image = _read_image(image_path)
    if query_image is None:
        return {"is_unknown": True, "movies": [], "error": "Không thể đọc file ảnh."}

    query_embedding = _detect_best_face(app, query_image)
    if query_embedding is None:
        return {"is_unknown": True, "movies": [], "error": "Không tìm thấy khuôn mặt."}

    max_results = _as_int(search_cfg.get("max_results", 20), 20)
    raw_matches = search_by_embedding(
        query_vec=query_embedding,
        top_k=max_results,
        min_score=suggestion_threshold,
    )

    if not raw_matches:
        return {"is_unknown": True, "movies": []}

    matches_by_movie: Dict[str, List[Dict[str, Any]]] = {}
    for match in raw_matches:
        movie_title = match.get("movie", "Unknown Movie")
        matches_by_movie.setdefault(movie_title, []).append(match)

    output_movies: List[Dict[str, Any]] = []
    for movie_title, candidates in matches_by_movie.items():
        if not candidates: continue

        # --- LOGIC QUYẾT ĐỊNH MỚI ---
        top_1_score = _as_float(candidates[0].get("score"))
        top_2_score = _as_float(candidates[1].get("score")) if len(candidates) > 1 else 0.0

        # Sắp xếp lại các cảnh theo thời gian bắt đầu
        for cand in candidates:
            if isinstance(cand.get("scenes"), list):
                cand["scenes"].sort(key=lambda s: s.get("start_time", 0))

        # Phân loại kết quả
        final_characters = []
        if top_1_score >= confident_threshold and (top_1_score - top_2_score) >= margin_threshold:
            # Trường hợp 1: Rất chắc chắn - điểm cao và bỏ xa đối thủ
            char = candidates[0]
            char["match_status"] = "CONFIDENT"
            char["match_label"] = "Khớp chính xác"
            final_characters.append(char)
        else:
            # Trường hợp 2: Gợi ý - điểm trên ngưỡng nhưng không chắc chắn, hoặc có đối thủ bám sát
            for char in candidates:
                score = _as_float(char.get("score"))
                if score >= suggestion_threshold:
                    char["match_status"] = "SUGGESTION"
                    char["match_label"] = "Gợi ý"
                    final_characters.append(char)

        if final_characters:
            output_movies.append({
                "movie": movie_title,
                "characters": final_characters,
            })

    output_movies.sort(key=lambda m: max([c.get('score', 0.0) for c in m['characters']]), reverse=True)

    return {
        "is_unknown": len(output_movies) == 0,
        "movies": output_movies,
    }