# utils/search_actor.py
"""
Tìm diễn viên/nhân vật dựa trên ảnh truy vấn bằng cách:
- Đọc warehouse/characters.json
- Lấy ảnh đại diện (rep_image) của từng cụm làm prototype
- Tính embedding bằng InsightFace
- So cosine similarity với ảnh truy vấn
- Trả về danh sách theo từng phim ở format BE đang dùng

Không phụ thuộc vào parquet hay PCA.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from utils.config_loader import load_config

# -------------------------------
# Model loader (cache toàn cục)
# -------------------------------
_APP: Optional[FaceAnalysis] = None


def _get_app() -> FaceAnalysis:
    global _APP
    if _APP is None:
        app = FaceAnalysis(name="buffalo_l")  # mặc định 512D
        # auto GPU nếu có, fallback CPU
        # Use 416x416 for better small face detection
        try:
            app.prepare(ctx_id=0, det_size=(416, 416))
        except Exception:
            app.prepare(ctx_id=-1, det_size=(416, 416))
        _APP = app
    return _APP


# -------------------------------
# Helpers
# -------------------------------
def _load_json(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    if v is None:
        return v
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        return v
    return (v / n).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: (D,)
    if a is None or b is None:
        return -1.0
    aa = _l2_normalize(a)
    bb = _l2_normalize(b)
    s = float(np.dot(aa, bb))
    # clamp đề phòng nhiễu float
    if s > 1.0:
        s = 1.0
    if s < -1.0:
        s = -1.0
    return s


def _read_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FaceNotFoundError(Exception):
    """Raised when no valid human face is detected in the image"""
    pass


def _detect_best_face(app: FaceAnalysis, img: np.ndarray, min_det_score: float = 0.5) -> Optional[np.ndarray]:
    """
    Trả về embedding tốt nhất trong ảnh (chọn khuôn mặt có score cao nhất).
    Raises FaceNotFoundError nếu không tìm thấy khuôn mặt hợp lệ.
    """
    faces = app.get(img)
    if not faces:
        raise FaceNotFoundError("Không tìm thấy khuôn mặt trong ảnh. Vui lòng tải ảnh có khuôn mặt rõ ràng.")
    
    # chọn face có det_score cao nhất
    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    det_score = float(getattr(best, "det_score", 0.0))
    
    # Check if detection score is too low (likely not a human face)
    if det_score < min_det_score:
        raise FaceNotFoundError(f"Không phát hiện khuôn mặt người hợp lệ (score: {det_score:.2f}). Vui lòng tải ảnh khuôn mặt rõ ràng hơn.")
    
    # best.normed_embedding là 512D đã chuẩn hóa, nếu không có thì dùng embedding thường
    if hasattr(best, "normed_embedding") and best.normed_embedding is not None:
        return np.array(best.normed_embedding, dtype=np.float32)
    emb = getattr(best, "embedding", None)
    if emb is None:
        return None
    return _l2_normalize(np.array(emb, dtype=np.float32))


# -------------------------------
# Characters DB (từ characters.json)
# -------------------------------
@dataclass
class CharacterProto:
    movie: str
    character_id: str
    rep_image: str
    preview_paths: List[str]


@lru_cache(maxsize=1)
def _load_characters_json() -> Tuple[List[CharacterProto], Dict[str, Dict[str, Any]]]:
    """
    Trả về:
      - danh sách CharacterProto (movie, character_id, rep_image, preview_paths)
      - raw dict để tra cứu scenes nếu cần
    """
    cfg = load_config()
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    path = storage.get("characters_json") or "warehouse/characters.json"

    data = _load_json(path)
    protos: List[CharacterProto] = []
    if isinstance(data, dict):
        for movie_title, chars in data.items():
            if not isinstance(chars, dict):
                continue
            for cid, entry in chars.items():
                rep = entry.get("rep_image") or ""
                previews = entry.get("preview_paths") or []
                protos.append(
                    CharacterProto(
                        movie=str(movie_title),
                        character_id=str(cid),
                        rep_image=str(rep),
                        preview_paths=[str(p) for p in previews],
                    )
                )
    return protos, (data if isinstance(data, dict) else {})


# -------------------------------
# Embedding cache cho prototypes
# -------------------------------
# cache theo (đường dẫn ảnh, mtime) để tránh phải tái tính nhiều lần
_PROTO_EMB_CACHE: Dict[Tuple[str, float], np.ndarray] = {}


def _embed_proto_image(app: FaceAnalysis, image_path: str) -> Optional[np.ndarray]:
    try:
        mtime = os.path.getmtime(image_path)
    except Exception:
        mtime = -1.0
    key = (image_path, mtime)
    if key in _PROTO_EMB_CACHE:
        return _PROTO_EMB_CACHE[key]

    img = _read_image(image_path)
    if img is None:
        return None
    emb = _detect_best_face(app, img)
    if emb is not None:
        _PROTO_EMB_CACHE[key] = emb
    return emb


# -------------------------------
# Public API
# -------------------------------
def search_actor(
    image_path: str,
    *,
    max_results: int = 50,
    score_floor: float = 0.30,
    min_count: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Nhận 1 ảnh truy vấn → trả về dict:
    {
      "<movie_title>": [
         {
           "character_id": "...",
           "distance": <similarity 0..1>,
           "rep_image": "...",
           "preview_paths": [...],
         },
         ...
      ],
      ...
    }
    Chỉ giữ các kết quả có similarity >= score_floor.
    
    Raises:
        FaceNotFoundError: Nếu không tìm thấy khuôn mặt người hợp lệ trong ảnh
    """
    app = _get_app()

    # 1) embedding ảnh truy vấn
    q_img = _read_image(image_path)
    if q_img is None:
        raise FaceNotFoundError("Không thể đọc file ảnh. Vui lòng kiểm tra lại.")
    
    # This will raise FaceNotFoundError if no valid face found
    q_emb = _detect_best_face(app, q_img)
    if q_emb is None:
        raise FaceNotFoundError("Không thể tạo embedding từ ảnh.")

    # 2) load prototypes từ characters.json
    protos, raw_chars = _load_characters_json()
    if not protos:
        return {}

    # 3) tính sim và gom theo movie
    by_movie: Dict[str, List[Dict[str, Any]]] = {}
    for proto in protos:
        emb = _embed_proto_image(app, proto.rep_image)
        if emb is None:
            continue
        sim = _cosine_sim(q_emb, emb)
        if sim < float(score_floor):
            continue
        rec = {
            "character_id": proto.character_id,
            "distance": float(sim),  # FE/BE hiện đang dùng 'distance' = similarity
            "rep_image": proto.rep_image,
            "preview_paths": proto.preview_paths,
        }
        by_movie.setdefault(proto.movie, []).append(rec)

    # 4) sort và cắt top theo từng phim
    for mv, items in by_movie.items():
        items.sort(key=lambda d: float(d.get("distance", 0.0)), reverse=True)
        if max_results and max_results > 0:
            by_movie[mv] = items[: int(max_results)]

    # 5) bỏ phim không đủ kết quả
    if min_count > 1:
        by_movie = {k: v for k, v in by_movie.items() if len(v) >= min_count}

    return by_movie
