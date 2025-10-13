# utils/config_loader.py
"""
Bộ nạp cấu hình trung tâm + preset cho pipeline.

Features:
  - load_config(): Đọc YAML chính (configs/config.yaml)
  - apply_preset(): Merge profiles + ENV overrides
  - load_movie_metadata(): Đọc metadata.json cho từng phim
  - deep_merge(): Merge nested dicts
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CFG_PATH = Path("configs/config.yaml")


# =============================================================================
# YAML loader
# =============================================================================

def _read_yaml(path: Path) -> Dict[str, Any]:
    """Đọc YAML file với error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Đọc file YAML cấu hình chính và đảm bảo các nhánh cần thiết tồn tại.
    """
    cfg_path = Path(path) if path else DEFAULT_CFG_PATH
    cfg = _read_yaml(cfg_path)

    # Đảm bảo tồn tại các nhóm chính để tránh KeyError
    defaults = {
        "storage": {},
        "embedding": {},
        "pca": {},
        "clustering": {},
        "cluster": {},  # alias
        "merge": {},
        "post_merge": {},
        "filter": {},
        "filter_clusters": {},
        "quality_filters": {},
        "centroid": {},
        "index": {},
        "search": {},
        "highlight": {},
        "recognition": {},
        "frontend": {},
        "preview": {"source": "frames", "max_images_per_cluster": 24},
        "tracklet": {"max_age": 3, "iou_threshold": 0.28},
        "profiles": {},  # quan trọng!
    }

    for key, default_val in defaults.items():
        cfg.setdefault(key, default_val)

    return cfg


# =============================================================================
# Deep merge utility
# =============================================================================

def deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge nhiều dict từ trái sang phải.
    Dict bên phải ghi đè dict bên trái.
    """
    result = {}

    for d in dicts:
        if not isinstance(d, dict):
            continue

        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

    return result


# =============================================================================
# Movie metadata loader
# =============================================================================

def load_movie_metadata(movie_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load metadata cho movie từ metadata.json.
    """
    meta_path = Path(cfg["storage"]["metadata_json"])

    if not meta_path.exists():
        return {"era": None, "genre": None, "context_tags": [], "custom_knobs": {}}

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            all_meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"era": None, "genre": None, "context_tags": [], "custom_knobs": {}}

    movie_meta = all_meta.get(movie_name, {})

    return {
        "era": movie_meta.get("era"),
        "genre": movie_meta.get("genre"),
        "context_tags": movie_meta.get("context_tags", []),
        "custom_knobs": movie_meta.get("custom_knobs", {}),
    }


# =============================================================================
# ENV override helpers
# =============================================================================
# ... (các hàm _env_float, _env_int giữ nguyên) ...
def _env_float(key: str, default: Optional[float]) -> Optional[float]:
    v = os.getenv(key)
    if v is None or v == "": return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(key: str, default: Optional[int]) -> Optional[int]:
    v = os.getenv(key)
    if v is None or v == "": return default
    try:
        return int(float(v))
    except Exception:
        return default


# =============================================================================
# Profile selection logic
# =============================================================================
# ... (hàm _choose_profile_key giữ nguyên) ...
def _choose_profile_key(era: Optional[str], genre: Optional[str], context_tags: Optional[List[str]]) -> str:
    era_key = (era or "").strip().lower()
    genre_key = (genre or "").strip().lower()
    tags = [t.strip().lower() for t in (context_tags or []) if t and t.strip()]
    if era_key in {"co_trang", "xua"} and "dong_duc" in tags: return "co_trang_dong_duc"
    if era_key == "hien_dai" and genre_key in {"tinh_cam", "tam_ly", "drama"}: return "hien_dai_tinh_cam"
    if genre_key in {"hanh_dong", "action"} and any(
        t in tags for t in ["dong_duc", "rung_lac"]): return "hanh_dong_rung_lac"
    if era_key in {"xua", "lang_que"}: return "xua_lang_que"
    if genre_key in {"kinh_di", "horror"} and "toi" in tags: return "kinh_di_toi"
    return "balanced"


# =============================================================================
# Main preset applier
# =============================================================================
# ... (hàm apply_preset giữ nguyên) ...
def apply_preset(cfg: Dict[str, Any], era: Optional[str] = None, genre: Optional[str] = None,
                 context_tags: Optional[List[str]] = None, profile: Optional[str] = None,
                 custom_knobs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    if profile:
        profile_key = profile
    else:
        profile_key = _choose_profile_key(era, genre, context_tags)
    profiles = cfg.get("profiles", {})
    profile_cfg = profiles.get(profile_key, {})
    if not profile_cfg:
        profile_cfg = profiles.get("balanced", {})
        if not profile_cfg:
            profile_key = "default"
            profile_cfg = {}
    merged = deep_merge(cfg, profile_cfg)
    if custom_knobs: merged = deep_merge(merged, custom_knobs)
    env_overrides = {}
    within = _env_float("FS_WITHIN", None)
    if within is not None: env_overrides.setdefault("merge", {})["within_movie_threshold"] = within
    min_size = _env_int("FS_MIN_SIZE", None)
    if min_size is not None: env_overrides.setdefault("filter_clusters", {})["min_size"] = min_size
    dist_pca = _env_float("FS_DIST_PCA", None)
    if dist_pca is not None: env_overrides.setdefault("cluster", {})["distance_threshold_pca"] = dist_pca
    max_age = _env_int("FS_MAX_AGE", None)
    if max_age is not None: env_overrides.setdefault("tracklet", {})["max_age"] = max_age
    iou = _env_float("FS_IOU", None)
    if iou is not None: env_overrides.setdefault("tracklet", {})["iou_threshold"] = iou
    if env_overrides: merged = deep_merge(merged, env_overrides)
    return profile_key, merged


# =============================================================================
# Backward compatibility
# =============================================================================
# ... (các hàm extract_knobs, get_recognition_settings giữ nguyên) ...
def extract_knobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "within": cfg.get("merge", {}).get("within_movie_threshold"),
        "min_size": cfg.get("filter_clusters", {}).get("min_size"),
        "dist_pca": cfg.get("cluster", {}).get("distance_threshold_pca"),
        "max_age": cfg.get("tracklet", {}).get("max_age"),
        "iou": cfg.get("tracklet", {}).get("iou_threshold"),
    }


def get_recognition_settings(cfg: dict) -> dict:
    if not isinstance(cfg, dict): return {"SIM_THRESHOLD": 0.3}
    recog = (cfg.get("recognition") or {}).copy()
    sim = recog.get("SIM_THRESHOLD")
    if sim is None: sim = (cfg.get("search") or {}).get("near_match_threshold")
    try:
        sim = float(sim)
    except Exception:
        sim = 0.3
    recog["SIM_THRESHOLD"] = sim
    return recog


# =============================================================================
# --- THÊM MỚI: CÁC HÀM TỰ ĐỘNG TINH CHỈNH ---
# =============================================================================

def get_video_duration(movie_title: str, cfg: Dict[str, Any]) -> Optional[float]:
    """
    Đọc metadata.json và trả về duration_seconds của một phim cụ thể.
    Hàm này đọc thông tin do ingestion_task.py tạo ra.
    """
    meta_path = Path(cfg["storage"]["metadata_json"])
    if not meta_path.exists():
        print(f"[AutoTuning] Cảnh báo: không tìm thấy file metadata tại {meta_path}")
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            all_meta = json.load(f)

        movie_meta = all_meta.get(movie_title, {})
        duration = movie_meta.get("duration_seconds")

        return float(duration) if duration is not None else None
    except Exception as e:
        print(f"[AutoTuning] Lỗi khi đọc duration cho phim '{movie_title}': {e}")
        return None


def determine_min_size_thresholds(duration_seconds: Optional[float]) -> Dict[str, int]:
    """
    Dựa vào độ dài video (giây), quyết định các ngưỡng min_size.
    Trả về một dictionary chứa các ngưỡng cho "main_only" (chỉ nhân vật chính)
    và "all" (tất cả nhân vật).
    """
    # Cung cấp giá trị mặc định an toàn nếu không có thông tin duration
    if duration_seconds is None:
        print("[AutoTuning] Không có thông tin duration, sử dụng ngưỡng mặc định cho phim trung bình.")
        return {"main_only": 7, "all": 3}

    # Video rất ngắn (< 10 phút)
    if duration_seconds < 600:
        return {"main_only": 3, "all": 2}

    # Video trung bình (10 - 50 phút)
    elif 600 <= duration_seconds < 3000:
        return {"main_only": 7, "all": 3}

    # Video dài (> 50 phút)
    else:
        return {"main_only": 15, "all": 5}