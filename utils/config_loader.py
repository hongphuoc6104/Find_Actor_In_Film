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

    Example:
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"c": 3, "d": 4}}
        result = deep_merge(base, override)
        # → {"a": {"b": 1, "c": 3, "d": 4}}
    """
    result = {}

    for d in dicts:
        if not isinstance(d, dict):
            continue

        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Merge nested dict
                result[key] = deep_merge(result[key], value)
            else:
                # Override trực tiếp
                result[key] = deepcopy(value)

    return result


# =============================================================================
# Movie metadata loader
# =============================================================================

def load_movie_metadata(movie_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load metadata cho movie từ metadata.json.

    Returns
    -------
    dict với keys:
        - era: str | None
        - genre: str | None
        - context_tags: List[str]
        - custom_knobs: Dict[str, Any]
    """
    meta_path = Path(cfg["storage"]["metadata_json"])

    if not meta_path.exists():
        return {
            "era": None,
            "genre": None,
            "context_tags": [],
            "custom_knobs": {},
        }

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            all_meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "era": None,
            "genre": None,
            "context_tags": [],
            "custom_knobs": {},
        }

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

def _env_float(key: str, default: Optional[float]) -> Optional[float]:
    """Read float from ENV, fallback to default."""
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(key: str, default: Optional[int]) -> Optional[int]:
    """Read int from ENV, fallback to default."""
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


# =============================================================================
# Profile selection logic
# =============================================================================

def _choose_profile_key(
        era: Optional[str],
        genre: Optional[str],
        context_tags: Optional[List[str]],
) -> str:
    """
    Chọn profile key dựa trên metadata.

    Logic:
      1. Ưu tiên context_tags đặc thù (dong_duc, toi, ngoai_troi)
      2. Kết hợp với era để tạo key
      3. Fallback "balanced" nếu không match

    Examples:
      era=co_trang + context_tags=[dong_duc] → "co_trang_dong_duc"
      era=hien_dai + genre=tinh_cam → "hien_dai_tinh_cam"
      era=xua → "xua_lang_que"
    """
    era_key = (era or "").strip().lower()
    genre_key = (genre or "").strip().lower()
    tags = [t.strip().lower() for t in (context_tags or []) if t and t.strip()]

    # Mapping rules (customize theo business logic)

    # 1) Co trang + dong duc
    if era_key in {"co_trang", "xua"} and "dong_duc" in tags:
        return "co_trang_dong_duc"

    # 2) Hien dai + tinh cam (drama)
    if era_key == "hien_dai" and genre_key in {"tinh_cam", "tam_ly", "drama"}:
        return "hien_dai_tinh_cam"

    # 3) Hanh dong + dong duc / ngoai troi
    if genre_key in {"hanh_dong", "action"} and any(t in tags for t in ["dong_duc", "rung_lac"]):
        return "hanh_dong_rung_lac"

    # 4) Xua + lang que
    if era_key in {"xua", "lang_que"}:
        return "xua_lang_que"

    # 5) Kinh di + toi
    if genre_key in {"kinh_di", "horror"} and "toi" in tags:
        return "kinh_di_toi"

    # 6) Fallback balanced
    return "balanced"


# =============================================================================
# Main preset applier
# =============================================================================

def apply_preset(
        cfg: Dict[str, Any],
        era: Optional[str] = None,
        genre: Optional[str] = None,
        context_tags: Optional[List[str]] = None,
        profile: Optional[str] = None,
        custom_knobs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Apply preset profile và merge vào config.

    Priority (cao → thấp):
      1. ENV variables (FS_WITHIN, FS_MIN_SIZE, ...)
      2. custom_knobs (per-movie overrides trong metadata.json)
      3. profile (manual selection hoặc auto-detected)
      4. base config

    Parameters
    ----------
    cfg : dict
        Base config từ load_config()
    era : str, optional
        "co_trang", "hien_dai", "xua"
    genre : str, optional
        "hanh_dong", "tam_ly", "kinh_di", ...
    context_tags : List[str], optional
        ["dong_duc", "toi", "ngoai_troi", ...]
    profile : str, optional
        Manual override profile key (vd: "co_trang_dong_duc")
    custom_knobs : dict, optional
        Per-movie overrides từ metadata.json

    Returns
    -------
    (profile_key, merged_cfg)
        - profile_key: tên profile đang áp dụng
        - merged_cfg: config đã merge
    """

    # 1) Xác định profile key
    if profile:
        profile_key = profile
    else:
        profile_key = _choose_profile_key(era, genre, context_tags)

    # 2) Lấy profile config
    profiles = cfg.get("profiles", {})
    profile_cfg = profiles.get(profile_key, {})

    if not profile_cfg:
        # Fallback balanced nếu profile không tồn tại
        profile_cfg = profiles.get("balanced", {})
        if not profile_cfg:
            profile_key = "default"
            profile_cfg = {}

    # 3) Deep merge: base → profile → custom_knobs
    merged = deep_merge(cfg, profile_cfg)

    if custom_knobs:
        merged = deep_merge(merged, custom_knobs)

    # 4) Apply ENV overrides (highest priority)
    env_overrides = {}

    # merge.within_movie_threshold
    within = _env_float("FS_WITHIN", None)
    if within is not None:
        env_overrides.setdefault("merge", {})["within_movie_threshold"] = within

    # filter_clusters.min_size
    min_size = _env_int("FS_MIN_SIZE", None)
    if min_size is not None:
        env_overrides.setdefault("filter_clusters", {})["min_size"] = min_size

    # cluster.distance_threshold_pca
    dist_pca = _env_float("FS_DIST_PCA", None)
    if dist_pca is not None:
        env_overrides.setdefault("cluster", {})["distance_threshold_pca"] = dist_pca

    # tracklet.max_age
    max_age = _env_int("FS_MAX_AGE", None)
    if max_age is not None:
        env_overrides.setdefault("tracklet", {})["max_age"] = max_age

    # tracklet.iou_threshold
    iou = _env_float("FS_IOU", None)
    if iou is not None:
        env_overrides.setdefault("tracklet", {})["iou_threshold"] = iou

    if env_overrides:
        merged = deep_merge(merged, env_overrides)

    return profile_key, merged


# =============================================================================
# Backward compatibility: extract knobs
# =============================================================================

def extract_knobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common tuning knobs từ config (for logging/display).

    Returns dict với keys:
        - within, min_size, dist_pca, max_age, iou
    """
    return {
        "within": cfg.get("merge", {}).get("within_movie_threshold"),
        "min_size": cfg.get("filter_clusters", {}).get("min_size"),
        "dist_pca": cfg.get("cluster", {}).get("distance_threshold_pca"),
        "max_age": cfg.get("tracklet", {}).get("max_age"),
        "iou": cfg.get("tracklet", {}).get("iou_threshold"),
    }

# ---------------------------------------------------------------------
# Backward-compat helper for services.recognition
# ---------------------------------------------------------------------
def get_recognition_settings(cfg: dict) -> dict:
    """
    Trả về dict cấu hình cho bước nhận diện, đảm bảo có khóa 'SIM_THRESHOLD'.
    Ưu tiên lấy từ cfg['recognition']['SIM_THRESHOLD'], nếu thiếu thì
    rớt xuống cfg['search']['near_match_threshold'], cuối cùng default = 0.3.
    """
    if not isinstance(cfg, dict):
        return {"SIM_THRESHOLD": 0.3}

    recog = (cfg.get("recognition") or {}).copy()
    sim = recog.get("SIM_THRESHOLD")

    if sim is None:
        # backward-compat: đọc từ 'search.near_match_threshold' nếu có
        sim = (cfg.get("search") or {}).get("near_match_threshold")

    try:
        sim = float(sim)
    except Exception:
        sim = 0.3

    recog["SIM_THRESHOLD"] = sim
    return recog
