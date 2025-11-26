# utils/config_loader.py
"""
Bộ nạp cấu hình trung tâm + preset cho pipeline.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# =============================================================================
# Constants & YAML Loader
# =============================================================================
DEFAULT_CFG_PATH = Path("configs/config.yaml")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CFG_PATH
    cfg = _read_yaml(cfg_path)
    # Thêm auto_tuning vào defaults để đảm bảo luôn tồn tại
    defaults = {
        "storage": {}, "embedding": {}, "pca": {}, "clustering": {}, "cluster": {},
        "merge": {}, "post_merge": {}, "filter": {}, "filter_clusters": {},
        "quality_filters": {}, "centroid": {}, "index": {}, "search": {},
        "highlight": {}, "recognition": {}, "frontend": {},
        "preview": {"source": "frames", "max_images_per_cluster": 24},
        "tracklet": {"max_age": 3, "iou_threshold": 0.28},
        "auto_tuning": {"rules": []}  # Đảm bảo nhánh này luôn tồn tại
    }
    for key, default_val in defaults.items():
        cfg.setdefault(key, default_val)
    return cfg


# =============================================================================
# Deep merge & Metadata Loader
# =============================================================================
def deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for d in dicts:
        if not isinstance(d, dict): continue
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
    return result


def load_movie_metadata(movie_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
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
# --- BỘ NÃO TỰ ĐỘNG TINH CHỈNH PHIÊN BẢN HOÀN CHỈNH ---
# =============================================================================

def generate_auto_profile(
        cfg: Dict[str, Any],
        video_profile: Dict[str, str]
) -> Dict[str, Any]:
    """
    Dựa trên Video Profile và các quy tắc trong config.yaml,
    tạo ra một bộ tham số ghi đè.
    """
    overrides = {}
    auto_tuning_cfg = cfg.get("auto_tuning", {})
    rules = auto_tuning_cfg.get("rules", [])

    print("\n--- Applying Auto-Tuning Rules ---")
    # 1. Áp dụng các quy tắc từ config.yaml (giữ nguyên logic cũ)
    for i, rule in enumerate(rules):
        conditions = rule.get("conditions", {})
        if not conditions: continue

        # Logic match: tất cả conditions phải khớp với video_profile
        is_match = all(video_profile.get(key) == value for key, value in conditions.items())

        if is_match:
            rule_overrides = rule.get("overrides", {})
            print(f"[AutoTuning] Rule matched: {conditions} -> Applying overrides: {rule_overrides}")
            overrides = deep_merge(overrides, rule_overrides)

    # 2. LOGIC MỚI: Đọc quy tắc min_size từ config
    min_size_rules = auto_tuning_cfg.get("min_size_rules", {})
    duration_base_map = min_size_rules.get("duration_base", {"Short": 3, "Medium": 5, "Long": 7})
    complexity_adj_map = min_size_rules.get("complexity_adjustment", {"Crowded": 1.5})

    duration_cat = video_profile.get("duration", "Medium")
    complexity_cat = video_profile.get("complexity")

    # Lấy ngưỡng cơ bản từ map
    final_min_size = duration_base_map.get(duration_cat, 5)

    # Điều chỉnh nếu phim đông đúc
    if complexity_cat and complexity_cat in complexity_adj_map:
        adjustment_factor = complexity_adj_map[complexity_cat]
        final_min_size = int(final_min_size * adjustment_factor)
        print(
            f"[AutoTuning] Video is {complexity_cat} -> Adjusting `min_size` up by {adjustment_factor}x to {final_min_size}.")

    print(
        f"[AutoTuning] Final `min_size` decided: {final_min_size} (based on duration='{duration_cat}', complexity='{complexity_cat}')")
    overrides = deep_merge(overrides, {"filter_clusters": {"min_size": final_min_size}})

    return overrides


def apply_preset(
        cfg: Dict[str, Any],
        video_profile: Optional[Dict[str, str]] = None,
        custom_knobs: Optional[Dict[str, Any]] = None,
        **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Áp dụng các lớp cấu hình theo thứ tự ưu tiên.
    Priority (cao → thấp): ENV vars -> custom_knobs -> auto_profile -> base config
    """
    merged = deepcopy(cfg)
    profile_key = "auto_tuned"

    # Lớp 1 (thấp nhất): Cấu hình tự động
    if video_profile:
        auto_profile_overrides = generate_auto_profile(merged, video_profile)
        merged = deep_merge(merged, auto_profile_overrides)

    # Lớp 2: Knobs tùy chỉnh cho từng phim
    if custom_knobs:
        merged = deep_merge(merged, custom_knobs)

    # Lớp 3 (cao nhất): Biến môi trường
    env_overrides = {}
    min_size_env = _env_int("FS_MIN_SIZE", None)
    if min_size_env is not None:
        env_overrides.setdefault("filter_clusters", {})["min_size"] = min_size_env

    if env_overrides:
        merged = deep_merge(merged, env_overrides)

    return profile_key, merged


# =============================================================================
# Các hàm tiện ích khác
# =============================================================================

def extract_knobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "within": cfg.get("merge", {}).get("within_movie_threshold"),
        "min_size": cfg.get("filter_clusters", {}).get("min_size"),
        "dist_pca": cfg.get("cluster", {}).get("distance_threshold_pca"),
        "max_age": cfg.get("tracklet", {}).get("max_age"),
        "iou": cfg.get("tracklet", {}).get("iou_threshold"),
        # [Cập nhật] Thêm min_face_size để debug dễ hơn trong log
        "min_face_size": cfg.get("quality_filters", {}).get("min_face_size")
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


def get_video_duration(movie_title: str, cfg: Dict[str, Any]) -> Optional[float]:
    meta_path = Path(cfg["storage"]["metadata_json"])
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            all_meta = json.load(f)
        duration = all_meta.get(movie_title, {}).get("duration_seconds")
        return float(duration) if duration is not None else None
    except Exception:
        return None