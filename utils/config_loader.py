
"""Helpers for loading and normalising project configuration."""

from __future__ import annotations

import math
import os
from typing import Any, Dict

import yaml


def load_config(cfg_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Tải file cấu hình YAML một cách an toàn từ gốc dự án."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_cfg_path = os.path.join(project_root, cfg_path)

    if not os.path.exists(absolute_cfg_path):
        raise FileNotFoundError(f"Không tìm thấy file config tại: {absolute_cfg_path}")


    with open(absolute_cfg_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise TypeError("Config file must define a mapping at the root level")
    return loaded


def _to_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(result):
        return fallback
    return float(result)


def _to_int(value: Any, fallback: int | None = None) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return fallback
    return result


def get_highlight_settings(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return highlight configuration merged with defaults."""

    if config is None:
        config = load_config()

    highlight_cfg: Dict[str, Any] = {}
    for key in ("highlight", "highlights"):
        candidate = config.get(key)
        if isinstance(candidate, dict):
            highlight_cfg = candidate
            break

    min_duration = _to_float(highlight_cfg.get("MIN_HL_DURATION_SEC"), 4.0)
    if "min_duration" in highlight_cfg:
        min_duration = _to_float(highlight_cfg.get("min_duration"), min_duration)

    merge_gap = _to_float(highlight_cfg.get("MERGE_GAP_SEC"), 6.0)
    if "max_gap_seconds" in highlight_cfg:
        merge_gap = _to_float(highlight_cfg.get("max_gap_seconds"), merge_gap)

    min_score = _to_float(highlight_cfg.get("MIN_SCORE"), 0.8)
    if "det_score_threshold" in highlight_cfg:
        min_score = _to_float(highlight_cfg.get("det_score_threshold"), min_score)

    sim_threshold_value = highlight_cfg.get("SIM_THRESHOLD")
    if sim_threshold_value is None:
        recognition_cfg = config.get("recognition") if isinstance(config, dict) else None
        if isinstance(recognition_cfg, dict):
            sim_threshold_value = recognition_cfg.get("SIM_THRESHOLD")
    sim_threshold = _to_float(sim_threshold_value, 0.3)

    top_k = _to_int(highlight_cfg.get("TOP_K_HL_PER_SCENE"), None)
    if "top_k_per_scene" in highlight_cfg and top_k is None:
        top_k = _to_int(highlight_cfg.get("top_k_per_scene"), None)
    if top_k is not None and top_k <= 0:
        top_k = None

    return {
        "MIN_HL_DURATION_SEC": min_duration,
        "MERGE_GAP_SEC": merge_gap,
        "MIN_SCORE": min_score,
        "TOP_K_HL_PER_SCENE": top_k,
        "SIM_THRESHOLD": sim_threshold,
    }


def get_recognition_settings(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return recognition thresholds with sensible fallbacks."""

    if config is None:
        config = load_config()

    recognition_cfg = config.get("recognition")
    search_cfg = config.get("search")

    search_threshold = 0.3
    if isinstance(search_cfg, dict):
        search_threshold = _to_float(search_cfg.get("near_match_threshold"), search_threshold)

    sim_threshold = search_threshold
    if isinstance(recognition_cfg, dict):
        sim_threshold = _to_float(recognition_cfg.get("SIM_THRESHOLD"), sim_threshold)

    return {"SIM_THRESHOLD": sim_threshold}


def get_frontend_settings(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return frontend timing configuration merged with defaults."""

    if config is None:
        config = load_config()

    frontend_cfg = config.get("frontend") if isinstance(config, dict) else None
    if not isinstance(frontend_cfg, dict):
        frontend_cfg = {}

    return {
        "SEEK_PAD_SEC": _to_float(frontend_cfg.get("SEEK_PAD_SEC"), 0.0),
        "PAUSE_TOLERANCE_SEC": _to_float(frontend_cfg.get("PAUSE_TOLERANCE_SEC"), 0.2),
        "MIN_VIEWABLE_SEC": _to_float(frontend_cfg.get("MIN_VIEWABLE_SEC"), 0.35),
    }