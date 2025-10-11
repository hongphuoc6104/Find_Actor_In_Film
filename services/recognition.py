# services/recognition.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.config_loader import load_config, get_recognition_settings
from utils.search_actor import search_actor

# --- optional (nếu có) để “đá” rebuild index & in stats ---
try:
    from utils.indexer import build_character_index  # type: ignore
except Exception:
    build_character_index = None  # type: ignore

DEBUG = os.getenv("FS_DEBUG", "1") != "0"  # bật debug mặc định


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _load_json_file(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@dataclass
class MovieMeta:
    title: str
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    total_frames: Optional[int] = None
    video_path: Optional[str] = None


def _read_metadata(cfg: dict) -> Dict[str, Any]:
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    meta_path = storage.get("metadata_json") or "Data/metadata.json"
    return _load_json_file(meta_path) or {}


def _reverse_movie_id_map(meta: Dict[str, Any]) -> Dict[str, str]:
    rev: Dict[str, str] = {}
    gen = meta.get("_generated") or {}
    id_map = gen.get("movie_id_map") or {}
    if isinstance(id_map, dict):
        for title, mid in id_map.items():
            try:
                rev[str(int(mid))] = str(title)
            except Exception:
                pass
    return rev


def _title_from_any(meta: Dict[str, Any], movie_key: Any) -> Optional[str]:
    s = str(movie_key).strip()
    if not s:
        return None
    if s in meta and s != "_generated":
        return s
    return _reverse_movie_id_map(meta).get(s)


def _movie_meta(meta: Dict[str, Any], movie_key: Any) -> Optional[MovieMeta]:
    title = _title_from_any(meta, movie_key)
    if not title:
        return None
    info = meta.get(title) or {}
    return MovieMeta(
        title=title,
        fps=info.get("fps"),
        duration_seconds=info.get("duration_seconds"),
        total_frames=info.get("total_frames"),
        video_path=info.get("video_path"),
    )


def _ensure_scenes(char_entry: Dict[str, Any], max_scenes: int = 8) -> None:
    sc = char_entry.get("scenes")
    if isinstance(sc, list) and sc:
        char_entry["scenes"] = sc[:max_scenes]


def recognize(image_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    cfg = load_config()
    meta_all = _read_metadata(cfg)

    search_cfg = cfg.get("search", {}) if isinstance(cfg, dict) else {}
    recognition_cfg = get_recognition_settings(cfg)

    present_threshold = _as_float(search_cfg.get("present_threshold", 0.5), 0.5)
    near_match_threshold = _as_float(recognition_cfg.get("SIM_THRESHOLD", 0.3), 0.3)
    near_match_threshold = min(near_match_threshold, present_threshold)
    # để chắc chắn không “lọc chết”:
    min_score = _as_float(search_cfg.get("min_score", 0.0), 0.0)

    margin_threshold = _as_float(search_cfg.get("margin_threshold", 0.05), 0.05)
    ratio_threshold = _as_float(search_cfg.get("ratio_threshold", 1.1), 1.1)

    max_results_cfg = _as_int(search_cfg.get("max_results", search_cfg.get("top_k", 200)), 200)
    if top_k is not None:
        max_results_cfg = max(max_results_cfg, int(top_k))

    def _run_search() -> Dict[str, List[Dict[str, Any]]]:
        try:
            payload = search_actor(
                image_path,
                max_results=max_results_cfg,
                score_floor=min_score,  # 0.0
            )
        except TypeError:
            payload = search_actor(image_path, max_results_cfg)
        if isinstance(payload, dict) and "results" in payload:
            return payload.get("results") or {}
        return payload or {}

    # 1) chạy search lần 1 (không lọc)
    matches_by_movie = _run_search()

    if DEBUG:
        total = sum(len(v or []) for v in matches_by_movie.values())
        best = 0.0
        for lst in matches_by_movie.values():
            for it in (lst or []):
                best = max(best, _as_float(it.get("distance"), 0.0))
        print(f"[Recognize][RAW] movies={len(matches_by_movie)} total={total} best_score={best:.4f}")

    # 2) nếu rỗng → thử build index (1 lần) rồi chạy lại
    if not matches_by_movie and build_character_index:
        try:
            build_character_index()
            matches_by_movie = _run_search()
            if DEBUG:
                total = sum(len(v or []) for v in matches_by_movie.values())
                print(f"[Recognize][RETRY after build] total={total}")
        except Exception as e:
            if DEBUG:
                print(f"[Recognize][RETRY build failed] {e}")

    if not matches_by_movie:
        if DEBUG:
            print("[Recognize] No candidates returned from search_actor.")
        return {
            "is_unknown": True,
            "movies": [],
            "thresholds": {
                "present_threshold": present_threshold,
                "near_match_threshold": near_match_threshold,
                "margin_threshold": margin_threshold,
                "ratio_threshold": ratio_threshold,
                "min_score": min_score,
            },
        }

    # 3) gom & lọc theo ngưỡng trình bày
    output_movies: List[Dict[str, Any]] = []

    for movie_key, candidates in matches_by_movie.items():
        if not isinstance(candidates, list) or not candidates:
            continue
        cand_list = [c for c in candidates if isinstance(c, dict)]
        if not cand_list:
            continue

        cand_list.sort(key=lambda it: _as_float(it.get("distance"), 0.0), reverse=True)

        mmeta = _movie_meta(meta_all, movie_key)
        mtitle = mmeta.title if mmeta else str(movie_key)

        kept_chars: List[Dict[str, Any]] = []
        for c in cand_list:
            s = _as_float(c.get("distance"), 0.0)
            # chỉ áp ngưỡng “near” để hiển thị
            if s < near_match_threshold:
                continue
            ent = {
                "character_id": str(c.get("character_id")),
                "score": s,
                "rep_image": c.get("rep_image"),
                "preview_paths": c.get("preview_paths") or [],
                "scenes": c.get("scenes") or [],
                "match_status": "present" if s >= present_threshold else "near_match",
                "match_label": "Xuất hiện" if s >= present_threshold else "Gần giống",
            }
            _ensure_scenes(ent, max_scenes=_as_int(search_cfg.get("max_scenes", 8), 8))
            kept_chars.append(ent)

        if not kept_chars:
            continue

        output_movies.append({
            "movie": mtitle,
            "movie_id": str(movie_key),
            "characters": kept_chars,
        })

    # sort phim theo best score
    def best_score(m: Dict[str, Any]) -> float:
        return max([_as_float(c.get("score"), 0.0) for c in m.get("characters", [])] or [0.0])

    output_movies.sort(key=best_score, reverse=True)

    if DEBUG:
        if output_movies:
            mx = best_score(output_movies[0])
            print(f"[Recognize] Movies kept={len(output_movies)} top_best={mx:.4f}")
        else:
            print("[Recognize] After thresholding: no movies")

    return {
        "is_unknown": len(output_movies) == 0,
        "bucket": "present",
        "movies": output_movies,
        "thresholds": {
            "present_threshold": present_threshold,
            "near_match_threshold": near_match_threshold,
            "margin_threshold": margin_threshold,
            "ratio_threshold": ratio_threshold,
            "min_score": min_score,
        },
    }
