"""Utilities for recognizing faces using a configured face index."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.config_loader import (
    get_highlight_settings,
    get_recognition_settings,
    load_config,
)
from utils.search_actor import search_actor
from utils.highlights import expand_highlight_scenes


_HIGHLIGHT_SETTINGS = get_highlight_settings()
_HIGHLIGHT_LIMIT = _HIGHLIGHT_SETTINGS["TOP_K_HL_PER_SCENE"]

def _as_float(value: Any, default: float = 0.0) -> float:
    """Safely cast ``value`` to ``float`` while providing a fallback."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """Safely cast ``value`` to ``int`` while providing a fallback."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_scene(scene: Any) -> Dict[str, Any] | None:
    """Return a defensive copy of the provided scene metadata."""

    if scene is None:
        return None
    if isinstance(scene, dict):
        scene_copy = dict(scene)

        # --- Copy timeline ---
        timeline = scene_copy.get("timeline")
        if isinstance(timeline, list):
            scene_copy["timeline"] = [
                dict(entry) if isinstance(entry, dict) else entry
                for entry in timeline
            ]

        # --- Copy highlights ---
        highlights = scene_copy.get("highlights")
        if isinstance(highlights, list):
            scene_copy["highlights"] = [
                dict(entry) if isinstance(entry, dict) else entry
                for entry in highlights
            ]

        return scene_copy

    if isinstance(scene, str):
        return {"frame": scene}
    return None



PRESENT_LABEL_VI = "Xuất hiện trong phim"
NEAR_MATCH_LABEL_VI = "Có nhân vật gần giống"


def recognize(image_path: str, top_k: int | None = None) -> Dict[str, Any]:
    """Recognize a face image using the configured index.

    The return payload is organised per movie and contains metadata required by
    the API layer to expose preview images and scene navigation. Matches are
    separated into "present" (>= ``present_threshold``) and "near_match"
    (between ``near_match_threshold`` and ``present_threshold``) buckets; the
    API returns only the "present" bucket when available and otherwise falls
    back to the "near_match" bucket.
    """

    cfg = load_config()
    search_cfg = cfg.get("search", {})
    recognition_cfg = get_recognition_settings(cfg)
    present_threshold = _as_float(
        search_cfg.get("present_threshold", search_cfg.get("threshold", 0.5)), 0.5
    )
    near_match_threshold = _as_float(recognition_cfg.get("SIM_THRESHOLD"), 0.3)
    min_score = _as_float(
        search_cfg.get("min_score", near_match_threshold), near_match_threshold
    )

    max_results_cfg = _as_int(
        search_cfg.get("max_results", search_cfg.get("top_k", 50)), 50
    )
    if top_k is not None:
        max_results_cfg = max(max_results_cfg, _as_int(top_k, max_results_cfg))

    matches_by_movie = search_actor(
        image_path,
        k=max_results_cfg,
        score_floor=min_score,
        max_results=max_results_cfg,
    )
    if not matches_by_movie:
        return {"is_unknown": True, "movies": []}

    best_score = 0.0
    movies_by_status: Dict[str, Dict[str, Any]] = {"present": {}, "near_match": {}}

    for movie_id, candidates in matches_by_movie.items():
        if not isinstance(candidates, list) or not candidates:
            continue

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            score = _as_float(candidate.get("distance"), 0.0)
            if score < near_match_threshold:
                continue

            status = "present" if score >= present_threshold else "near_match"
            label = PRESENT_LABEL_VI if status == "present" else NEAR_MATCH_LABEL_VI

            movie_key = str(movie_id)
            movie_entry = movies_by_status[status].setdefault(
                movie_key,
                {
                    "movie_id": movie_key,
                    "movie": candidate.get("movie"),
                    "characters": [],
                    "match_status": status,
                    "match_label": label,
                },
            )
            if not movie_entry.get("movie") and candidate.get("movie"):
                movie_entry["movie"] = candidate.get("movie")

            scenes = candidate.get("scenes")
            flattened_scenes = expand_highlight_scenes(
                scenes, highlight_limit=_HIGHLIGHT_LIMIT
            )

            if flattened_scenes:
                total_scenes = len(flattened_scenes)
                first_scene = _normalize_scene(flattened_scenes[0])
                next_cursor = 1 if total_scenes > 1 else None
            else:
                total_scenes = len(scenes) if isinstance(scenes, list) else 0
                first_scene = (
                    _normalize_scene(scenes[0])
                    if isinstance(scenes, list) and scenes
                    else None
                )
                next_cursor = (
                    1 if isinstance(scenes, list) and len(scenes) > 1 else None
                )

            formatted_character = {
                "movie_id": movie_key,
                "movie": candidate.get("movie"),
                "character_id": str(candidate.get("character_id", "")),
                "score": score,
                "distance": score,
                "count": _as_int(candidate.get("count")),
                "track_count": _as_int(
                    candidate.get("track_count"),
                    _as_int(candidate.get("count")),
                ),
                "rep_image": candidate.get("rep_image"),
                "previews": candidate.get("previews") or [],
                "preview_paths": candidate.get("preview_paths") or [],
                "raw_cluster_ids": candidate.get("raw_cluster_ids") or [],
                "movies": candidate.get("movies") or [],
                "scene": first_scene,
                "scene_index": 0 if first_scene is not None else None,
                "scene_cursor": next_cursor,
                "next_scene_cursor": next_cursor,
                "total_scenes": total_scenes,
                "has_more_scenes": next_cursor is not None,
                "match_status": status,
                "match_label": label,
            }

            if formatted_character["character_id"]:
                movie_entry["characters"].append(formatted_character)
                best_score = max(best_score, score)

    selected_status = "present" if movies_by_status["present"] else "near_match"
    selected_movies_map = movies_by_status[selected_status]

    if not selected_movies_map:
        return {"is_unknown": True, "movies": []}

    movies: List[Dict[str, Any]] = []
    for movie_entry in selected_movies_map.values():
        characters = movie_entry.get("characters", [])
        if not characters:
            continue
        characters.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        movie_payload = {
            "movie_id": movie_entry.get("movie_id"),
            "movie": movie_entry.get("movie"),
            "score": characters[0].get("score", 0.0),
            "characters": characters,
            "match_status": movie_entry.get("match_status"),
            "match_label": movie_entry.get("match_label"),
        }
        movies.append(movie_payload)

    if not movies:
        return {"is_unknown": True, "movies": []}

    movies.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    is_unknown = best_score < present_threshold

    return {
        "is_unknown": is_unknown,
        "best_score": best_score,
        "movies": movies,
    }
