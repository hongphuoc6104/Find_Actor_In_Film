"""Utilities for recognizing faces using a configured face index."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.config_loader import load_config
from utils.search_actor import search_actor


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
        timeline = scene_copy.get("timeline")
        if isinstance(timeline, list):
            scene_copy["timeline"] = [
                dict(entry) if isinstance(entry, dict) else entry
                for entry in timeline
            ]
        return scene_copy
    if isinstance(scene, str):
        return {"frame": scene}
    return None


def recognize(image_path: str, top_k: int | None = None) -> Dict[str, Any]:
    """Recognize a face image using the configured index.

    The return payload is organised per movie and contains metadata required by
    the API layer to expose preview images and scene navigation.
    """

    cfg = load_config()
    search_cfg = cfg.get("search", {})
    threshold = _as_float(search_cfg.get("threshold", 0.5), 0.5)
    if top_k is None:
        top_k = _as_int(search_cfg.get("top_k", 5), 5)

    matches_by_movie = search_actor(image_path, k=top_k)
    if not matches_by_movie:
        return {"is_unknown": True, "movies": []}

    all_candidates: List[Dict[str, Any]] = []
    for movie_id, candidates in matches_by_movie.items():
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate_copy = dict(candidate)
                candidate_copy["movie_id"] = str(movie_id)
                all_candidates.append(candidate_copy)

    if not all_candidates:
        return {"is_unknown": True, "movies": []}

    all_candidates.sort(key=lambda item: _as_float(item.get("distance"), 0.0), reverse=True)
    best_score = _as_float(all_candidates[0].get("distance"), 0.0)
    is_unknown = best_score < threshold

    movies: List[Dict[str, Any]] = []
    for movie_id, candidates in matches_by_movie.items():
        if not isinstance(candidates, list) or not candidates:
            continue

        movie_name: str | None = None
        formatted_characters: List[Dict[str, Any]] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            movie_name = movie_name or candidate.get("movie")
            scenes = candidate.get("scenes")
            total_scenes = len(scenes) if isinstance(scenes, list) else 0
            first_scene = None
            if isinstance(scenes, list) and scenes:
                first_scene = _normalize_scene(scenes[0])
            next_cursor = 1 if isinstance(scenes, list) and len(scenes) > 1 else None

            formatted_characters.append(
                {
                    "movie_id": str(movie_id),
                    "movie": candidate.get("movie"),
                    "character_id": str(candidate.get("character_id", "")),
                    "score": _as_float(candidate.get("distance"), 0.0),
                    "distance": _as_float(candidate.get("distance"), 0.0),
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
                }
            )

        if not formatted_characters:
            continue

        formatted_characters.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        movies.append(
            {
                "movie_id": str(movie_id),
                "movie": movie_name,
                "score": formatted_characters[0].get("score", 0.0),
                "characters": formatted_characters,
            }
        )

    movies.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    return {
        "is_unknown": is_unknown,
        "best_score": best_score,
        "movies": movies,
    }
