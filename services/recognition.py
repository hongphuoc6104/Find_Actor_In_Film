"""Utilities for recognizing faces using a configured face index."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from utils.config_loader import (
    get_highlight_settings,
    get_recognition_settings,
    load_config,
)
from utils.search_actor import search_actor
from utils.highlights import normalise_highlights



LOGGER = logging.getLogger(__name__)



_HIGHLIGHT_SETTINGS = get_highlight_settings()
_HIGHLIGHT_LIMIT = _HIGHLIGHT_SETTINGS["TOP_K_HL_PER_SCENE"]
_MERGE_GAP = float(_HIGHLIGHT_SETTINGS["MERGE_GAP_SEC"])


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


def _maybe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return float(result)


def _normalize_scene(scene: Any, *, highlight_limit: int | None = None) -> Dict[str, Any] | None:
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

        scene_start = (
            _maybe_float(scene_copy.get("video_start_timestamp"))
            or _maybe_float(scene_copy.get("start_time"))
            or _maybe_float(scene_copy.get("clip_start_timestamp"))
        )
        scene_end = (
            _maybe_float(scene_copy.get("video_end_timestamp"))
            or _maybe_float(scene_copy.get("end_time"))
            or _maybe_float(scene_copy.get("clip_end_timestamp"))
        )

        highlights = normalise_highlights(
            scene_copy.get("highlights"),
            highlight_limit=highlight_limit,
            merge_gap=_MERGE_GAP,
            scene_start=scene_start,
            scene_end=scene_end,
            logger=LOGGER,
            scene_identifier={
                "scene_index": scene_copy.get("scene_index"),
                "movie": scene_copy.get("movie"),
            },
        )

        scene_copy["highlights"] = highlights
        scene_copy["highlight_total"] = len(highlights)
        scene_copy["highlight_merge_gap"] = _MERGE_GAP
        scene_copy.setdefault("highlight_index", 0 if highlights else None)


        return scene_copy

    if isinstance(scene, str):
        return {
            "frame": scene,
            "highlights": [],
            "highlight_total": 0,
            "highlight_index": None,
            "highlight_merge_gap": _MERGE_GAP,
        }
    return None

def _build_scene_variants(
    raw_scenes: Any, *, movie: Any | None = None
) -> List[Dict[str, Any]]:
    """Expand raw scenes into cursorable highlight variants."""

    variants: List[Dict[str, Any]] = []
    fallback_variants: List[Dict[str, Any]] = []

    if isinstance(raw_scenes, list):
        for idx, scene in enumerate(raw_scenes):
            normalized = _normalize_scene(scene, highlight_limit=_HIGHLIGHT_LIMIT)
            if not isinstance(normalized, dict):
                continue

            raw_scene_index = normalized.get("scene_index")
            try:
                source_scene_index = int(raw_scene_index)
            except (TypeError, ValueError):
                source_scene_index = idx

            normalized["source_scene_index"] = source_scene_index

            highlights = normalized.get("highlights")
            if not isinstance(highlights, list):
                highlights = []
                normalized["highlights"] = highlights

            highlight_total = len(highlights)

            try:
                LOGGER.debug(
                    "DEBUG_HL recognition scene prepared",
                    extra={
                        "scene": source_scene_index,
                        "highlight_total": highlight_total,
                        "movie": movie,
                    },
                )
            except Exception:  # pragma: no cover - defensive logging
                pass

            if highlight_total:
                for highlight_index in range(highlight_total):
                    scene_copy = dict(normalized)
                    scene_copy["highlight_index"] = highlight_index
                    scene_copy["highlight_total"] = highlight_total
                    scene_copy["scene_index"] = len(variants)
                    scene_copy["source_scene_index"] = source_scene_index
                    highlight = highlights[highlight_index]
                    if isinstance(highlight, dict):
                        start = _maybe_float(highlight.get("start"))
                        end = _maybe_float(highlight.get("end"))
                        duration = _maybe_float(highlight.get("duration"))
                        if start is not None:
                            scene_copy["start_time"] = start
                            scene_copy["video_start_timestamp"] = start
                            scene_copy["clip_start_timestamp"] = start
                        if end is not None:
                            scene_copy["end_time"] = end
                            scene_copy["video_end_timestamp"] = end
                            scene_copy["clip_end_timestamp"] = end
                        if duration is not None:
                            scene_copy["duration"] = duration
                    variants.append(scene_copy)
            else:
                fallback_scene = dict(normalized)
                fallback_scene["highlight_index"] = None
                fallback_scene["highlight_total"] = 0
                fallback_scene["scene_index"] = len(fallback_variants)
                fallback_scene["source_scene_index"] = source_scene_index
                fallback_variants.append(fallback_scene)

    return variants if variants else fallback_variants




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
            scene_variants = _build_scene_variants(
                scenes, movie=candidate.get("movie")
            )

            if scene_variants:
                total_scenes = len(scene_variants)
                first_scene = dict(scene_variants[0])
                next_cursor = 1 if total_scenes > 1 else None
            else:
                total_scenes = len(scenes) if isinstance(scenes, list) else 0
                first_scene = (
                    _normalize_scene(scenes[0], highlight_limit=_HIGHLIGHT_LIMIT)
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

            if scene_variants:
                formatted_character["scenes"] = [dict(scene) for scene in scene_variants]
            if isinstance(first_scene, dict):
                formatted_character["highlight_total"] = first_scene.get(
                    "highlight_total", len(scene_variants)
                )
            else:
                formatted_character["highlight_total"] = len(scene_variants)


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
