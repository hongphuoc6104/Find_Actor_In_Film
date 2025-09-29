"""Shared utilities for working with highlight scene metadata."""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _parse_float(value: Any) -> float | None:
    """Best-effort conversion of ``value`` to a finite ``float``."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return float(result)


def _parse_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to ``int``."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _select_top_highlights(
    highlights: List[Dict[str, Any]], limit: int | None
) -> List[Dict[str, Any]]:
    """Select the top highlights according to score, match count and start time."""

    if not isinstance(highlights, list):
        return []
    if limit is None or limit <= 0 or len(highlights) <= limit:
        return [dict(item) for item in highlights if isinstance(item, dict)]

    def _score(item: Dict[str, Any]) -> float:
        for key in ("max_score", "max_det_score", "avg_similarity"):
            value = _parse_float(item.get(key))
            if value is not None:
                return value
        return 0.0

    def _match_count(item: Dict[str, Any]) -> int:
        value = _parse_int(item.get("match_count"))
        return value if value is not None else 0

    ranked = sorted(
        (item for item in highlights if isinstance(item, dict)),
        key=lambda item: (
            -_score(item),
            -_match_count(item),
            _parse_float(item.get("start")) or float("inf"),
        ),
    )
    top = ranked[:limit]
    top.sort(key=lambda item: _parse_float(item.get("start")) or float("inf"))
    return [dict(item) for item in top]


def expand_highlight_scenes(
    raw_scenes: Any, *, highlight_limit: int | None
) -> List[Dict[str, Any]]:
    """Flatten highlight intervals into cursorable scene entries.

    Parameters
    ----------
    raw_scenes:
        The raw ``scenes`` collection attached to a character payload. Each
        scene may contain a ``highlights`` list describing sub-intervals within
        the scene.
    highlight_limit:
        Maximum number of highlight intervals to retain per scene. ``None``
        keeps all highlight entries.

    Returns
    -------
    list of dict
        A list where each entry corresponds to a highlight interval, augmented
        with cursor metadata so the API layer can page through highlights as if
        they were standalone scenes.
    """

    if not isinstance(raw_scenes, list):
        return []

    expanded: List[Dict[str, Any]] = []

    for source_index, scene in enumerate(raw_scenes):
        if not isinstance(scene, dict):
            continue

        highlights = scene.get("highlights")
        if not isinstance(highlights, list) or not highlights:
            continue

        raw_scene_index = scene.get("scene_index")
        source_scene_index: int | None = None
        if isinstance(raw_scene_index, int):
            source_scene_index = raw_scene_index
        else:
            try:
                source_scene_index = int(raw_scene_index)
            except (TypeError, ValueError):
                source_scene_index = None

        if source_scene_index is None:
            source_scene_index = source_index

        ranked_highlights = _select_top_highlights(highlights, highlight_limit)

        valid_highlights: List[Dict[str, Any]] = []
        for highlight in ranked_highlights:
            start = _parse_float(highlight.get("start"))
            end = _parse_float(highlight.get("end"))
            if start is None or end is None or end < start:
                continue
            highlight_copy = dict(highlight)
            highlight_copy["start"] = start
            highlight_copy["end"] = end
            valid_highlights.append(highlight_copy)

        if not valid_highlights:
            continue

        highlight_total = len(valid_highlights)
        for highlight_index, highlight in enumerate(valid_highlights):
            cursor_index = len(expanded)
            scene_copy = dict(scene)
            scene_copy["highlights"] = [highlight]
            scene_copy["highlight_index"] = highlight_index
            scene_copy["highlight_total"] = highlight_total
            scene_copy["source_scene_index"] = source_scene_index
            scene_copy["scene_index"] = cursor_index

            highlight_start = highlight.get("start")
            highlight_end = highlight.get("end")
            highlight_duration = _parse_float(highlight.get("duration"))
            if (
                highlight_duration is None
                and isinstance(highlight_start, (int, float))
                and isinstance(highlight_end, (int, float))
            ):
                highlight_duration = max(highlight_end - highlight_start, 0.0)
                highlight["duration"] = highlight_duration
            if isinstance(highlight_start, (int, float)):
                scene_copy["start_time"] = float(highlight_start)
                scene_copy["video_start_timestamp"] = float(highlight_start)
                scene_copy["clip_start_timestamp"] = float(highlight_start)
            if isinstance(highlight_end, (int, float)):
                scene_copy["end_time"] = float(highlight_end)
                scene_copy["video_end_timestamp"] = float(highlight_end)
                scene_copy["clip_end_timestamp"] = float(highlight_end)
            if isinstance(highlight_duration, (int, float)):
                scene_copy["duration"] = float(highlight_duration)

            expanded.append(scene_copy)

    return expanded
