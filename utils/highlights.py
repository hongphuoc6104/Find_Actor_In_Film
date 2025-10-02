"""Shared utilities for working with highlight scene metadata."""
from __future__ import annotations
import logging

import math
from typing import Any, Dict, Iterable, List, Sequence


LOGGER = logging.getLogger(__name__)


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


def _collect_numbers(entries: Iterable[Any]) -> List[float]:
    values: List[float] = []
    for entry in entries:
        parsed = _parse_float(entry)
        if parsed is not None:
            values.append(parsed)
    return values


def _max_or_none(values: Sequence[float]) -> float | None:
    return max(values) if values else None


def _min_or_none(values: Sequence[float]) -> float | None:
    return min(values) if values else None


def normalise_highlights(
    highlights: Any,
    *,
    highlight_limit: int | None = None,
    merge_gap: float | None = None,
    scene_start: float | None = None,
    scene_end: float | None = None,
    logger: logging.Logger | None = None,
    scene_identifier: Any | None = None,
) -> List[Dict[str, Any]]:
    """Coalesce highlight entries and annotate similarity metadata.


    Parameters
    ----------
    highlights:
        Raw highlight payload from metadata. Non-dict entries are ignored.
    highlight_limit:
        Maximum number of highlights to keep before merging. ``None`` keeps all
        entries.
    merge_gap:
        Maximum gap (seconds) between segments for them to be merged.
    scene_start, scene_end:
        Bounds used when padding isolated highlights.
    logger:
        Optional logger used for ``DEBUG_HL`` telemetry. Defaults to the module
        logger.
    scene_identifier:
        Identifier included in debug logs for easier traceability.
    """

    if not isinstance(highlights, list) or not highlights:
        return []

    effective_logger = logger or LOGGER
    selected = _select_top_highlights(highlights, highlight_limit)
    prepared_segments: List[Dict[str, Any]] = []

    merge_gap_value = 0.0
    if merge_gap is not None:
        merge_gap_value = max(float(merge_gap), 0.0)

    start_bound = _parse_float(scene_start)
    end_bound = _parse_float(scene_end)

    for index, raw_highlight in enumerate(selected):
        start = _parse_float(raw_highlight.get("start"))
        end = _parse_float(raw_highlight.get("end"))
        if start is None or end is None:
            continue

        if end < start:
            start, end = end, start

        highlight_copy = dict(raw_highlight)
        match_count = _parse_int(highlight_copy.get("match_count"))
        if match_count is None or match_count <= 0:
            match_count = 1

        duration = _parse_float(highlight_copy.get("duration"))
        if duration is None:
            duration = max(end - start, 0.0)

        # Pad isolated hits shorter than 1s by +/-2s within scene bounds.
        if match_count <= 1 and duration < 1.0:
            pad_seconds = 2.0
            if start_bound is not None:
                start = max(start - pad_seconds, start_bound)
            else:
                start -= pad_seconds
            if end_bound is not None:
                end = min(end + pad_seconds, end_bound)
            else:
                end += pad_seconds
            if end < start:
                end = start
            duration = max(end - start, 0.0)

        highlight_copy["start"] = round(start, 3)
        highlight_copy["end"] = round(end, 3)
        highlight_copy["duration"] = round(duration, 3)
        highlight_copy["match_count"] = int(match_count)

        det_scores = []
        for key in ("max_det_score", "min_det_score", "det_score"):
            det_scores.extend(_collect_numbers([highlight_copy.get(key)]))

        similarity_candidates: List[float] = []
        for key in (
                "score",
                "max_score",
                "avg_similarity",
                "max_similarity",
                "actor_similarity",
                "similarity",
        ):
            similarity_candidates.extend(_collect_numbers([highlight_copy.get(key)]))

        prepared_segments.append(
            {
                "start": float(highlight_copy["start"]),
                "end": float(highlight_copy["end"]),
                "match_count": int(match_count),
                "det_scores": det_scores,
                "similarities": similarity_candidates,
                "payload": highlight_copy,
                "index": index,
            }
        )

    if not prepared_segments:
        return []

    prepared_segments.sort(key=lambda item: (item["start"], item["end"], item["index"]))

    merged_segments: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    def _finalise(segment: Dict[str, Any]) -> Dict[str, Any]:
        start_value = segment["start"]
        end_value = segment["end"]
        duration_value = max(end_value - start_value, 0.0)
        det_scores = segment.get("det_scores", [])
        similarities = segment.get("similarities", [])
        sources = segment.get("sources", [])
        total_match_count = int(segment.get("match_count", 0))

        best_similarity = _max_or_none(similarities)
        if best_similarity is None:
            best_similarity = 0.0
        similarity_percent = round(best_similarity * 100.0, 2)

        merged_payload: Dict[str, Any] = {
            "start": round(start_value, 3),
            "end": round(end_value, 3),
            "duration": round(duration_value, 3),
            "match_count": total_match_count,
            "sources": [
                {"raw": dict(source)}
                for source in sources
                if isinstance(source, dict)
            ],
        }

        if det_scores:
            merged_payload["max_det_score"] = max(det_scores)
            merged_payload["min_det_score"] = min(det_scores)

        merged_payload["score"] = float(best_similarity)
        merged_payload["max_score"] = float(best_similarity)
        if similarities:
            merged_payload["max_similarity"] = max(similarities)
            merged_payload["min_similarity"] = min(similarities)
            merged_payload["avg_similarity"] = sum(similarities) / len(similarities)

        merged_payload["similarity_percent"] = similarity_percent

        # Merge union-able fields from sources
        matched_clusters: set[str] = set()
        matched_final_ids: set[str] = set()
        supporting_detections: List[Any] = []
        has_target = False

        for source in sources:
            clusters = source.get("matched_cluster_ids")
            if isinstance(clusters, (list, set, tuple)):
                matched_clusters.update(str(c) for c in clusters if c is not None)
            final_ids = source.get("matched_final_character_ids")
            if isinstance(final_ids, (list, set, tuple)):
                matched_final_ids.update(str(c) for c in final_ids if c is not None)
            detections = source.get("supporting_detections")
            if isinstance(detections, list):
                supporting_detections.extend(
                    det for det in detections if isinstance(det, dict)
                )
            if source.get("has_target"):
                has_target = True

        if matched_clusters:
            merged_payload["matched_cluster_ids"] = sorted(matched_clusters)
        if matched_final_ids:
            merged_payload["matched_final_character_ids"] = sorted(matched_final_ids)
        if supporting_detections:
            merged_payload["supporting_detections"] = supporting_detections
        if has_target:
            merged_payload["has_target"] = True

        return merged_payload

    for segment in prepared_segments:
        if current is None:
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "match_count": segment["match_count"],
                "det_scores": list(segment["det_scores"]),
                "similarities": list(segment["similarities"]),
                "sources": [segment["payload"]],
            }
            continue

        gap = segment["start"] - current["end"]
        if gap <= merge_gap_value:
            current["end"] = max(current["end"], segment["end"])
            current["match_count"] = int(current.get("match_count", 0)) + int(
                segment["match_count"]
            )
            current.setdefault("det_scores", []).extend(segment["det_scores"])
            current.setdefault("similarities", []).extend(segment["similarities"])
            current.setdefault("sources", []).append(segment["payload"])
        else:
            merged_segments.append(_finalise(current))
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "match_count": segment["match_count"],
                "det_scores": list(segment["det_scores"]),
                "similarities": list(segment["similarities"]),
                "sources": [segment["payload"]],
            }

    if current is not None:
        merged_segments.append(_finalise(current))

    if merged_segments:
        try:
            effective_logger.debug(
                "DEBUG_HL normalise highlights",
                extra={
                    "scene": scene_identifier,
                    "merge_gap": merge_gap_value,
                    "raw_count": len(prepared_segments),
                    "merged_count": len(merged_segments),
                    "raw": [
                        (seg["start"], seg["end"])
                        for seg in prepared_segments
                    ],
                    "merged": [
                        (seg.get("start"), seg.get("end")) for seg in merged_segments
                    ],
                },
            )
        except Exception:  # pragma: no cover - logging safety
            pass

    return merged_segments
