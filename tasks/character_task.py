from __future__ import annotations

import glob
import json
import logging

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from prefect import task
from sklearn.cluster import AgglomerativeClustering

from utils.config_loader import get_highlight_settings, load_config
from utils.indexer import build_index
from utils.vector_utils import _mean_vector, l2_normalize
from tasks.filter_clusters_task import filter_clusters_task
from utils.highlights import normalise_highlights

DEFAULT_CLIP_FPS = 8.0
MIN_CLIP_DURATION = 5.0
MAX_CLIP_DURATION = 10.0
PRE_CLIP_BUFFER_SECONDS = 1.0
POST_CLIP_BUFFER_SECONDS = 1.0
DEFAULT_FRAME_EXTENSIONS = [".jpg", ".jpeg", ".png"]
_HIGHLIGHT_SETTINGS = get_highlight_settings()
LOGGER = logging.getLogger(__name__)

DEFAULT_HIGHLIGHT_DET_SCORE = float(_HIGHLIGHT_SETTINGS["MIN_SCORE"])
DEFAULT_HIGHLIGHT_GAP_SECONDS = float(_HIGHLIGHT_SETTINGS["MERGE_GAP_SEC"])
DEFAULT_HIGHLIGHT_SIMILARITY = float(_HIGHLIGHT_SETTINGS["SIM_THRESHOLD"])
MAX_HIGHLIGHT_SAMPLES = 10

HIGHLIGHT_MIN_DURATION = float(_HIGHLIGHT_SETTINGS["MIN_HL_DURATION_SEC"])  # tối thiểu 4s
HIGHLIGHT_MAX_DURATION = 60.0  # tối đa 60s
HIGHLIGHT_EXTEND_SECONDS = 2.0  # mở rộng thêm 2s trước sau
# Giới hạn số highlight trên mỗi cảnh (None nghĩa là không giới hạn)
TOP_HIGHLIGHTS_PER_SCENE = _HIGHLIGHT_SETTINGS["TOP_K_HL_PER_SCENE"]
HIGHLIGHT_MIN_SCORE = float(_HIGHLIGHT_SETTINGS["MIN_SCORE"])
# HIGHLIGHT_MIN_CONFIDENCE = 0.85
# HIGHLIGHT_MAX_GAP_SECONDS = 2.0
# HIGHLIGHT_MAX_GAP_SECONDS = 2.0


def _frame_to_int(frame_name: Any) -> int:
    base = os.path.splitext(str(frame_name))[0]
    digits = "".join(ch for ch in base if ch.isdigit())
    try:
        return int(digits)
    except ValueError:
        return -1


def _timestamp_from_frame(frame_idx: int, fps: float | None) -> float | None:
    if fps is None or fps <= 0 or frame_idx < 0:
        return None
    return round(frame_idx / fps, 3)


def _parse_time(value: Any) -> float | None:
    """Convert ``value`` to a finite float timestamp if possible."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return float(result)


def _extract_similarity(entry: Dict[str, Any]) -> float | None:
    for key in ("actor_similarity", "similarity", "character_similarity"):
        similarity = _to_float(entry.get(key))
        if similarity is not None:
            return similarity
    return None



def _as_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype="float32")
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1)


def _normalize_bbox(bbox: Any) -> List[int]:
    if isinstance(bbox, (list, tuple)):
        return [int(float(x)) for x in bbox]
    arr = np.asarray(bbox).astype("float32").tolist()
    return [int(float(x)) for x in arr]


def _safe_slug(value: Any, fallback: str = "scene") -> str:
    """Return a filesystem-friendly slug for ``value``."""

    if value is None:
        text = fallback
    else:
        try:
            if isinstance(value, (int, np.integer)):
                text = str(int(value))
            else:
                text = str(value)
        except Exception:
            text = fallback

    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    sanitized = sanitized.strip("_") or fallback
    return sanitized[:80]

def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return float(result)


def _coerce_str_set(value: Any) -> set[str]:
    result: set[str] = set()
    if value is None:
        return result
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item is None or item != item:
                continue
            try:
                result.add(str(item))
            except Exception:
                continue
        return result
    if value != value:
        return result
    try:
        result.add(str(value))
    except Exception:
        return set()
    return result

def _load_cluster_metadata(
    meta_file: str, movie_id: Any, cluster_id: Any
) -> List[Dict[str, Any]]:
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "Failed to read cluster metadata for movie_id=%s cluster_id=%s from %s: %s",
            movie_id,
            cluster_id,
            meta_file,
            exc,
        )
    return []



def _summarise_detection(
    entry: Dict[str, Any],
    timestamp: float,
    det_score: float,
    similarity: float | None,
    clusters: set[str],
    final_ids: set[str],
    min_duration: float | None = HIGHLIGHT_MIN_DURATION,
) -> Dict[str, Any] | None:
    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "det_score": det_score,
    }
    for key in ("frame", "frame_index", "order", "track_id"):
        value = entry.get(key)
        if value is None or value != value:
            continue
        summary[key] = value
    if similarity is not None:
        summary["actor_similarity"] = similarity
    if clusters:
        summary["cluster_ids"] = sorted(clusters)
    if final_ids:
        summary["final_character_ids"] = sorted(final_ids)
    for identity_key in ("character_id", "final_character_id", "scene_final_character_id"):
        value = entry.get(identity_key)
        if value is None or value != value:
            continue
        summary[identity_key] = str(value)
    return summary


def _finalise_highlight(
    accumulator: Dict[str, Any],
    *,
    timeline_start: float | None = None,
    timeline_end: float | None = None,
    min_duration: float | None = None,
    highlight_support: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    start_val = _parse_time(accumulator.get("start"))
    end_val = _parse_time(accumulator.get("end"))

    if start_val is None or end_val is None:
        return None

    start = float(start_val)
    end = float(end_val)
    if end < start:
        start, end = end, start

    start_bound = _parse_time(timeline_start)
    if start_bound is None:
        start_bound = 0.0
    end_bound = _parse_time(timeline_end)

    extend_seconds = float(HIGHLIGHT_EXTEND_SECONDS) if HIGHLIGHT_EXTEND_SECONDS else 0.0
    if extend_seconds > 0:
        start -= extend_seconds
        end += extend_seconds

    start = max(start, start_bound)
    if end_bound is not None:
        end = min(end, end_bound)

    target_min_duration = _to_float(min_duration) or 0.0

    duration = end - start
    if target_min_duration > 0 and duration < target_min_duration:
        deficit = target_min_duration - duration
        available_before = max(start - start_bound, 0.0)
        available_after = float("inf")
        if end_bound is not None:
            available_after = max(end_bound - end, 0.0)

        extend_before = min(deficit / 2.0, available_before)
        if extend_before > 0:
            start -= extend_before
            deficit -= extend_before

        extend_after = min(deficit, available_after)
        if extend_after > 0:
            end += extend_after
            deficit -= extend_after

        if deficit > 0:
            remaining_before = max(available_before - extend_before, 0.0)
            extra_before = min(deficit, remaining_before)
            if extra_before > 0:
                start -= extra_before
                deficit -= extra_before

        if deficit > 0:
            if end_bound is None:
                end += deficit
                deficit = 0.0
            else:
                remaining_after = max(end_bound - end, 0.0)
                extra_after = min(deficit, remaining_after)
                if extra_after > 0:
                    end += extra_after
                    deficit -= extra_after

        start = max(start, start_bound)
        if end_bound is not None and end > end_bound:
            end = end_bound

        if target_min_duration > 0:
            current_duration = end - start
            if end_bound is None:
                if current_duration < target_min_duration:
                    end = start + target_min_duration
            else:
                if current_duration < target_min_duration:
                    desired_end = start + target_min_duration
                    if desired_end <= end_bound:
                        end = max(end, desired_end)
                    else:
                        end = end_bound
                        start = max(start_bound, end - target_min_duration)

                end = min(end, end_bound)
                min_start_allowed = end - target_min_duration
                if start > min_start_allowed:
                    start = max(start_bound, min_start_allowed)


        if end < start:
            end = start


        duration = end - start

        if duration <= 0:
            return None

        max_duration = _to_float(HIGHLIGHT_MAX_DURATION)
        if max_duration and duration > max_duration:
            end = start + max_duration
            if end_bound is not None and end > end_bound:
                end = end_bound
                start = max(start, end - max_duration)
        duration = end - start
        if duration <= 0:
            return None

    det_scores: List[float] = list(accumulator.get("det_scores", []))
    sim_values: List[float] = list(accumulator.get("similarities", []))
    weight_sum = _to_float(accumulator.get("weight_sum")) or 0.0
    weighted_similarity_sum = _to_float(accumulator.get("weighted_similarity_sum"))

    score: float | None = None
    score_source: str | None = None
    if weighted_similarity_sum is not None and weight_sum > 0:
        score = weighted_similarity_sum / weight_sum
        score_source = "similarity_weighted"
    elif sim_values:
        score = sum(sim_values) / len(sim_values)
        score_source = "similarity_average"

    match_count = int(accumulator.get("match_count", 0) or 0)
    matched_clusters = accumulator.get("matched_cluster_ids", set()) or set()
    matched_final_ids = accumulator.get("matched_final_character_ids", set()) or set()

    if score is None or not np.isfinite(score):
        has_matches = bool(match_count) or bool(matched_clusters) or bool(
            matched_final_ids
        )
        fallback_score: float | None = None
        fallback_source: str | None = None

        if has_matches:
            if det_scores:
                fallback_score = max(det_scores)
                fallback_source = "fallback_det_score"
            else:
                fallback_score = float(HIGHLIGHT_MIN_SCORE)
                fallback_source = "fallback_min_score"

        if fallback_score is None or not np.isfinite(fallback_score):
            return None

        score = fallback_score
        score_source = fallback_source

    highlight: Dict[str, Any] = {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "match_count": match_count,
        "supporting_detections": list(accumulator.get("supporting_detections", [])),
        "matched_cluster_ids": sorted(matched_clusters),
        "matched_final_character_ids": sorted(matched_final_ids),
        "has_target": True,
    }

    score = float(score)
    highlight["score"] = round(score, 6)
    highlight["max_score"] = round(score, 6)
    if score_source:
        highlight["score_source"] = score_source

    if det_scores:
        highlight["max_det_score"] = max(det_scores)
        highlight["min_det_score"] = min(det_scores)

    if sim_values:
        avg_similarity = sum(sim_values) / len(sim_values)
        highlight["avg_similarity"] = round(avg_similarity, 6)
        highlight["max_similarity"] = max(sim_values)
        highlight["min_similarity"] = min(sim_values)

    if duration < HIGHLIGHT_MIN_DURATION:
        end = start + HIGHLIGHT_MIN_DURATION
        duration = end - start
        # ghi lại vào highlight
        highlight["end"] = round(end, 3)
        highlight["duration"] = round(duration, 3)

    support_payload: Dict[str, Any] = {}
    if isinstance(highlight_support, dict):
        support_payload.update(highlight_support)

    target_support_min_duration = _to_float(min_duration)
    if target_support_min_duration is None:
        target_support_min_duration = _to_float(support_payload.get("min_duration"))
    if target_support_min_duration is not None:
        support_payload["min_duration"] = float(target_support_min_duration)

    if support_payload:
        highlight["highlight_support"] = support_payload

    return highlight


def _make_highlight_matcher(
    final_character_id: Any,
    allowed_clusters: set[str] | None,
    similarity_threshold: float | None,
) -> Any:
    allowed_final_ids: set[str] = set()
    if final_character_id is not None:
        try:
            allowed_final_ids.add(str(final_character_id))
        except Exception:
            pass
    allowed_clusters = {str(c) for c in (allowed_clusters or set()) if c is not None}

    def _matcher(entry: Dict[str, Any]) -> bool:
        clusters = _coerce_str_set(entry.get("cluster_id"))
        clusters.update(_coerce_str_set(entry.get("cluster_ids")))
        final_ids = _coerce_str_set(entry.get("final_character_id"))
        final_ids.update(_coerce_str_set(entry.get("final_character_ids")))

        has_final_match = bool(allowed_final_ids and final_ids & allowed_final_ids)
        has_cluster_match = bool(allowed_clusters and clusters & allowed_clusters)

        if allowed_final_ids or allowed_clusters:
            return has_final_match or has_cluster_match


        similarity: float | None = None
        for key in ("actor_similarity", "similarity", "character_similarity"):
            similarity = _to_float(entry.get(key))
            if similarity is not None:
                break

        return (
            similarity_threshold is not None
            and similarity is not None
            and similarity >= similarity_threshold
        )


    return _matcher

def _segment_target_hits(
    hits: List[Dict[str, Any]], merge_gap: float
) -> List[List[Dict[str, Any]]]:
    if not hits:
        return []

    gap_value = max(float(merge_gap), 0.0) if merge_gap is not None else 0.0
    sorted_hits = sorted(hits, key=lambda item: item["timestamp"])

    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_ts: float | None = None

    for hit in sorted_hits:
        timestamp = float(hit["timestamp"])
        if not current:
            current = [hit]
            last_ts = timestamp
            continue

        gap = timestamp - (last_ts if last_ts is not None else timestamp)
        if gap <= gap_value:
            current.append(hit)
        else:
            segments.append(current)
            current = [hit]
        last_ts = timestamp

    if current:
        segments.append(current)

    return segments


def _accumulate_segment_hits(segment_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not segment_hits:
        return {}

    matched_clusters: set[str] = set()
    matched_final_ids: set[str] = set()
    det_scores: List[float] = []
    similarities: List[float] = []
    weight_sum = 0.0
    weighted_similarity_sum = 0.0
    supporting: List[Dict[str, Any]] = []

    for hit in segment_hits:
        matched_clusters.update(hit.get("clusters", set()))
        matched_final_ids.update(hit.get("final_ids", set()))

        det_score = hit.get("det_score")
        if det_score is not None:
            det_scores.append(float(det_score))

        similarity = hit.get("similarity")
        if similarity is not None:
            similarities.append(float(similarity))
            weight = float(det_score) if det_score is not None else 1.0
            weight_sum += weight
            weighted_similarity_sum += float(similarity) * weight

        detection_summary = hit.get("summary")
        if (
            isinstance(detection_summary, dict)
            and len(supporting) < MAX_HIGHLIGHT_SAMPLES
        ):
            supporting.append(detection_summary)

    return {
        "start": float(segment_hits[0]["timestamp"]),
        "end": float(segment_hits[-1]["timestamp"]),
        "det_scores": det_scores,
        "similarities": similarities,
        "weight_sum": weight_sum,
        "weighted_similarity_sum": weighted_similarity_sum,
        "matched_cluster_ids": matched_clusters,
        "matched_final_character_ids": matched_final_ids,
        "supporting_detections": supporting,
        "match_count": len(segment_hits),
    }



def _build_highlights(
    entries: List[Dict[str, Any]],
    *,
    det_th: float = DEFAULT_HIGHLIGHT_DET_SCORE,
    max_gap: float = DEFAULT_HIGHLIGHT_GAP_SECONDS,
    match_fn: Any | None = None,
    sim_threshold: float | None = DEFAULT_HIGHLIGHT_SIMILARITY,
    min_duration: float = HIGHLIGHT_MIN_DURATION,
    min_score: float | None = HIGHLIGHT_MIN_SCORE,
) -> List[Dict[str, Any]]:
    timeline_start: float | None = None
    timeline_end: float | None = None
    hits: List[Dict[str, Any]] = []

    # duyệt từng entry để chọn "hit"
    for entry in entries:
        timestamp = _parse_time(entry.get("timestamp"))
        if timestamp is None:
            continue
        if timeline_start is None or timestamp < timeline_start:
            timeline_start = timestamp
        if timeline_end is None or timestamp > timeline_end:
            timeline_end = timestamp

        # if match_fn is not None and not match_fn(entry):
        #     continue

        # det_score = _to_float(entry.get("det_score"))
        # if det_th is not None and (det_score is None or det_score < det_th):
        #     # chỉ bỏ nếu KHÔNG có match_fn positive
        #     if not (match_fn and match_fn(entry)):
        #         continue
        #
        # similarity = _extract_similarity(entry)
        # # Nếu không có match_fn thì mới lọc theo similarity threshold
        # if match_fn is None:
        #     if sim_threshold is not None:
        #         if similarity is None or similarity < sim_threshold:
        #             continue
        #     elif similarity is None:
        #         continue

        det_score = _to_float(entry.get("det_score"))

        similarity = _extract_similarity(entry)

        matches_target = bool(match_fn and match_fn(entry))
        if det_th is not None and (det_score is None or det_score < det_th):
            if not matches_target:
                continue

        if match_fn is not None:
            if not matches_target:
                continue
        else:
            if sim_threshold is not None:
                if similarity is None or similarity < sim_threshold:
                    continue
            elif similarity is None:
                continue



        clusters = _coerce_str_set(entry.get("cluster_id"))
        clusters.update(_coerce_str_set(entry.get("cluster_ids")))
        final_ids = _coerce_str_set(entry.get("final_character_id"))
        final_ids.update(_coerce_str_set(entry.get("final_character_ids")))

        detection_summary = _summarise_detection(
            entry,
            timestamp,
            det_score if det_score is not None else 0.0,
            similarity,
            clusters,
            final_ids,
        )

        hits.append(
            {
                "timestamp": float(timestamp),
                "det_score": det_score,
                "similarity": similarity,
                "clusters": clusters,
                "final_ids": final_ids,
                "summary": detection_summary,
            }
        )

    # 🔑 xử lý sau khi gom đủ hits
    if not hits:
        return []

    merge_gap = _to_float(max_gap) or 0.0
    segments = _segment_target_hits(hits, merge_gap)


    highlight_support_meta: Dict[str, Any] = {
        "det_score_threshold": float(det_th) if det_th is not None else det_th,
        "min_duration": float(min_duration) if min_duration is not None else None,
    }
    if sim_threshold is not None:
        highlight_support_meta["similarity_threshold"] = float(sim_threshold)
    if min_score is not None:
        highlight_support_meta["min_score"] = float(min_score)
    highlight_support_meta = {
        key: value for key, value in highlight_support_meta.items() if value is not None
    }


    highlights: List[Dict[str, Any]] = []
    for segment_hits in segments:
        accumulator = _accumulate_segment_hits(segment_hits)
        if not accumulator:
            continue

        highlight = _finalise_highlight(
            accumulator,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
            min_duration=min_duration,
            highlight_support=highlight_support_meta,
        )
        if not highlight:
            continue

        score_value = _to_float(highlight.get("score"))
        if min_score is not None and score_value is not None and score_value < min_score:
            continue

        highlights.append(highlight)

    return highlights



def _limit_highlights_per_scene(
    highlights: List[Dict[str, Any]], limit: int | None
) -> List[Dict[str, Any]]:
    """Giới hạn số highlight theo cấu hình mà vẫn giữ thứ tự thời gian."""

    if not highlights:
        return []
    if limit is None or limit <= 0 or len(highlights) <= limit:
        return list(highlights)

    def _score(item: Dict[str, Any]) -> float:
        for key in ("max_det_score", "max_score", "avg_similarity"):
            value = _to_float(item.get(key))
            if value is not None:
                return float(value)
        return 0.0

    def _match_count(item: Dict[str, Any]) -> int:
        count = item.get("match_count")
        if isinstance(count, (int, np.integer)):
            return int(count)
        try:
            return int(count)
        except (TypeError, ValueError):
            return 0

    ranked = sorted(
        highlights,
        key=lambda item: (
            -_score(item),
            -_match_count(item),
            _parse_time(item.get("start")) or float("inf"),
        ),
    )
    limited = ranked[:limit]
    limited.sort(key=lambda item: _parse_time(item.get("start")) or float("inf"))
    return limited


def _summarise_highlight_support(
    highlights: List[Dict[str, Any]],
    *,
    det_threshold: float,
    similarity_threshold: float | None,
    min_duration: float | None = None,
    min_score: float | None = None
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "highlight_count": len(highlights),
        "det_score_threshold": det_threshold,
        "similarity_threshold": similarity_threshold,
    }

    if min_duration is not None:
        summary["min_duration"] = float(min_duration)
    if min_score is not None:
        summary["min_score"] = float(min_score)


    if not highlights:
        summary["matched_cluster_ids"] = []
        summary["matched_final_character_ids"] = []
        summary["match_count"] = 0
        return summary

    det_scores_max = [h.get("max_det_score") for h in highlights if _to_float(h.get("max_det_score")) is not None]
    det_scores_min = [h.get("min_det_score") for h in highlights if _to_float(h.get("min_det_score")) is not None]
    sim_values = [h.get("avg_similarity") for h in highlights if _to_float(h.get("avg_similarity")) is not None]
    sim_max_values = [h.get("max_similarity") for h in highlights if _to_float(h.get("max_similarity")) is not None]
    sim_min_values = [h.get("min_similarity") for h in highlights if _to_float(h.get("min_similarity")) is not None]

    if det_scores_max:
        summary["max_det_score"] = max(float(v) for v in det_scores_max)
    if det_scores_min:
        summary["min_det_score"] = min(float(v) for v in det_scores_min)
    if sim_values:
        avg = sum(float(v) for v in sim_values) / len(sim_values)
        summary["avg_similarity"] = round(avg, 6)
    if sim_max_values:
        summary["max_similarity"] = max(float(v) for v in sim_max_values)
    if sim_min_values:
        summary["min_similarity"] = min(float(v) for v in sim_min_values)

    clusters = set()
    final_ids = set()
    total_matches = 0
    for highlight in highlights:
        clusters.update(_coerce_str_set(highlight.get("matched_cluster_ids")))
        final_ids.update(_coerce_str_set(highlight.get("matched_final_character_ids")))
        count = highlight.get("match_count")
        if isinstance(count, (int, np.integer)):
            total_matches += int(count)

    summary["matched_cluster_ids"] = sorted(clusters)
    summary["matched_final_character_ids"] = sorted(final_ids)
    summary["match_count"] = total_matches
    return summary




def _resolve_frame_file(
    frames_dir: str,
    frame_idx: int | None,
    sample_names: List[str],
) -> Tuple[str | None, str | None]:
    if frame_idx is None or frame_idx < 0 or not frames_dir:
        return None, None

    sample_names = [name for name in sample_names if isinstance(name, str) and name]
    candidates: List[str] = []
    seen: set[str] = set()
    fallback_widths: set[int] = set()
    fallback_exts: set[str] = set()

    for name in sample_names:
        base, ext = os.path.splitext(name)
        if ext:
            fallback_exts.add(ext)
        matches = list(re.finditer(r"\d+", base))
        if matches:
            for match in matches:
                digits = match.group(0)
                width = len(digits)
                fallback_widths.add(width)
                try:
                    padded = str(int(frame_idx)).zfill(width)
                except (TypeError, ValueError):
                    continue
                candidate = f"{base[:match.start()]}{padded}{base[match.end():]}{ext}"
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
        elif ext:
            try:
                padded = str(int(frame_idx))
            except (TypeError, ValueError):
                continue
            candidate = f"{base}_{padded}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    if not fallback_exts:
        fallback_exts.update(DEFAULT_FRAME_EXTENSIONS)

    if not fallback_widths:
        try:
            fallback_widths.add(len(str(int(frame_idx))))
        except (TypeError, ValueError):
            fallback_widths.add(0)

    for width in sorted(fallback_widths):
        if width <= 0:
            continue
        try:
            padded = str(int(frame_idx)).zfill(width)
        except (TypeError, ValueError):
            continue
        for ext in fallback_exts:
            candidate = f"{padded}{ext}"
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    for candidate in candidates:
        frame_path = os.path.join(frames_dir, candidate)
        if os.path.exists(frame_path):
            return candidate, frame_path

    try:
        str_idx = str(int(frame_idx))
    except (TypeError, ValueError):
        return None, None

    search_widths = {len(str_idx)}.union({w for w in fallback_widths if w > 0})
    for width in sorted(search_widths):
        padded = str_idx.zfill(width)
        pattern = os.path.join(frames_dir, f"*{padded}*")
        for match in sorted(glob.glob(pattern)):
            if os.path.isfile(match):
                return os.path.basename(match), match

    pattern = os.path.join(frames_dir, f"*{str_idx}*")
    for match in sorted(glob.glob(pattern)):
        if os.path.isfile(match):
            return os.path.basename(match), match

    return None, None


def _prepare_track_timeline(
    track_df: pd.DataFrame,
    frames_dir: str | None,
    fps: float | None,
    track_id: Any,
    *,
    final_character_id: Any | None = None,
    reference_embedding: np.ndarray | List[float] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create timeline entries and frame records for a given track."""


    base_dir = frames_dir if frames_dir and os.path.isdir(frames_dir) else None


    reference_vec: np.ndarray | None = None
    if reference_embedding is not None:
        try:
            reference_vec = l2_normalize(_as_array(reference_embedding))
        except Exception:
            reference_vec = None


    timeline_items: List[Dict[str, Any]] = []
    sample_frame_names: List[str] = []
    sample_name_set: set[str] = set()

    def _remember_sample(name: str | None) -> None:
        if not name or not isinstance(name, str):
            return
        if name in sample_name_set:
            return
        sample_name_set.add(name)
        sample_frame_names.append(name)

    for order_idx, row in enumerate(track_df.itertuples()):
        frame_name = getattr(row, "frame", None)
        if not isinstance(frame_name, str) or not frame_name:
            continue

        raw_idx = getattr(row, "frame_index", None)
        frame_idx = None
        if raw_idx is not None and raw_idx == raw_idx:
            try:
                frame_idx = int(raw_idx)
            except (TypeError, ValueError):
                frame_idx = _frame_to_int(frame_name)
        else:
            frame_idx = _frame_to_int(frame_name)

        timestamp = _timestamp_from_frame(frame_idx, fps)
        bbox = _normalize_bbox(getattr(row, "bbox", [])) if hasattr(row, "bbox") else []
        det_score = getattr(row, "det_score", None)
        det_score_val = None
        if det_score is not None and det_score == det_score:
            try:
                det_score_val = float(det_score)
            except (TypeError, ValueError):
                det_score_val = None

        entry: Dict[str, Any] = {
            "order": order_idx,
            "track_id": int(track_id) if track_id == track_id else None,
            "frame": frame_name,
            "frame_index": frame_idx if frame_idx is not None and frame_idx >= 0 else None,
            "timestamp": timestamp,
            "bbox": bbox,
            "is_buffer": False,
        }
        if det_score_val is not None:
            entry["det_score"] = det_score_val

        cluster_value = getattr(row, "cluster_id", None)
        if cluster_value is not None and cluster_value == cluster_value:
            try:
                entry["cluster_id"] = str(cluster_value)
            except Exception:
                pass
        final_value = getattr(row, "final_character_id", None)
        if final_value is not None and final_value == final_value:
            try:
                entry["final_character_id"] = str(final_value)
            except Exception:
                pass
        character_value = getattr(row, "character_id", None)
        if character_value is not None and character_value == character_value:
            try:
                entry["character_id"] = str(character_value)
            except Exception:
                pass
        if final_character_id is not None:
            try:
                entry["scene_final_character_id"] = str(final_character_id)
            except Exception:
                pass

        if reference_vec is not None and hasattr(row, "emb"):
            emb_value = getattr(row, "emb")
            if emb_value is not None:
                try:
                    emb_vec = l2_normalize(_as_array(emb_value))
                except Exception:
                    emb_vec = None
                if emb_vec is not None and emb_vec.size == reference_vec.size:
                    similarity = float(np.dot(emb_vec, reference_vec))
                    if np.isfinite(similarity):
                        entry["actor_similarity"] = similarity

        frame_path = None
        if base_dir:
            frame_path = os.path.join(base_dir, frame_name)
            if not os.path.exists(frame_path):
                frame_path = None
            else:
                _remember_sample(frame_name)

        timeline_items.append(
            {
                "entry": entry,
                "frame_index": frame_idx if frame_idx is not None and frame_idx >= 0 else None,
                "frame_path": frame_path,
                "is_buffer": False,
            }
        )

    if base_dir:
        for item in timeline_items:
            if item.get("frame_path") or item.get("frame_index") is None:
                continue
            frame_name, frame_path = _resolve_frame_file(
                base_dir,
                item.get("frame_index"),
                sample_frame_names,
            )
            if frame_path:
                item["frame_path"] = frame_path
                if frame_name:
                    if not item["entry"].get("frame"):
                        item["entry"]["frame"] = frame_name
                    _remember_sample(frame_name)

    if base_dir and timeline_items:
        fps_value = fps if fps and fps > 0 else DEFAULT_CLIP_FPS
        pre_frames = int(round(PRE_CLIP_BUFFER_SECONDS * fps_value))
        post_frames = int(round(POST_CLIP_BUFFER_SECONDS * fps_value))
        indices = {
            item.get("frame_index")
            for item in timeline_items
            if item.get("frame_index") is not None
        }
        indices = {int(idx) for idx in indices if isinstance(idx, (int, np.integer))}

        if indices:
            clip_start = min(indices)
            clip_end = max(indices)
            buffer_start = max(0, clip_start - pre_frames)
            buffer_end = clip_end + post_frames
            existing = set(indices)

            def _add_range(idx_range: range) -> None:
                for idx in idx_range:
                    if idx < 0 or idx in existing:
                        continue
                    frame_name, frame_path = _resolve_frame_file(
                        base_dir,
                        idx,
                        sample_frame_names,
                    )
                    if not frame_path:
                        continue
                    entry = {
                        "order": None,
                        "track_id": int(track_id) if track_id == track_id else None,
                        "frame": frame_name or os.path.basename(frame_path),
                        "frame_index": idx,
                        "timestamp": _timestamp_from_frame(idx, fps),
                        "bbox": [],
                        "is_buffer": True,
                    }
                    timeline_items.append(
                        {
                            "entry": entry,
                            "frame_index": idx,
                            "frame_path": frame_path,
                            "is_buffer": True,
                        }
                    )
                    existing.add(idx)
                    _remember_sample(frame_name or os.path.basename(frame_path))

            if buffer_start < clip_start:
                _add_range(range(buffer_start, clip_start))
            if buffer_end > clip_end:
                _add_range(range(clip_end + 1, buffer_end + 1))

    if not timeline_items:
        return [], []

    def _sort_key(item: Dict[str, Any]) -> Tuple[int, int]:
        frame_idx = item.get("frame_index")
        if frame_idx is None:
            return (int(1e12), int(item["entry"].get("order", 0) or 0))
        return (int(frame_idx), int(item["entry"].get("order", 0) or 0))

    timeline_items.sort(key=_sort_key)

    timeline: List[Dict[str, Any]] = []
    frame_records: List[Dict[str, Any]] = []

    for new_order, item in enumerate(timeline_items):
        entry = item["entry"]
        entry["order"] = new_order
        entry["is_buffer"] = bool(item.get("is_buffer", False))
        if (not entry.get("frame")) and item.get("frame_path"):
            entry["frame"] = os.path.basename(item["frame_path"])
        timeline.append(entry)

        frame_path = item.get("frame_path")
        if frame_path:
            frame_records.append(
                {
                    "timeline_index": new_order,
                    "frame_path": frame_path,
                    "frame_index": item.get("frame_index"),
                    "is_buffer": bool(item.get("is_buffer", False)),
                }
            )

    return timeline, frame_records

def _select_clip_frames(
    frame_records: List[Dict[str, Any]], fps_value: float
) -> List[Dict[str, Any]]:
    """Select a contiguous window of frames that fits within the target duration."""

    if not frame_records:
        return []

    effective_fps = fps_value if fps_value and fps_value > 0 else DEFAULT_CLIP_FPS
    min_frames = max(1, int(round(MIN_CLIP_DURATION * effective_fps)))
    max_frames = max(min_frames, int(round(MAX_CLIP_DURATION * effective_fps)))

    focus_indices = [
        idx
        for idx, record in enumerate(frame_records)
        if not record.get("is_buffer", False)
    ]
    if not focus_indices:
        focus_indices = list(range(len(frame_records)))

    focus_start = focus_indices[0]
    focus_end = focus_indices[-1]

    pre_buffer_count = sum(
        1 for record in frame_records[:focus_start] if record.get("is_buffer", False)
    )
    post_buffer_count = sum(
        1 for record in frame_records[focus_end + 1 :] if record.get("is_buffer", False)
    )

    required_frames = (focus_end - focus_start + 1) + pre_buffer_count + post_buffer_count
    max_frames = max(max_frames, required_frames)

    if len(frame_records) <= max_frames:
        return frame_records

    window_start = max(0, focus_start - pre_buffer_count)
    window_end = min(len(frame_records) - 1, focus_end + post_buffer_count)

    current_len = window_end - window_start + 1
    if current_len > max_frames:
        excess = current_len - max_frames
        while excess > 0 and window_start < focus_start:
            window_start += 1
            excess -= 1
        while excess > 0 and window_end > focus_end:
            window_end -= 1
            excess -= 1

    current_len = window_end - window_start + 1
    desired_len = min(max_frames, len(frame_records))
    left_capacity = window_start
    right_capacity = len(frame_records) - 1 - window_end

    while current_len < desired_len and (left_capacity > 0 or right_capacity > 0):
        take_left = min(left_capacity, (desired_len - current_len + 1) // 2)
        take_right = min(right_capacity, desired_len - current_len - take_left)
        if take_left:
            window_start -= take_left
            left_capacity -= take_left
            current_len += take_left
        if take_right:
            window_end += take_right
            right_capacity -= take_right
            current_len += take_right
        if take_left == 0 and take_right == 0:
            break

    return frame_records[window_start : window_end + 1]


def _get_highlight_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the user-provided highlight configuration if available."""

    if not isinstance(cfg, dict):
        return {}

    for key in ("highlight", "highlights"):
        candidate = cfg.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


@task(name="Build Character Profiles Task")
def character_task():
    """Xây dựng hồ sơ nhân vật riêng cho từng phim."""

    print("\n--- Starting Character Profile Task ---")
    cfg = load_config()
    storage_cfg = cfg.get("storage", {})
    post_merge_cfg = cfg.get("post_merge", {})
    highlight_cfg = _get_highlight_config(cfg)

    highlight_det_threshold = _parse_time(highlight_cfg.get("det_score_threshold"))
    if highlight_det_threshold is None:
        highlight_det_threshold = DEFAULT_HIGHLIGHT_DET_SCORE

    highlight_gap_seconds = _parse_time(highlight_cfg.get("max_gap_seconds"))
    if highlight_gap_seconds is None:
        highlight_gap_seconds = DEFAULT_HIGHLIGHT_GAP_SECONDS

    similarity_keys = ("min_similarity", "similarity_threshold", "min_similarity_score")
    highlight_similarity_threshold: float | None = None
    for key in similarity_keys:
        highlight_similarity_threshold = _parse_time(highlight_cfg.get(key))
        if highlight_similarity_threshold is not None:
            break
    if highlight_similarity_threshold is None:
        highlight_similarity_threshold = DEFAULT_HIGHLIGHT_SIMILARITY

    clusters_path = storage_cfg["warehouse_clusters"]
    embeddings_path = storage_cfg.get("warehouse_embeddings")
    output_json_path = storage_cfg["characters_json"]
    merged_parquet_path = storage_cfg.get("clusters_merged_parquet")
    frames_root = storage_cfg.get("frames_root")
    video_root = storage_cfg.get("video_root")
    video_root_path: Path | None = None
    if video_root:
        video_root_path = Path(video_root)
        if not video_root_path.is_absolute():
            video_root_path = (Path.cwd() / video_root_path).resolve()
        else:
            video_root_path = video_root_path.resolve()


    print(f"[Character] Loading clustered data from {clusters_path}...")

    clusters_df = pd.read_parquet(clusters_path)
    if clusters_df.empty:
        print("[Character] No clustered data to process. Skipping task.")
        return None

    if "movie_id" not in clusters_df.columns:
        if "movie" in clusters_df.columns:
            clusters_df["movie_id"] = (
                clusters_df["movie"].astype("category").cat.codes.astype(int)
            )
        else:
            clusters_df["movie_id"] = 0

    movie_name_by_id: Dict[int, str] = {}
    if "movie" in clusters_df.columns:
        movie_name_by_id = (
            clusters_df.groupby("movie_id")["movie"].apply(
                lambda values: next(
                    (str(v) for v in values if isinstance(v, str) and v),
                    str(values.iloc[0]) if len(values) else "",
                )
            )
        ).to_dict()

    movie_media_meta: Dict[str, Dict[str, Any]] = {}

    def _store_media_meta(
        keys: List[str],
        *,
        fps_value: float | None = None,
        video_source: str | None = None,
    ) -> None:
        for key in keys:
            if not key:
                continue
            movie_key = str(key)
            entry = movie_media_meta.setdefault(movie_key, {})
            if fps_value is not None:
                entry["fps"] = float(fps_value)
            if video_source:
                entry["video_source"] = video_source

    def _normalise_video_source(path_value: Any) -> str | None:
        if path_value is None:
            return None
        try:
            candidate = Path(str(path_value))
        except Exception:
            return None
        if video_root_path is not None:
            if not candidate.is_absolute():
                candidate = video_root_path / candidate
            try:
                candidate = candidate.resolve()
            except OSError:
                candidate = candidate.absolute()
            try:
                relative = candidate.relative_to(video_root_path)
                return relative.as_posix()
            except ValueError:
                return candidate.as_posix()
        return candidate.as_posix()

    metadata_path = storage_cfg.get("metadata_json")
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"[WARN] Could not read metadata file at {metadata_path}")
        else:
            if isinstance(metadata, dict):
                for movie_key, info in metadata.items():
                    info_dict: Dict[str, Any] = (
                        info if isinstance(info, dict) else {}
                    )
                    fps_candidates = [
                        info_dict.get("fps"),
                        info_dict.get("FPS"),
                    ]
                    video_candidate = info_dict.get("video_source") or info_dict.get(
                        "video_path"
                    )

                    nested_video = info_dict.get("video") or info_dict.get(
                        "video_info"
                    )
                    if isinstance(nested_video, dict):
                        fps_candidates.extend(
                            [nested_video.get("fps"), nested_video.get("FPS")]
                        )
                        for key in ("source", "path", "video_path", "file"):
                            if video_candidate:
                                break
                            value = nested_video.get(key)
                            if isinstance(value, str) and value:
                                video_candidate = value
                    elif isinstance(nested_video, str) and not video_candidate:
                        video_candidate = nested_video

                    for key in ("source", "path", "video", "file"):
                        if video_candidate:
                            break
                        value = info_dict.get(key)
                        if isinstance(value, str) and value:
                            video_candidate = value

                    fps_value = None
                    for candidate in fps_candidates:
                        parsed = _parse_time(candidate)
                        if parsed is not None:
                            fps_value = parsed
                            break

                    video_source = _normalise_video_source(video_candidate)

                    alias_keys = {str(movie_key)}
                    for alias_key in (
                        "movie",
                        "movie_name",
                        "movie_folder",
                        "folder",
                        "name",
                        "title",
                        "movie_id",
                    ):
                        alias_value = info_dict.get(alias_key)
                        if alias_value is None:
                            continue
                        try:
                            alias_str = str(alias_value)
                        except Exception:
                            continue
                        if alias_str:
                            alias_keys.add(alias_str)

                    if fps_value is not None or video_source:
                        _store_media_meta(
                            list(alias_keys),
                            fps_value=fps_value,
                            video_source=video_source,
                        )
    full_embeddings: pd.DataFrame | None = None
    if embeddings_path and os.path.exists(embeddings_path):
        full_embeddings = pd.read_parquet(embeddings_path)
        if not full_embeddings.empty:
            if "movie_id" not in full_embeddings.columns:
                if "movie" in full_embeddings.columns:
                    full_embeddings["movie_id"] = (
                        full_embeddings["movie"].astype("category").cat.codes.astype(int)
                    )
                else:
                    full_embeddings["movie_id"] = 0
            if "frame_index" not in full_embeddings.columns:
                full_embeddings["frame_index"] = full_embeddings["frame"].apply(
                    _frame_to_int
                )
    else:
        print("[WARN] Full embeddings parquet not found – scene metadata will be limited.")

    per_movie_records: List[pd.DataFrame] = []
    centroid_rows: List[Dict[str, Any]] = []
    characters: Dict[str, Dict[str, Any]] = {}

    for movie_id, movie_group in clusters_df.groupby("movie_id"):
        movie_group = movie_group.copy()
        movie_name = movie_name_by_id.get(movie_id, str(movie_id))


        meta_candidates: List[Dict[str, Any]] = []
        if movie_name:
            candidate = movie_media_meta.get(str(movie_name))
            if isinstance(candidate, dict):
                meta_candidates.append(candidate)
        candidate = movie_media_meta.get(str(movie_id))
        if isinstance(candidate, dict):
            meta_candidates.append(candidate)

        movie_meta: Dict[str, Any] = {}
        for candidate in meta_candidates:
            for key, value in candidate.items():
                if value is None:
                    continue
                movie_meta[key] = value

        fps_value = _parse_time(movie_meta.get("fps"))
        if fps_value is None:
            fps_value = _parse_time(movie_meta.get("video_fps"))
        fps = fps_value
        video_source = movie_meta.get("video_source")


        print(f"[Character] Processing movie_id={movie_id} ({movie_name})")
        centroids = (
            movie_group.groupby("cluster_id")["track_centroid"]
            .apply(
                lambda rows: l2_normalize(
                    _mean_vector([_as_array(v) for v in rows if v is not None])
                )
            )
            .reset_index()
        )

        if centroids.empty:
            continue

        centroid_vectors = np.stack(centroids["track_centroid"].to_list()).astype("float32")

        if post_merge_cfg.get("enable", False) and len(centroid_vectors) > 1:
            distance_th = float(post_merge_cfg.get("distance_threshold", 0.35))
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_th,
                metric=post_merge_cfg.get("metric", "cosine"),
                linkage=post_merge_cfg.get("linkage", "average"),
            )
            labels = clusterer.fit_predict(centroid_vectors)
            label_map = {
                lbl: idx for idx, lbl in enumerate(sorted(set(labels.tolist())))
            }
            mapping = {
                row.cluster_id: label_map[label]
                for row, label in zip(centroids.itertuples(), labels.tolist())
            }
        else:
            mapping = {
                row.cluster_id: idx
                for idx, row in enumerate(
                    centroids.sort_values("cluster_id").itertuples()
                )
            }

        movie_group["final_character_id"] = movie_group["cluster_id"].map(mapping).astype(str)
        per_movie_records.append(movie_group)

        movie_characters: Dict[str, Dict[str, Any]] = {}

        if full_embeddings is not None and not full_embeddings.empty:
            movie_embeddings = full_embeddings[full_embeddings["movie_id"] == movie_id]
        else:
            movie_embeddings = pd.DataFrame()

        for final_id, tracks in movie_group.groupby("final_character_id"):
            track_ids = (
                tracks.get("track_id")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            track_vectors = [
                _as_array(v) for v in tracks["track_centroid"] if v is not None
            ]
            if not track_vectors:
                continue
            centroid_vec = l2_normalize(_mean_vector(track_vectors))

            centroid_rows.append(
                {
                    "movie_id": int(movie_id),
                    "movie": movie_name,
                    "character_id": str(final_id),
                    "embedding": centroid_vec.tolist(),
                }
            )

            scenes: List[Dict[str, Any]] = []
            rep_row: pd.Series | None = None
            scene_frame_count = 0

            allowed_clusters: set[str] = set()
            if "cluster_id" in tracks.columns:
                allowed_clusters = {
                    str(val)
                    for val in tracks["cluster_id"].dropna().astype(str).unique().tolist()
                    if str(val)
                }


            movie_frames_dir = None
            if frames_root:
                potential_dir = os.path.join(frames_root, movie_name)
                if os.path.isdir(potential_dir):
                    movie_frames_dir = potential_dir

            if not movie_embeddings.empty and track_ids:
                scene_rows = movie_embeddings["track_id"].isin(track_ids)
                scene_df = movie_embeddings[scene_rows].copy()
                if not scene_df.empty:
                    if "frame_index" not in scene_df.columns:
                        scene_df["frame_index"] = scene_df["frame"].apply(_frame_to_int)
                    scene_df.sort_values(["track_id", "frame_index"], inplace=True)

                    grouped_tracks: List[Tuple[float, Any, pd.DataFrame]] = []
                    for track_key, track_df in scene_df.groupby("track_id"):
                        track_df = track_df.sort_values("frame_index")
                        first_idx = None
                        if "frame_index" in track_df.columns:
                            first_val = track_df["frame_index"].dropna().min()
                            if first_val == first_val:
                                try:
                                    first_idx = float(first_val)
                                except (TypeError, ValueError):
                                    first_idx = None
                        order_value = first_idx if first_idx is not None else float("inf")
                        grouped_tracks.append((order_value, track_key, track_df))

                    grouped_tracks.sort(
                        key=lambda item: (
                            item[0],
                            _safe_slug(item[1], "track"),
                        )
                    )
                    cluster_lookup: Dict[Any, Any] = {}
                    final_lookup: Dict[Any, Any] = {}
                    if "track_id" in tracks.columns:
                        track_meta = tracks.dropna(subset=["track_id"]).drop_duplicates(
                            "track_id"
                        )
                        if "cluster_id" in track_meta.columns:
                            cluster_lookup = track_meta.set_index("track_id")[
                                "cluster_id"
                            ].to_dict()
                            cluster_lookup.update(
                                {str(k): v for k, v in cluster_lookup.items() if k is not None}
                            )
                        if "final_character_id" in track_meta.columns:
                            final_lookup = track_meta.set_index("track_id")[
                                "final_character_id"
                            ].to_dict()
                            final_lookup.update(
                                {str(k): v for k, v in final_lookup.items() if k is not None}
                            )
                    for order_idx, (_, track_key, track_df) in enumerate(grouped_tracks):
                        track_df = track_df.copy()
                        lookup_key_variants = [track_key]
                        try:
                            lookup_key_variants.append(int(track_key))
                        except (TypeError, ValueError):
                            pass
                        try:
                            lookup_key_variants.append(str(track_key))
                        except Exception:
                            pass

                        for variant in lookup_key_variants:
                            if (
                                "cluster_id" not in track_df.columns
                                or track_df["cluster_id"].isna().all()
                            ) and variant in cluster_lookup:
                                track_df["cluster_id"] = cluster_lookup[variant]
                            if (
                                "final_character_id" not in track_df.columns
                                or track_df["final_character_id"].isna().all()
                            ) and variant in final_lookup:
                                track_df["final_character_id"] = final_lookup[variant]

                        timeline_entries, frame_records = _prepare_track_timeline(
                            track_df,
                            movie_frames_dir,
                            fps,
                            track_key,
                            final_character_id=final_id,
                            reference_embedding=centroid_vec,
                        )
                        if not timeline_entries:
                            continue

                        track_clusters: set[str] = set()
                        if "cluster_id" in track_df.columns:
                            track_clusters = {
                                str(val)
                                for val in track_df["cluster_id"]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                                if str(val)
                            }

                        matcher_clusters = track_clusters or allowed_clusters
                        highlight_matcher = _make_highlight_matcher(
                            final_id,
                            matcher_clusters,
                            highlight_similarity_threshold,
                        )
                        highlights = _build_highlights(
                            timeline_entries,
                            det_th=float(highlight_det_threshold),
                            max_gap=float(highlight_gap_seconds),
                            match_fn=highlight_matcher,
                            sim_threshold=(
                                float(highlight_similarity_threshold)
                                if highlight_similarity_threshold is not None
                                else DEFAULT_HIGHLIGHT_SIMILARITY
                            ),
                            min_duration=HIGHLIGHT_MIN_DURATION,
                            min_score=HIGHLIGHT_MIN_SCORE,
                        )
                        highlights = _limit_highlights_per_scene(
                            highlights, TOP_HIGHLIGHTS_PER_SCENE
                        )

                        clip_fps_value = float(fps) if fps else DEFAULT_CLIP_FPS
                        timeline_to_store = timeline_entries

                        if not timeline_to_store:
                            continue
                        base_timestamp = None

                        for seq_idx, entry in enumerate(timeline_to_store):
                            entry["order"] = seq_idx

                            ts_val = _parse_time(entry.get("timestamp"))
                            if base_timestamp is None and ts_val is not None:
                                base_timestamp = ts_val
                            if base_timestamp is not None and ts_val is not None:
                                offset_val = max(ts_val - base_timestamp, 0.0)
                                rounded_offset = round(offset_val, 3)
                                entry["clip_offset"] = rounded_offset
                                entry.setdefault("relative_time", rounded_offset)
                                entry.setdefault("relative_timestamp", rounded_offset)
                                entry.setdefault("relative_offset", rounded_offset)
                            elif clip_fps_value:
                                fallback_offset = round(seq_idx / float(clip_fps_value), 3)
                                entry.setdefault("clip_offset", fallback_offset)
                                entry.setdefault("relative_time", fallback_offset)
                                entry.setdefault("relative_timestamp", fallback_offset)
                                entry.setdefault("relative_offset", fallback_offset)

                        clip_start_entry = timeline_to_store[0]
                        clip_end_entry = timeline_to_store[-1]
                        focus_entry = next(
                            (
                                item
                                for item in timeline_to_store
                                if not item.get("is_buffer")
                            ),
                            clip_start_entry,
                        )
                        clip_start_frame_idx: int | None = None
                        for candidate in timeline_to_store:
                            idx_val = candidate.get("frame_index")
                            if idx_val is None:
                                continue
                            try:
                                clip_start_frame_idx = int(idx_val)
                            except (TypeError, ValueError):
                                clip_start_frame_idx = None
                            else:
                                break

                        clip_start_timeline_idx = clip_start_entry.get("order", 0)
                        clip_end_timeline_idx = clip_end_entry.get(
                            "order", len(timeline_to_store) - 1
                        )

                        width = None
                        height = None
                        for record in frame_records:
                            frame_path = record.get("frame_path")
                            if not frame_path or not os.path.exists(frame_path):
                                continue
                            sample_img = cv2.imread(frame_path)
                            if sample_img is None:
                                continue
                            height, width = sample_img.shape[:2]
                            break

                        scene_frame_count += len(timeline_to_store)

                        start_timestamp = _parse_time(clip_start_entry.get("timestamp"))
                        end_timestamp = _parse_time(clip_end_entry.get("timestamp"))

                        normalised_highlights = normalise_highlights(
                            highlights,
                            merge_gap=float(highlight_gap_seconds),
                            scene_start=start_timestamp,
                            scene_end=end_timestamp,
                            highlight_limit=None,
                            logger=LOGGER,
                            scene_identifier={
                                "movie": movie_name,
                                "track": track_key,
                                "order": order_idx,
                            },
                        )
                        highlights = normalised_highlights

                        focus_timestamp = _parse_time(focus_entry.get("timestamp"))

                        highlight_support = _summarise_highlight_support(
                            highlights,
                            det_threshold=float(highlight_det_threshold),
                            similarity_threshold=highlight_similarity_threshold,
                            min_duration=HIGHLIGHT_MIN_DURATION,
                            min_score=HIGHLIGHT_MIN_SCORE,
                        )
                        if matcher_clusters:
                            highlight_support["allowed_cluster_ids"] = sorted(
                                matcher_clusters
                            )
                        elif allowed_clusters:
                            highlight_support["allowed_cluster_ids"] = sorted(
                                allowed_clusters
                            )
                        if final_id is not None:
                            try:
                                highlight_support["target_final_character_id"] = str(
                                    final_id
                                )
                            except Exception:
                                pass

                        if start_timestamp is not None:
                            start_timestamp = round(start_timestamp, 3)
                        if end_timestamp is not None:
                            end_timestamp = round(end_timestamp, 3)
                        if focus_timestamp is not None:
                            focus_timestamp = round(focus_timestamp, 3)

                        duration_value = None
                        if start_timestamp is not None and end_timestamp is not None:
                            duration_value = max(end_timestamp - start_timestamp, 0.0)
                        if duration_value is None and clip_fps_value:
                            duration_value = len(timeline_to_store) / float(clip_fps_value)
                        if duration_value is not None:
                            duration_value = round(duration_value, 3)

                        scene_timestamp = focus_timestamp
                        if scene_timestamp is None:
                            scene_timestamp = start_timestamp

                        try:
                            track_int = int(track_key)
                        except (TypeError, ValueError):
                            track_int = None

                        scene_entry = {
                            "order": order_idx,
                            "track_id": track_int,
                            "frame": focus_entry.get("frame")
                                     or clip_start_entry.get("frame"),
                            "frame_index": focus_entry.get("frame_index")
                            if focus_entry.get("frame_index") is not None
                            else clip_start_entry.get("frame_index"),
                            "timestamp": scene_timestamp
                            if scene_timestamp is not None
                            else focus_entry.get("timestamp"),
                            "bbox": focus_entry.get("bbox"),
                            "det_score": focus_entry.get("det_score"),
                            "timeline": timeline_to_store,
                            "frame_count": len(timeline_to_store),
                            "clip_path": None,
                            "clip_fps": float(clip_fps_value)
                            if clip_fps_value
                            else None,
                            "duration": duration_value,
                            "end_frame": clip_end_entry.get("frame"),
                            "end_frame_index": clip_end_entry.get("frame_index"),
                            "end_timestamp": end_timestamp
                            if end_timestamp is not None
                            else clip_end_entry.get("timestamp"),
                            "clip_start_frame": clip_start_entry.get("frame"),
                            "clip_start_frame_index": clip_start_frame_idx,
                            "clip_start_timestamp": start_timestamp
                            if start_timestamp is not None
                            else clip_start_entry.get("timestamp"),
                            "clip_start_index": clip_start_timeline_idx,
                            "clip_end_index": clip_end_timeline_idx,
                            "width": int(width) if width else None,
                            "height": int(height) if height else None,
                            "video_source": video_source,
                            "video_start_timestamp": start_timestamp,
                            "video_end_timestamp": end_timestamp,
                            "video_fps": float(fps) if fps else None,
                            "start_time": start_timestamp,
                            "end_time": end_timestamp,
                        }
                        scene_entry["highlights"] = highlights
                        scene_entry["highlight_total"] = len(highlights)
                        scene_entry["highlight_index"] = 0 if highlights else None
                        scene_entry["highlight_merge_gap"] = float(
                            highlight_gap_seconds
                        )
                        scene_entry["highlight_support"] = highlight_support
                        scene_entry["highlight_det_score_threshold"] = float(
                            highlight_det_threshold
                        )
                        scene_entry["highlight_similarity_threshold"] = (
                            float(highlight_similarity_threshold)
                            if highlight_similarity_threshold is not None
                            else None
                        )
                        scenes.append(scene_entry)

                    rep_idx = (
                        scene_df["det_score"].idxmax()
                        if "det_score" in scene_df
                        else scene_df.index[0]
                    )
                    rep_row = scene_df.loc[rep_idx]

                if rep_row is None and not tracks.empty:
                    rep_idx = tracks["det_score"].idxmax() if "det_score" in tracks else tracks.index[0]
                    rep_row = tracks.loc[rep_idx]

                rep_image = {}
                if rep_row is not None:
                    rep_bbox = (
                        rep_row["bbox"] if "bbox" in rep_row else []
                    )
                    rep_image = {
                        "movie": rep_row.get("movie", movie_name),
                        "frame": rep_row.get("frame"),

                        "bbox": _normalize_bbox(rep_bbox) if rep_bbox is not None else [],
                    "det_score": float(rep_row.get("det_score", 0.0)),
                }

            previews_root = storage_cfg.get("cluster_previews_root")
            preview_paths: List[str] = []
            preview_entries: List[Dict[str, Any]] = []
            if previews_root:
                for raw_cluster_id in sorted(tracks["cluster_id"].astype(str).unique()):
                    cluster_dir = os.path.join(previews_root, f"cluster_{raw_cluster_id}")
                    if not os.path.isdir(cluster_dir):
                        continue
                    meta_file = os.path.join(cluster_dir, "metadata.json")
                    if os.path.exists(meta_file):
                        meta_entries = _load_cluster_metadata(
                            meta_file, movie_id, raw_cluster_id
                        )
                        for entry in meta_entries:
                            entry = dict(entry)
                            entry["cluster_id"] = raw_cluster_id
                            preview_img = entry.get("preview_image")
                            annotated_img = entry.get("annotated_image")
                            if preview_img and not os.path.isabs(preview_img):
                                entry["preview_image"] = os.path.join(cluster_dir, preview_img)
                            if annotated_img and not os.path.isabs(annotated_img):
                                entry["annotated_image"] = os.path.join(
                                    cluster_dir, annotated_img
                                )
                            if entry.get("preview_image"):
                                preview_paths.append(entry["preview_image"])
                            preview_entries.append(entry)
                    else:
                        images = [
                            os.path.join(cluster_dir, f)
                            for f in sorted(os.listdir(cluster_dir))
                            if f.lower().endswith((".jpg", ".png"))
                        ]
                        preview_paths.extend(images)

            movie_characters[str(final_id)] = {
                "movie": movie_name,
                "movie_id": int(movie_id),
                "count": int(scene_frame_count)
                if scene_frame_count
                else int(len(track_ids)),
                "track_count": len(track_ids),
                "rep_image": rep_image,
                "preview_paths": preview_paths,
                "previews": preview_entries,
                "scenes": scenes,
                "embedding": centroid_vec.tolist(),
                "raw_cluster_ids": sorted(
                    tracks["cluster_id"].astype(str).unique().tolist()
                ),
            }

        if movie_characters:
            characters[str(int(movie_id))] = movie_characters

    merged_df = pd.concat(per_movie_records, ignore_index=True)

    if merged_parquet_path:
        os.makedirs(os.path.dirname(merged_parquet_path), exist_ok=True)
        pd.DataFrame(centroid_rows).to_parquet(merged_parquet_path, index=False)
        print(f"[Character] Saved centroid data to {merged_parquet_path}")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2, ensure_ascii=False)

    before_clusters = clusters_df["cluster_id"].nunique()
    after_clusters = sum(len(v) for v in characters.values())
    print(
        f"[Character] Clusters before merge: {before_clusters}, after per-movie merge: {after_clusters}"
    )
    print(
        f"[Character] Saved {after_clusters} character profiles grouped across {len(characters)} movies to {output_json_path}"
    )

    filter_clusters_task(merged_df, output_json_path, cfg)

    index_path = storage_cfg.get("index_path")
    if index_path:
        print(f"[Character] Building index at {index_path}...")
        build_index(output_json_path, index_path)

    print("[Character] Task completed successfully ✅")
    return output_json_path
