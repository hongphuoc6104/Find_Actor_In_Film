"""FastAPI application exposing the face recognition service."""

from __future__ import annotations
import math
import json
import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.recognition import recognize
from utils.config_loader import get_highlight_settings, load_config
from utils.highlights import normalise_highlights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_ROUTE = "/frames"
PREVIEWS_ROUTE = "/previews"
CLIPS_ROUTE = "/clips"
VIDEO_URL_PREFIX: str | None = None
VIDEO_MOUNT_ROUTE: str | None = None
logger = logging.getLogger(__name__)

_HIGHLIGHT_SETTINGS = get_highlight_settings()
_HIGHLIGHT_LIMIT = _HIGHLIGHT_SETTINGS["TOP_K_HL_PER_SCENE"]
_MERGE_GAP = float(_HIGHLIGHT_SETTINGS["MERGE_GAP_SEC"])

def _resolve_frames_root() -> Path | None:
    """Return the absolute path configured for extracted frames."""

    storage_cfg = load_config().get("storage", {})
    raw_root = storage_cfg.get("frames_root")
    if not raw_root:
        return None

    frames_root = Path(raw_root)
    if not frames_root.is_absolute():
        frames_root = PROJECT_ROOT / frames_root

    return frames_root


def _resolve_previews_root() -> Path | None:
    """Return the absolute path configured for preview images."""

    storage_cfg = load_config().get("storage", {})
    raw_root = storage_cfg.get("cluster_previews_root")
    if not raw_root:
        return None

    previews_root = Path(raw_root)
    if not previews_root.is_absolute():
        previews_root = PROJECT_ROOT / previews_root

    return previews_root


def _resolve_scene_clips_root() -> Path | None:
    """Return the absolute path configured for generated scene clips."""

    storage_cfg = load_config().get("storage", {})
    raw_root = storage_cfg.get("scene_clips_root")
    if not raw_root:
        return None

    clips_root = Path(raw_root)
    if not clips_root.is_absolute():
        clips_root = PROJECT_ROOT / clips_root

    return clips_root

def _resolve_video_root() -> Path | None:
    """Return the absolute path configured for source video files."""

    storage_cfg = load_config().get("storage", {})
    raw_root = storage_cfg.get("video_root")
    if not raw_root:
        return None

    video_root = Path(raw_root)
    if not video_root.is_absolute():
        video_root = PROJECT_ROOT / video_root

    return video_root




def _resolve_characters_path() -> Path:
    """Locate the characters metadata file defined in the configuration."""

    storage_cfg = load_config().get("storage", {})
    characters_path = storage_cfg.get("characters_json")
    if not characters_path:
        raise FileNotFoundError("Character metadata path is not configured")

    path = Path(characters_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


@lru_cache(maxsize=1)
def _load_character_metadata() -> Dict[str, Dict[str, Any]]:
    """Load and cache the character metadata grouped by movie."""

    path = _resolve_characters_path()
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    movies: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return movies

    for movie_id, characters in raw.items():
        if not isinstance(characters, dict):
            continue
        movie_key = str(movie_id)
        movies[movie_key] = {}
        for character_id, info in characters.items():
            if isinstance(info, dict):
                movies[movie_key][str(character_id)] = info

    return movies


def _clear_character_cache() -> None:
    """Invalidate the cached metadata to pick up fresh results."""

    _load_character_metadata.cache_clear()


def _get_character_map() -> Dict[str, Dict[str, Any]]:
    """Return the cached character metadata or raise an HTTP error."""

    try:
        return _load_character_metadata()
    except FileNotFoundError as exc:  # pragma: no cover - depends on deployment
        raise HTTPException(
            status_code=503, detail="Character metadata not available"
        ) from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
        raise HTTPException(
            status_code=500, detail="Character metadata is invalid"
        ) from exc


def _list_movies() -> List[Dict[str, Any]]:
    """Produce an overview of movies with available character data."""

    metadata = _get_character_map()
    movies: List[Dict[str, Any]] = []

    for movie_id, characters in metadata.items():
        if not isinstance(characters, dict) or not characters:
            continue

        first_info = next(
            (info for info in characters.values() if isinstance(info, dict)),
            {},
        )
        movie_name = None
        if isinstance(first_info, dict):
            movie_name = (
                first_info.get("movie")
                or first_info.get("movie_name")
                or first_info.get("title")
            )

        character_count = sum(1 for info in characters.values() if isinstance(info, dict))
        scene_count = sum(
            len(info.get("scenes") or [])
            for info in characters.values()
            if isinstance(info, dict)
        )
        preview_count = sum(
            len(info.get("previews") or [])
            for info in characters.values()
            if isinstance(info, dict)
        )

        movies.append(
            {
                "movie_id": movie_id,
                "movie": movie_name,
                "character_count": character_count,
                "scene_count": scene_count,
                "preview_count": preview_count,
            }
        )

    movies.sort(key=lambda item: (item.get("movie") or "", item["movie_id"]))
    return movies


def _get_character(movie_id: str, character_id: str) -> Dict[str, Any] | None:
    """Fetch a character entry from the metadata store."""

    metadata = _get_character_map()
    movie_map = metadata.get(str(movie_id))
    if not isinstance(movie_map, dict):
        return None
    entry = movie_map.get(str(character_id))
    return entry if isinstance(entry, dict) else None


FRAMES_ROOT = _resolve_frames_root()
PREVIEWS_ROOT = _resolve_previews_root()
SCENE_CLIPS_ROOT = _resolve_scene_clips_root()
VIDEO_ROOT = _resolve_video_root()
app = FastAPI(title="Find Actor in Film API")

DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRAMES_ROOT:
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount(
        FRAMES_ROUTE,
        StaticFiles(directory=str(FRAMES_ROOT)),
        name="frames",
    )

if PREVIEWS_ROOT:
    PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount(
        PREVIEWS_ROUTE,
        StaticFiles(directory=str(PREVIEWS_ROOT)),
        name="previews",
    )

if SCENE_CLIPS_ROOT:
    SCENE_CLIPS_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount(
        CLIPS_ROUTE,
        StaticFiles(directory=str(SCENE_CLIPS_ROOT)),
        name="clips",
    )

if VIDEO_ROOT:
    try:
        VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            _derived_video_prefix = VIDEO_ROOT.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            _derived_video_prefix = None

        if _derived_video_prefix:
            VIDEO_URL_PREFIX = _derived_video_prefix
        else:
            VIDEO_URL_PREFIX = "videos"

        VIDEO_MOUNT_ROUTE = f"/{VIDEO_URL_PREFIX.lstrip('/')}"
        app.mount(
            VIDEO_MOUNT_ROUTE,
            StaticFiles(directory=str(VIDEO_ROOT)),
            name="videos",
        )
    except (PermissionError, OSError) as exc:
        logger.warning(
            "Unable to configure video static files for %s: %s", VIDEO_ROOT, exc
        )
        VIDEO_URL_PREFIX = None
        VIDEO_MOUNT_ROUTE = None
else:
    VIDEO_URL_PREFIX = None
    VIDEO_MOUNT_ROUTE = None

def _build_static_url(
    path: str, root: Path | None, route: str | None, *, prefix: str | None = None
) -> str:
    """Convert a filesystem path within ``root`` to a served URL."""

    if root is None:
        return path

    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = root / target_path

    try:
        relative = target_path.relative_to(root)
    except ValueError:
        return path

    relative_str = relative.as_posix()
    base = prefix.rstrip("/") if prefix else (route or "").rstrip("/")
    if not base:
        return relative_str
    return f"{base}/{relative_str}"


def _build_frame_url(
    path: str,
    *,
    movie: str | None = None,
    movie_id: str | int | None = None,
) -> str:
    """Convert an extracted frame path into an API URL."""

    if not path:
        return path

    candidate_path = Path(path)
    if not candidate_path.is_absolute() and len(candidate_path.parts) == 1:
        candidate_prefixes = [
            str(movie).strip() if movie else None,
            str(movie_id).strip() if movie_id is not None else None,
        ]
        for prefix in candidate_prefixes:
            if prefix:
                candidate_path = Path(prefix) / candidate_path
                break

    path_str = (
        str(candidate_path)
        if candidate_path.is_absolute()
        else candidate_path.as_posix()
    )
    return _build_static_url(path_str, FRAMES_ROOT, FRAMES_ROUTE)


def _apply_frame_metadata(
    payload: Dict[str, Any],
    *,
    movie: str | None = None,
    movie_id: str | int | None = None,
) -> Dict[str, Any]:
    """Attach frame filename and URL metadata to ``payload`` in place."""

    if not isinstance(payload, dict):
        return payload

    effective_movie = payload.get("movie") or movie
    effective_movie_id = payload.get("movie_id") or movie_id

    frame_value = payload.get("frame")
    if isinstance(frame_value, str) and frame_value:
        frame_url = _build_frame_url(
            frame_value, movie=effective_movie, movie_id=effective_movie_id
        )
        payload["frame_url"] = frame_url
        payload["frame_name"] = Path(frame_value).name
        payload["frame"] = frame_url
    else:
        frame_url = payload.get("frame_url")
        if isinstance(frame_url, str) and frame_url:
            payload.setdefault("frame_name", Path(frame_url).name)

    return payload

def _parse_float(value: Any) -> float | None:
    """Best-effort conversion of a value to a finite float."""

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
    """Best-effort conversion of a value to ``int``."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _build_preview_url(path: str) -> str:
    """Convert an absolute preview path into an API URL."""

    return _build_static_url(path, PREVIEWS_ROOT, PREVIEWS_ROUTE)


def _build_clip_url(path: str) -> str:
    """Convert a scene clip path into an API URL."""

    return _build_static_url(path, SCENE_CLIPS_ROOT, CLIPS_ROUTE)

def _build_video_url(path: str) -> str:
    """Convert a source video path into an API URL."""

    route = VIDEO_MOUNT_ROUTE or "/videos"
    prefix = VIDEO_URL_PREFIX if VIDEO_URL_PREFIX else None
    return _build_static_url(path, VIDEO_ROOT, route, prefix=prefix)



def _convert_scene_entry(
    scene: Any,
    *,
    movie: str | None = None,
    movie_id: str | int | None = None,
) -> Any:
    """Normalise and convert scene metadata for API responses."""

    if scene is None:
        return None

    source_scene: Dict[str, Any]
    if not isinstance(scene, dict):
        if isinstance(scene, str):
            source_scene = {}
            scene = {"frame": scene}
        else:
            return scene
    else:
        source_scene = scene
        scene = dict(scene)

    scene.setdefault("highlights", source_scene.get("highlights", []))

    converted = scene.copy()
    effective_movie = converted.get("movie") or movie
    effective_movie_id = converted.get("movie_id") or movie_id

    _apply_frame_metadata(
        converted, movie=effective_movie, movie_id=effective_movie_id
    )

    # def _parse_float(value: Any) -> float | None:
    #     if value is None:
    #         return None
    #     try:
    #         result = float(value)
    #     except (TypeError, ValueError):
    #         return None
    #     if not math.isfinite(result):
    #         return None
    #     return float(result)

    # --- Video source / URL ---
    video_source = None
    for key in (
        "video_source",
        "video_path",
        "video",
        "videoFile",
        "video_source_path",
        "source_video",
    ):
        value = converted.get(key)
        if isinstance(value, str) and value:
            video_source = value
            break

    if isinstance(video_source, str) and video_source:
        # video_source = os.path.basename(video_source)
        # converted["video_source"] = video_source
        # video_url = _build_video_url(video_source)
        raw_video_source = video_source
        normalized_source = raw_video_source.replace("\\", "/").strip()
        safe_relative: str | None = None

        if normalized_source:
            if VIDEO_ROOT is not None:
                try:
                    root_path = VIDEO_ROOT.resolve()
                    candidate_path = Path(normalized_source)
                    if candidate_path.is_absolute():
                        resolved_candidate = candidate_path.resolve()
                    else:
                        resolved_candidate = (root_path / candidate_path).resolve()
                    safe_relative = resolved_candidate.relative_to(root_path).as_posix()
                except (OSError, RuntimeError, ValueError):
                    safe_relative = None

            if safe_relative is None and normalized_source:
                pure_path = PurePosixPath(normalized_source)
                cleaned_parts: list[str] = []
                unsafe = False
                for part in pure_path.parts:
                    if part in ("", "."):
                        continue
                    if part == ".." or (
                        len(part) == 2 and part[1] == ":" and part[0].isalpha()
                    ):
                        unsafe = True
                        break
                    cleaned_parts.append(part)
                if not unsafe and cleaned_parts:
                    safe_relative = PurePosixPath(*cleaned_parts).as_posix()

        served_name = safe_relative or os.path.basename(normalized_source)
        if not served_name:
            served_name = normalized_source
        if safe_relative is None and normalized_source and normalized_source != served_name:
            logger.debug(
                "Falling back to basename for unsafe video source '%s' -> '%s'",
                raw_video_source,
                served_name,
            )
        video_url = _build_video_url(served_name)
        converted["video_source"] = video_url
        converted["video_url"] = video_url
        converted["video"] = video_url
    else:
        video_url = converted.get("video_url")

    # --- Start / end / duration ---
    start_time = None
    end_time = None
    duration = None

    start_candidates = (
        converted.get("start_time"),
        converted.get("video_start_timestamp"),
        converted.get("clip_start_timestamp"),
        converted.get("timestamp"),
    )
    for candidate in start_candidates:
        parsed = _parse_float(candidate)
        if parsed is not None:
            start_time = parsed
            break

    end_candidates = (
        converted.get("end_time"),
        converted.get("video_end_timestamp"),
        converted.get("end_timestamp"),
    )
    for candidate in end_candidates:
        parsed = _parse_float(candidate)
        if parsed is not None:
            end_time = parsed
            break

    duration_candidates = (
        converted.get("duration"),
        converted.get("video_duration"),
        converted.get("clip_duration"),
    )
    for candidate in duration_candidates:
        parsed = _parse_float(candidate)
        if parsed is not None:
            duration = parsed
            break

    if duration is None and start_time is not None and end_time is not None:
        diff = end_time - start_time
        if diff >= 0:
            duration = round(diff, 3)

    if start_time is not None:
        converted["start_time"] = start_time
        converted["video_start_timestamp"] = start_time
    if end_time is not None:
        converted["end_time"] = end_time
        converted["video_end_timestamp"] = end_time
    if duration is not None:
        converted["duration"] = duration

    # --- Clip URL ---
    clip_source = converted.get("clip_path") or converted.get("clip")
    if isinstance(clip_source, str) and clip_source:
        clip_url = _build_clip_url(clip_source)
        converted["clip_path"] = clip_source
        converted["clip"] = clip_url
        converted["clip_url"] = clip_url
    else:
        clip_url = converted.get("clip_url")

    # --- Timeline ---
    timeline = converted.get("timeline")
    if isinstance(timeline, list):
        converted["timeline"] = []
        for item in timeline:
            if isinstance(item, dict):
                item_copy = dict(item)
                effective_item_movie = item_copy.get("movie") or effective_movie
                effective_item_movie_id = (
                    item_copy.get("movie_id") or effective_movie_id
                )
                _apply_frame_metadata(
                    item_copy,
                    movie=effective_item_movie,
                    movie_id=effective_item_movie_id,
                )
                converted["timeline"].append(item_copy)
            else:
                converted["timeline"].append(item)

    # --- Highlights ---
    scene_start = (
        _parse_float(converted.get("video_start_timestamp"))
        or _parse_float(converted.get("start_time"))
        or _parse_float(converted.get("clip_start_timestamp"))
    )
    scene_end = (
        _parse_float(converted.get("video_end_timestamp"))
        or _parse_float(converted.get("end_time"))
        or _parse_float(converted.get("clip_end_timestamp"))
    )
    raw_highlights = converted.get("highlights")
    if raw_highlights is None:
        raw_highlights = []
    highlights = normalise_highlights(
        raw_highlights,
        highlight_limit=_HIGHLIGHT_LIMIT,
        merge_gap=_MERGE_GAP,
        scene_start=scene_start,
        scene_end=scene_end,
        logger=logger,
        scene_identifier={
            "scene": converted.get("scene_index"),
            "movie": movie,
            "character": movie_id,
        },
    )
    normalized: List[Dict[str, Any]] = []
    for h in highlights or []:
        if not isinstance(h, dict):
            continue
        entry: Dict[str, Any] = dict(h)

        clusters = h.get("matched_cluster_ids")
        if isinstance(clusters, list):
            entry["matched_cluster_ids"] = [
                str(c)
                for c in clusters
                if c is not None and str(c)
            ]

        final_ids = h.get("matched_final_character_ids")
        if isinstance(final_ids, list):
            entry["matched_final_character_ids"] = [
                str(c)
                for c in final_ids
                if c is not None and str(c)
            ]

        detections = h.get("supporting_detections")
        if isinstance(detections, list):
            support_entries = []
            for det in detections:
                if not isinstance(det, dict):
                    continue
                det_entry: Dict[str, Any] = {}
                timestamp = _parse_float(det.get("timestamp"))
                if timestamp is not None:
                    det_entry["timestamp"] = timestamp
                det_score = _parse_float(det.get("det_score"))
                if det_score is not None:
                    det_entry["det_score"] = det_score
                actor_similarity = _parse_float(det.get("actor_similarity"))
                if actor_similarity is not None:
                    det_entry["actor_similarity"] = actor_similarity
                for key in ("frame", "frame_index", "order", "track_id"):
                    value = det.get(key)
                    if value is not None:
                        det_entry[key] = value
                det_clusters = det.get("cluster_ids")
                if isinstance(det_clusters, list):
                    det_entry["cluster_ids"] = [
                        str(c)
                        for c in det_clusters
                        if c is not None and str(c)
                    ]
                det_final_ids = det.get("final_character_ids")
                if isinstance(det_final_ids, list):
                    det_entry["final_character_ids"] = [
                        str(c)
                        for c in det_final_ids
                        if c is not None and str(c)
                    ]
                for identity_key in (
                    "character_id",
                    "final_character_id",
                    "scene_final_character_id",
                ):
                    value = det.get(identity_key)
                    if value is not None:
                        det_entry[identity_key] = str(value)
                if det_entry:
                    support_entries.append(det_entry)
            if support_entries:
                entry["supporting_detections"] = support_entries
        normalized.append(entry)

    converted["highlights"] = normalized
    highlight_total = len(normalized)
    converted["highlight_total"] = highlight_total
    if converted.get("highlight_index") is None:
        converted["highlight_index"] = 0 if highlight_total else None
    converted.setdefault("highlight_merge_gap", _MERGE_GAP)

    try:
        logger.debug(
            "DEBUG_HL api scene converted",
            extra={
                "scene": converted.get("scene_index"),
                "highlight_total": highlight_total,
                "movie": movie,
            },
        )
    except Exception:  # pragma: no cover - defensive logging
        pass
    support_meta = converted.get("highlight_support")
    if isinstance(support_meta, dict):
        support_copy: Dict[str, Any] = {}
        for key, value in support_meta.items():
            if key in {
                "det_score_threshold",
                "similarity_threshold",
                "min_similarity",
                "max_similarity",
                "avg_similarity",
                "min_det_score",
                "max_det_score",
                "min_duration",
                "min_score",
            }:
                parsed = _parse_float(value)
                if parsed is not None:
                    support_copy[key] = parsed
                continue
            if key in {"highlight_count", "match_count"}:
                try:
                    support_copy[key] = int(value)
                except (TypeError, ValueError):
                    continue
                continue
            if key in {
                "matched_cluster_ids",
                "matched_final_character_ids",
                "allowed_cluster_ids",
            } and isinstance(value, list):
                support_copy[key] = [str(v) for v in value if v is not None and str(v)]
                continue
            if key == "target_final_character_id" and value is not None:
                support_copy[key] = str(value)
                continue
            support_copy[key] = value
        converted["highlight_support"] = support_copy

    return converted


def _build_scene_entries(character: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of scene entries, falling back when highlights are absent."""

    raw_scenes = character.get("scenes")
    prepared: List[Dict[str, Any]] = []
    fallback_scenes: List[Dict[str, Any]] = []

    if isinstance(raw_scenes, list):
        for idx, scene in enumerate(raw_scenes):
            if not isinstance(scene, dict):
                continue
            scene_copy = dict(scene)

            scene_start = (
                _parse_float(scene_copy.get("video_start_timestamp"))
                or _parse_float(scene_copy.get("start_time"))
                or _parse_float(scene_copy.get("clip_start_timestamp"))
            )
            scene_end = (
                _parse_float(scene_copy.get("video_end_timestamp"))
                or _parse_float(scene_copy.get("end_time"))
                or _parse_float(scene_copy.get("clip_end_timestamp"))
            )

            highlights = normalise_highlights(
                scene_copy.get("highlights"),
                highlight_limit=_HIGHLIGHT_LIMIT,
                merge_gap=_MERGE_GAP,
                scene_start=scene_start,
                scene_end=scene_end,
                logger=logger,
                scene_identifier={
                    "scene": scene_copy.get("scene_index", idx),
                    "movie": character.get("movie"),
                    "character": character.get("character_id"),
                },
            )
            scene_copy["highlights"] = highlights
            highlight_total = len(highlights)
            scene_copy["highlight_total"] = highlight_total
            scene_copy.setdefault("highlight_merge_gap", _MERGE_GAP)

            raw_scene_index = scene_copy.get("scene_index")
            try:
                source_scene_index = int(raw_scene_index)
            except (TypeError, ValueError):
                source_scene_index = idx
            scene_copy["scene_index"] = source_scene_index
            scene_copy["source_scene_index"] = source_scene_index

            try:
                logger.debug(
                    "DEBUG_HL api scene prepared",
                    extra={
                        "scene": source_scene_index,
                        "highlight_total": highlight_total,
                        "movie": character.get("movie"),
                    },
                )
            except Exception:  # pragma: no cover - defensive logging
                pass

            if highlight_total:
                for highlight_index in range(highlight_total):
                    scene_variant = dict(scene_copy)
                    scene_variant["highlight_index"] = highlight_index
                    scene_variant["highlight_total"] = highlight_total
                    scene_variant["scene_index"] = len(prepared)
                    scene_variant["source_scene_index"] = source_scene_index
                    highlight = highlights[highlight_index]
                    if isinstance(highlight, dict):
                        start = _parse_float(highlight.get("start"))
                        end = _parse_float(highlight.get("end"))
                        duration = _parse_float(highlight.get("duration"))
                        if start is not None:
                            scene_variant["start_time"] = start
                            scene_variant["video_start_timestamp"] = start
                            scene_variant["clip_start_timestamp"] = start
                        if end is not None:
                            scene_variant["end_time"] = end
                            scene_variant["video_end_timestamp"] = end
                            scene_variant["clip_end_timestamp"] = end
                        if duration is not None:
                            scene_variant["duration"] = duration
                    prepared.append(scene_variant)
            else:
                fallback_scene = dict(scene_copy)
                fallback_scene["highlight_index"] = None
                fallback_scene["highlight_total"] = 0
                fallback_scene["scene_index"] = len(fallback_scenes)
                fallback_scene["source_scene_index"] = source_scene_index
                fallback_scenes.append(fallback_scene)

    if prepared:
        return prepared
    if fallback_scenes:
        return fallback_scenes

    def _coerce_scene(value: Any) -> Dict[str, Any] | None:
        if isinstance(value, dict):
            scene_copy = dict(value)
            scene_copy.setdefault("highlights", [])
            scene_copy.setdefault("highlight_total", 0)
            scene_copy.setdefault("highlight_index", None)
            scene_copy.setdefault("highlight_merge_gap", _MERGE_GAP)
            return scene_copy
        if isinstance(value, str) and value:
            return {
                "frame": value,
                "highlights": [],
                "highlight_total": 0,
                "highlight_index": None,
                "highlight_merge_gap": _MERGE_GAP,
            }
        return None

    def _normalise_scene(scene: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
        scene_copy = dict(scene)
        raw_index = scene_copy.get("scene_index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = fallback_index
        scene_copy.setdefault("scene_index", index)
        scene_copy.setdefault("source_scene_index", index)
        scene_copy.setdefault("highlights", [])
        scene_copy.setdefault("highlight_total", 0)
        scene_copy.setdefault("highlight_index", None)
        scene_copy.setdefault("highlight_merge_gap", _MERGE_GAP)
        return scene_copy

    fallback: List[Dict[str, Any]] = []
    if isinstance(raw_scenes, list):
        for idx, raw_scene in enumerate(raw_scenes):
            scene_dict = _coerce_scene(raw_scene)
            if not scene_dict:
                continue
            fallback.append(_normalise_scene(scene_dict, idx))

    if not fallback:
        single_scene = _coerce_scene(character.get("scene"))
        if single_scene:
            fallback.append(_normalise_scene(single_scene, 0))

    return fallback



def _convert_preview_entry(
    entry: Dict[str, Any],
    *,
    movie: str | None = None,
    movie_id: str | int | None = None,
) -> Dict[str, Any]:
    """Convert preview metadata paths to URLs."""

    converted = entry.copy()
    effective_movie = converted.get("movie") or movie
    effective_movie_id = converted.get("movie_id") or movie_id

    for key in ("preview_image", "annotated_image", "image", "thumbnail"):
        value = converted.get(key)
        if isinstance(value, str) and value:
            converted[key] = _build_preview_url(value)

    _apply_frame_metadata(
        converted, movie=effective_movie, movie_id=effective_movie_id
    )
    return converted


def _convert_candidate_media(candidate: Any) -> Any:
    """Transform media paths inside a recognition candidate."""

    if not isinstance(candidate, dict):
        return candidate

    converted = candidate.copy()
    movie_name = converted.get("movie")
    movie_identifier = converted.get("movie_id")

    preview_paths = converted.get("preview_paths")
    if isinstance(preview_paths, list):
        converted["preview_paths"] = [
            _build_preview_url(path) if isinstance(path, str) and path else path
            for path in preview_paths
        ]

    previews = converted.get("previews")
    if isinstance(previews, list):
        converted["previews"] = [
            _convert_preview_entry(
                item, movie=movie_name, movie_id=movie_identifier
            )
            if isinstance(item, dict)
            else item
            for item in previews
        ]

    rep_image = converted.get("rep_image")
    if isinstance(rep_image, dict):
        converted["rep_image"] = _convert_scene_entry(
            rep_image, movie=movie_name, movie_id=movie_identifier
        )

    scene = converted.get("scene")
    if isinstance(scene, dict) or isinstance(scene, str):
        converted["scene"] = _convert_scene_entry(
            scene, movie=movie_name, movie_id=movie_identifier
        )

    scenes = converted.get("scenes")
    if isinstance(scenes, list):
        converted["scenes"] = [
            _convert_scene_entry(
                item, movie=movie_name, movie_id=movie_identifier
            )
            if isinstance(item, dict)
            else item
            for item in scenes
        ]

    return converted


def _convert_preview_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace filesystem paths in the response payload with URLs."""

    if PREVIEWS_ROOT is None or not isinstance(payload, dict):
        return payload

    converted = payload.copy()

    movies = converted.get("movies")
    if isinstance(movies, list):
        converted_movies: List[Any] = []
        for movie in movies:
            if not isinstance(movie, dict):
                converted_movies.append(movie)
                continue
            movie_copy = movie.copy()
            characters = movie_copy.get("characters")
            if isinstance(characters, list):
                movie_copy["characters"] = [
                    _convert_candidate_media(character)
                    for character in characters
                ]
            converted_movies.append(movie_copy)
        converted["movies"] = converted_movies

    candidates = converted.get("candidates")
    if isinstance(candidates, list):
        converted["candidates"] = [
            _convert_candidate_media(candidate) for candidate in candidates
        ]

    scene = converted.get("scene")
    if isinstance(scene, dict) or isinstance(scene, str):
        converted["scene"] = _convert_scene_entry(
            scene,
            movie=converted.get("movie")
            or converted.get("movie_name"),
            movie_id=converted.get("movie_id"),
        )

    return converted


class SceneRequest(BaseModel):
    movie_id: str
    character_id: str
    cursor: int = 0

class UploadMetadata(BaseModel):
    movie_id: Optional[str] = None
    source: Optional[str] = None
    refresh: bool = False

class UploadRequest(UploadMetadata):
    path: Optional[str] = None
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None



def _run_pipeline_background(payload: Optional[Dict[str, Any]] = None) -> None:
    """Execute the data pipeline in a background task."""

    temp_path: str | None = None
    if isinstance(payload, dict):
        path_candidate = payload.get("path")
        if isinstance(path_candidate, str):
            temp_path = path_candidate


    try:
        from flows.pipeline import main_pipeline
    except Exception as exc:  # pragma: no cover - depends on optional deps
        logger.exception("Failed to import pipeline flow: %s", exc)
        return

    try:
        logger.info("Starting pipeline job", extra={"payload": payload})
        main_pipeline()
        logger.info("Pipeline job completed successfully")
    except Exception as exc:  # pragma: no cover - runtime pipeline errors
        logger.exception("Pipeline execution failed: %s", exc)
    finally:
        _clear_character_cache()
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning("Failed to remove temporary upload: %s", temp_path)

@app.get("/movies")
async def list_movies_endpoint() -> List[Dict[str, Any]]:
    """Return the list of movies that contain recognition data."""

    return _list_movies()


@app.post("/scene")
async def scene_endpoint(request: SceneRequest) -> Dict[str, Any]:
    """Return the scene associated with a character at a given cursor."""

    character = _get_character(request.movie_id, request.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    # scenes = character.get("scenes")
    # if not isinstance(scenes, list) or not scenes:
    #     raise HTTPException(status_code=404, detail="No scenes available")

    expanded_scenes = _build_scene_entries(character)
    if not expanded_scenes:
        raise HTTPException(status_code=404, detail="No scenes available")



    cursor = request.cursor or 0
    # if cursor < 0 or cursor >= len(scenes):
    if cursor < 0 or cursor >= len(expanded_scenes):
        raise HTTPException(status_code=404, detail="Scene cursor out of range")

    # scene_raw = scenes[cursor]
    scene_raw = expanded_scenes[cursor]
    movie_label = (
        character.get("movie")
        or character.get("movie_name")
        or character.get("movie_folder")
    )
    scene_payload = _convert_scene_entry(
        scene_raw, movie=movie_label, movie_id=character.get("movie_id")
    )
    # next_cursor = cursor + 1 if cursor + 1 < len(scenes) else None

    if isinstance(scene_payload, dict):
        scene_payload["scene_index"] = cursor
        if scene_raw.get("highlight_index") is not None:
            scene_payload.setdefault("highlight_index", scene_raw["highlight_index"])
        if scene_raw.get("highlight_total") is not None:
            scene_payload.setdefault("highlight_total", scene_raw["highlight_total"])
        if scene_raw.get("source_scene_index") is not None:
            scene_payload.setdefault(
                "source_scene_index", scene_raw["source_scene_index"]
            )
    total_scenes = len(expanded_scenes)
    next_cursor = cursor + 1 if cursor + 1 < total_scenes else None


    return {
        "movie_id": str(request.movie_id),
        "character_id": str(request.character_id),
        "scene_index": cursor,
        "scene": scene_payload,
        "next_cursor": next_cursor,
        # "total_scenes": len(scenes),
        "total_scenes": total_scenes,
        "has_more": next_cursor is not None,
    }


@app.post("/upload")
async def upload_endpoint(
        background_tasks: BackgroundTasks,
        video: UploadFile | None = File(None),
        movie_id: str | None = Form(None),
        source: str | None = Form(None),
        refresh: bool = Form(False),
        metadata: str | None = Form(None),
) -> Dict[str, Any]:
    """Trigger the processing pipeline when a new video is uploaded."""

    extra_metadata: Dict[str, Any] | None = None
    if metadata:
        try:
            candidate = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid metadata payload") from exc
        else:
            if isinstance(candidate, dict):
                extra_metadata = candidate
            else:
                raise HTTPException(
                    status_code=400, detail="Metadata payload must be a JSON object"
                )

    upload_payload = UploadRequest(
        movie_id=movie_id,
        source=source,
        refresh=refresh,
        metadata=extra_metadata,
    )

    temp_path: str | None = None
    try:
        if video is not None:
            suffix = Path(video.filename or "").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                temp_path = tmp_file.name
                await video.seek(0)
                shutil.copyfileobj(video.file, tmp_file)
                if tmp_file.tell() == 0:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty")

            upload_payload.path = temp_path
            upload_payload.filename = video.filename
    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning("Failed to remove temporary upload: %s", temp_path)
        raise
    finally:
        if video is not None:
            await video.close()

    body = upload_payload.model_dump(exclude_unset=True, exclude_none=True)

    if body.get("refresh"):
        _clear_character_cache()

    background_tasks.add_task(_run_pipeline_background, body or None)

    return {
        "status": "scheduled",
        "detail": "Pipeline execution triggered",
        "payload": body or None,
    }


@app.post("/recognize")
async def recognize_endpoint(image: UploadFile = File(...)) -> Dict[str, Any]:
    """Recognize faces from an uploaded image."""

    temp_path: str | None = None
    suffix = Path(image.filename or "").suffix

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            await image.seek(0)
            shutil.copyfileobj(image.file, tmp_file)
            if tmp_file.tell() == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

        if temp_path is None:
            raise RuntimeError("Failed to persist uploaded image")

        result = recognize(temp_path)
        return _convert_preview_paths(result)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive error path
        raise HTTPException(status_code=500, detail="Recognition failed") from exc
    finally:
        await image.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError as exc:
                logger.warning(
                    "Failed to delete temporary file %s: %s", temp_path, exc
                )
