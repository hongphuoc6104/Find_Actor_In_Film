"""FastAPI application exposing the face recognition service."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.recognition import recognize
from utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREVIEWS_ROUTE = "/previews"

logger = logging.getLogger(__name__)


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


PREVIEWS_ROOT = _resolve_previews_root()

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

if PREVIEWS_ROOT:
    PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount(
        PREVIEWS_ROUTE,
        StaticFiles(directory=str(PREVIEWS_ROOT)),
        name="previews",
    )


def _build_preview_url(path: str) -> str:
    """Convert an absolute preview path into an API URL."""

    if PREVIEWS_ROOT is None:
        return path

    preview_path = Path(path)
    if not preview_path.is_absolute():
        preview_path = PREVIEWS_ROOT / preview_path

    try:
        relative = preview_path.relative_to(PREVIEWS_ROOT)
    except ValueError:
        return path

    return f"{PREVIEWS_ROUTE.rstrip('/')}/{relative.as_posix()}"


def _convert_scene_entry(scene: Any) -> Any:
    """Normalise and convert scene metadata for API responses."""

    if scene is None:
        return None

    if not isinstance(scene, dict):
        if isinstance(scene, str):
            scene = {"frame": scene}
        else:
            return scene

    converted = scene.copy()
    frame = converted.get("frame")
    if isinstance(frame, str) and frame:
        converted["frame"] = _build_preview_url(frame)

    return converted


def _convert_preview_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert preview metadata paths to URLs."""

    converted = entry.copy()
    for key in ("preview_image", "annotated_image", "frame", "image", "thumbnail"):
        value = converted.get(key)
        if isinstance(value, str) and value:
            converted[key] = _build_preview_url(value)
    return converted


def _convert_candidate_media(candidate: Any) -> Any:
    """Transform media paths inside a recognition candidate."""

    if not isinstance(candidate, dict):
        return candidate

    converted = candidate.copy()

    preview_paths = converted.get("preview_paths")
    if isinstance(preview_paths, list):
        converted["preview_paths"] = [
            _build_preview_url(path) if isinstance(path, str) and path else path
            for path in preview_paths
        ]

    previews = converted.get("previews")
    if isinstance(previews, list):
        converted["previews"] = [
            _convert_preview_entry(item) if isinstance(item, dict) else item
            for item in previews
        ]

    rep_image = converted.get("rep_image")
    if isinstance(rep_image, dict):
        converted["rep_image"] = _convert_scene_entry(rep_image)

    scene = converted.get("scene")
    if isinstance(scene, dict) or isinstance(scene, str):
        converted["scene"] = _convert_scene_entry(scene)

    scenes = converted.get("scenes")
    if isinstance(scenes, list):
        converted["scenes"] = [
            _convert_scene_entry(item) if isinstance(item, dict) else item
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
        converted["scene"] = _convert_scene_entry(scene)

    return converted


class SceneRequest(BaseModel):
    movie_id: str
    character_id: str
    cursor: int = 0


class UploadRequest(BaseModel):
    movie_id: Optional[str] = None
    path: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    refresh: bool = False


def _run_pipeline_background(payload: Optional[Dict[str, Any]] = None) -> None:
    """Execute the data pipeline in a background task."""

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

    scenes = character.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise HTTPException(status_code=404, detail="No scenes available")

    cursor = request.cursor or 0
    if cursor < 0 or cursor >= len(scenes):
        raise HTTPException(status_code=404, detail="Scene cursor out of range")

    scene_raw = scenes[cursor]
    scene_payload = _convert_scene_entry(scene_raw)
    next_cursor = cursor + 1 if cursor + 1 < len(scenes) else None

    return {
        "movie_id": str(request.movie_id),
        "character_id": str(request.character_id),
        "scene_index": cursor,
        "scene": scene_payload,
        "next_cursor": next_cursor,
        "total_scenes": len(scenes),
        "has_more": next_cursor is not None,
    }


@app.post("/upload")
async def upload_endpoint(
    background_tasks: BackgroundTasks, payload: Optional[UploadRequest] = None
) -> Dict[str, Any]:
    """Trigger the processing pipeline when a new video is uploaded."""

    body = payload.dict(exclude_unset=True) if payload else {}
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
            except OSError:
                pass
