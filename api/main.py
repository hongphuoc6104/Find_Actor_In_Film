"""FastAPI application exposing the face recognition service."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.recognition import recognize
from utils.config_loader import load_config

PREVIEWS_ROUTE = "/previews"


def _resolve_previews_root() -> Path | None:
    """Return the absolute path configured for preview images."""

    storage_cfg = load_config().get("storage", {})
    raw_root = storage_cfg.get("cluster_previews_root")
    if not raw_root:
        return None

    previews_root = Path(raw_root)
    if not previews_root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        previews_root = project_root / previews_root

    return previews_root


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

PREVIEWS_ROOT = _resolve_previews_root()
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


def _convert_preview_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace filesystem paths in the response payload with URLs."""

    if PREVIEWS_ROOT is None:
        return payload

    converted = payload.copy()
    candidates = converted.get("candidates")
    if not isinstance(candidates, list):
        return converted

    converted_candidates: list[Any] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            converted_candidates.append(candidate)
            continue

        candidate_copy = candidate.copy()
        preview_paths = candidate_copy.get("preview_paths")
        if isinstance(preview_paths, list):
            candidate_copy["preview_paths"] = [
                _build_preview_url(path)
                if isinstance(path, str) and path
                else path
                for path in preview_paths
            ]

        converted_candidates.append(candidate_copy)

    converted["candidates"] = converted_candidates
    return converted


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

        return _convert_preview_paths(recognize(temp_path))
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
