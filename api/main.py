"""FastAPI application exposing the face recognition service."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.recognition import recognize

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

        return recognize(temp_path)
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