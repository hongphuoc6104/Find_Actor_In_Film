from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from services.recognition import recognize
from utils.config_loader import load_config

app = FastAPI(title="Face Discovery API", version="0.4")

FRONTEND_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _read_metadata(cfg: dict) -> Dict[str, Any]:
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    meta_path = storage.get("metadata_json") or "Data/metadata.json"
    return _read_json(Path(meta_path))

def _reverse_movie_id_map(meta: Dict[str, Any]) -> Dict[str, str]:
    gen = meta.get("_generated") or {}
    id_map = gen.get("movie_id_map") or {}
    rev: Dict[str, str] = {}
    for title, mid in id_map.items():
        try:
            rev[str(int(mid))] = str(title)
        except Exception:
            pass
    return rev

def _resolve_movie_title(meta: Dict[str, Any], key: str) -> Optional[str]:
    key = str(key).strip()
    if not key:
        return None
    if key in meta and key != "_generated":
        return key
    rev = _reverse_movie_id_map(meta)
    return rev.get(key)

def _guess_video_file(video_root: Path, title: str) -> Optional[Path]:
    for ext in [".mp4", ".mkv", ".mov", ".avi"]:
        p = (video_root / f"{title}{ext}").resolve()
        if p.exists():
            return p
    # fallback: bất kỳ file có stem == title
    if video_root.exists():
        for p in video_root.iterdir():
            if p.is_file() and p.stem == title:
                return p.resolve()
    return None

def _get_video_path(cfg: dict, movie_or_id: str) -> Optional[Path]:
    meta = _read_metadata(cfg)
    title = _resolve_movie_title(meta, movie_or_id)
    if not title:
        return None
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    video_root = Path(storage.get("video_root") or "Data/video").resolve()

    movie_info = meta.get(title) or {}
    meta_path = movie_info.get("video_path")
    if meta_path:
        p = Path(meta_path)
        if not p.is_absolute():
            p = (Path(os.getcwd()) / p).resolve()
        if p.exists():
            return p
    return _guess_video_file(video_root, title)

def _list_movies_payload(cfg: dict) -> List[Dict[str, Any]]:
    """
    Trả danh sách phim + thống kê:
      - num_characters: số cụm nhân vật tìm được
      - num_scenes: tổng số khoảng xuất hiện của tất cả nhân vật trong phim
      - num_previews: số ảnh preview trong thư mục warehouse/cluster_previews/<movie>
    Hỗ trợ cả 2 schema characters.json:
      A) {"characters":[{"movie":"...", "character_id":"...", "scenes":[...]}, ...]}
      B) {"<movie>": {"<char_id>": {"preview_paths":[...], "scenes":[...]}, ...}, ...}
    """
    meta = _read_metadata(cfg)
    gen = meta.get("_generated") or {}
    id_map = gen.get("movie_id_map") or {}

    ch_path = Path("warehouse/characters.json")
    ch = _read_json(ch_path)

    # Chuẩn hoá về dict: movie_title -> list[{"character_id":..., "scenes":[...]}]
    char_by_movie: Dict[str, List[dict]] = {}

    if isinstance(ch.get("characters"), list):
        # Schema A
        for c in ch["characters"]:
            mv = str(c.get("movie") or c.get("movie_title") or "").strip()
            if not mv:
                continue
            item = {
                "character_id": str(c.get("character_id")),
                "scenes": c.get("scenes") or []
            }
            char_by_movie.setdefault(mv, []).append(item)
    else:
        # Schema B (như file bạn đang có)
        for mv, chars in (ch or {}).items():
            if mv == "_generated":
                continue
            if not isinstance(chars, dict):
                continue
            for cid, payload in chars.items():
                item = {
                    "character_id": str(cid),
                    "scenes": (payload or {}).get("scenes") or []
                }
                char_by_movie.setdefault(mv, []).append(item)

    previews_root = Path("warehouse/cluster_previews")
    def count_previews(title: str) -> int:
        d = previews_root / title
        if not d.exists():
            return 0
        return sum(1 for p in d.glob("**/*") if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"])

    movies: List[Dict[str, Any]] = []
    for title, payload in meta.items():
        if title == "_generated":
            continue
        # movie_id
        movie_id = None
        try:
            if title in id_map:
                movie_id = int(id_map[title])
        except Exception:
            movie_id = None

        chars = char_by_movie.get(title, [])
        num_characters = len(chars)
        num_scenes = sum(len(c.get("scenes") or []) for c in chars)

        movies.append({
            "movie_id": movie_id,
            "movie": title,
            "label": title,
            "video_path": (payload or {}).get("video_path", ""),
            "num_characters": num_characters,
            "num_scenes": num_scenes,
            "num_previews": count_previews(title),
        })

    movies.sort(key=lambda m: str(m.get("movie") or ""))
    return movies


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class RecognizeResponse(BaseModel):
    is_unknown: bool
    bucket: Optional[str] = None
    movies: List[Dict[str, Any]] = []
    thresholds: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/movies")
def list_movies() -> JSONResponse:
    cfg = load_config()
    return JSONResponse({"movies": _list_movies_payload(cfg)})

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_endpoint(
    image: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50),
    # ↓ hạ mặc định để test cho dễ
    min_score: float = Query(0.25, ge=0.0, le=1.0),
) -> JSONResponse:
    content = await image.read()
    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_path = tmp_dir / image.filename
    with open(img_path, "wb") as f:
        f.write(content)

    payload = recognize(str(img_path), top_k=top_k)

    # --- debug: in ra raw matches trước khi lọc
    try:
        if isinstance(payload, dict) and isinstance(payload.get("movies"), list):
            raw = [(m.get("movie"), round((m.get("score") or m.get("similarity") or 0.0), 3))
                   for m in payload["movies"]]
            print(f"[Recognize][RAW topK] {raw}")
    except Exception as e:
        print(f"[Recognize] raw-log error: {e}")

    # --- filter theo min_score
    try:
        if isinstance(payload, dict) and isinstance(payload.get("movies"), list):
            filtered = []
            for m in payload["movies"]:
                sc = m.get("score") or m.get("similarity") or 0.0
                if sc is None:
                    sc = 0.0
                if sc >= min_score:
                    filtered.append(m)
            print(f"[Recognize] Matches >= {min_score}: {[(x.get('movie'), round(x.get('score',0),3)) for x in filtered]}")
            payload["movies"] = filtered
    except Exception as e:
        print(f"[Recognize] post-filter error: {e}")

    return JSONResponse(payload)


@app.get("/videos/{movie}")
def get_video(movie: str):
    """Phát video gốc theo movie title hoặc id."""
    cfg = load_config()
    p = _get_video_path(cfg, movie)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail=f"Video not found for '{movie}'")

    ext = p.suffix.lower()
    media = "video/mp4"
    if ext == ".mkv":
        media = "video/x-matroska"
    elif ext == ".mov":
        media = "video/quicktime"
    elif ext == ".avi":
        media = "video/x-msvideo"

    return FileResponse(path=str(p), media_type=media, filename=p.name)

@app.get("/video_url/{movie}")
def get_video_url(movie: str):
    cfg = load_config()
    meta = _read_metadata(cfg)
    title = _resolve_movie_title(meta, movie) or movie
    return {"movie": title, "url": f"/videos/{title}"}
