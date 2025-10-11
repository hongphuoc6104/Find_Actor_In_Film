# tasks/preview_clusters_task.py
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from prefect import task

from utils.config_loader import load_config


# ... (Toàn bộ các hàm helper từ _atomic_write_json đến _crop_from_frame_or_video giữ nguyên) ...
def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _invert_movie_map(meta_path: Path) -> Dict[int, str]:
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        m = (data.get("_generated") or {}).get("movie_id_map") or {}
        return {int(v): str(k) for k, v in m.items()}
    except Exception:
        return {}


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _read_frame_from_video(video_path: Path, frame_idx: int, fps: float) -> Optional[np.ndarray]:
    if not video_path.exists(): return None
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    if fps and fps > 0:
        sec = float(frame_idx) / float(fps)
        cap.set(cv.CAP_PROP_POS_MSEC, sec * 1000.0)
    else:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_FRAME_PATTERNS = [
    "frame_{:07d}.jpg", "{:07d}.jpg", "{:06d}.jpg", "{:05d}.jpg", "{}.jpg",
    "frame_{:07d}.jpeg", "{:07d}.jpeg", "{}.jpeg", "frame_{:07d}.png", "{:07d}.png", "{}.png",
]
_FRAME_INT_COLS = ["frame", "frame_idx", "frame_index", "frame_num", "fid", "image_id"]
_FRAME_NAME_COLS = ["frame_filename", "image_name", "filename", "frame_path", "image_path", "crop_path",
                    "face_crop_path"]


def _build_dir_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if not root.exists(): return idx
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            idx[p.name] = p
    return idx


def _parse_frame_number_from_path(path_str: str) -> Optional[int]:
    if not isinstance(path_str, str) or not path_str: return None
    base = os.path.basename(path_str)
    m = re.search(r"(\d{5,8})", base)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _find_by_frame_number(frames_root_movie: Path, frames_idx: Dict[str, Path], frame_num: int) -> Optional[Path]:
    for pat in _FRAME_PATTERNS:
        name = pat.format(frame_num)
        p = frames_root_movie / name
        if p.exists(): return p
        p2 = frames_idx.get(name)
        if p2: return p2
    digits = str(frame_num)
    rx = re.compile(rf".*{re.escape(digits)}.*", re.IGNORECASE)
    for name, path in frames_idx.items():
        if rx.match(name): return path
    return None


def _crop_from_frame_or_video(img: np.ndarray, row: pd.Series) -> Optional[np.ndarray]:
    if img is None: return None
    h, w = img.shape[:2]
    x1, y1 = row.get("x1", row.get("left")), row.get("y1", row.get("top"))
    x2, y2 = row.get("x2", row.get("right")), row.get("y2", row.get("bottom"))
    try:
        if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
            x1, y1, x2, y2 = _clamp_bbox(int(x1), int(y1), int(x2), int(y2), w, h)
            face = img[y1:y2, x1:x2].copy()
            return face if face.size else None
    except Exception:
        return None
    return None


# ------------------------------- main task ---------------------------------- #

# THAY ĐỔI 1: Nhận dataframe đã được lọc làm tham số
@task(name="Preview Clusters Task")
def preview_clusters_task(filtered_clusters_df: pd.DataFrame | None = None, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    storage = cfg["storage"]
    previews_root = Path(storage["cluster_previews_root"])
    meta_path = Path(storage["metadata_json"])
    movie_id_to_label = _invert_movie_map(meta_path)

    # THAY ĐỔI 2: Ưu tiên sử dụng dataframe được truyền vào
    df = filtered_clusters_df
    if df is None or df.empty:
        # Fallback: đọc file clusters_merged nếu không có dataframe
        clusters_path = Path(storage["clusters_merged_parquet"])
        if not clusters_path.exists():
            print(f"[Preview] Missing clusters parquet: {clusters_path}")
            return str(previews_root)
        df = pd.read_parquet(clusters_path)

    if df.empty:
        print("[Preview] Empty clusters dataframe.")
        return str(previews_root)

    # ... (Phần logic còn lại của task giữ nguyên, vì nó đã đọc từ dataframe 'df') ...
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    target_movie_id: Optional[int] = None
    if active_movie:
        print(f"[Preview] Single-movie mode → '{active_movie}'")
        for mid, name in movie_id_to_label.items():
            if name == active_movie:
                target_movie_id = mid
                break
        if target_movie_id is None:
            print(f"[Preview] Could not resolve movie_id for '{active_movie}'. Skipping.")
            return str(previews_root)

    prefer_source = (cfg.get("preview") or {}).get("source", "frames").lower()
    max_images = int((cfg.get("preview") or {}).get("max_images_per_cluster", 24))
    frames_root = Path(storage.get("frames_root", "Data/frames"))
    face_crops_root = Path(storage.get("face_crops_root", "Data/face_crops"))
    videos_root = Path(storage.get("video_root", "Data/video"))

    if target_movie_id is not None and "movie_id" in df.columns:
        before = len(df)
        df = df[df["movie_id"] == target_movie_id]
        print(f"[Preview] Filtered to movie_id={target_movie_id}: {before} → {len(df)} rows")
        if df.empty:
            print(f"[Preview] No clusters for movie '{active_movie}'.")
            return str(previews_root)

    char_col = "final_character_id" if "final_character_id" in df.columns else "cluster_id"
    if char_col not in df.columns:
        print(f"[Preview] No character column found (final_character_id/cluster_id).")
        return str(previews_root)

    cols = set(df.columns)
    has_any_frame_int = any(c in cols for c in _FRAME_INT_COLS)
    has_any_frame_name = any(c in cols for c in _FRAME_NAME_COLS)
    print(
        f"[Preview] Columns: int_frame={has_any_frame_int}, name_like={has_any_frame_name}, has_bbox={all(k in cols for k in ['x1', 'y1', 'x2', 'y2'])}")

    groups = df.groupby(["movie_id", char_col], dropna=False)
    total_clusters = 0
    made_images = 0
    skipped_no_image = 0
    skipped_no_bbox = 0
    frames_index_cache: Dict[str, Dict[str, Path]] = {}
    crops_index_cache: Dict[str, Dict[str, Path]] = {}

    for (movie_id, char_id), g in groups:
        total_clusters += 1
        movie_label = movie_id_to_label.get(int(movie_id), str(movie_id))
        cluster_dir = previews_root / movie_label / str(char_id)
        _ensure_dir(cluster_dir)

        frames_dir = frames_root / movie_label
        if movie_label not in frames_index_cache: frames_index_cache[movie_label] = _build_dir_index(frames_dir)
        frames_idx = frames_index_cache[movie_label]
        if movie_label not in crops_index_cache: crops_index_cache[movie_label] = _build_dir_index(
            face_crops_root / movie_label)
        crops_idx = crops_index_cache[movie_label]

        candidates = g.copy()
        sort_keys = []
        if "det_score" in candidates.columns: sort_keys.append(("det_score", False))
        if "blur_score" in candidates.columns: sort_keys.append(("blur_score", False))
        if sort_keys:
            by, ascending = [k for k, _ in sort_keys], [asc for _, asc in sort_keys]
            candidates = candidates.sort_values(by=by, ascending=ascending)
        candidates = candidates.head(max_images)

        meta_entries: List[Dict[str, Any]] = []
        for _, row in candidates.iterrows():
            out_name = f"{len(meta_entries):03d}.jpg"
            out_path = cluster_dir / out_name
            img: Optional[np.ndarray] = None
            src_label, src_path_str = None, None
            frame_num: Optional[int] = None

            for c in _FRAME_INT_COLS:
                if c in row and pd.notna(row[c]):
                    try:
                        frame_num = int(row[c]); break
                    except Exception:
                        pass
            if frame_num is None:
                for c in ["frame_path", "image_path", "frame_filename", "filename", "image_name"]:
                    val = row.get(c)
                    fn = _parse_frame_number_from_path(val) if isinstance(val, str) else None
                    if fn is not None: frame_num = fn; break

            if frame_num is not None:
                p = _find_by_frame_number(frames_dir, frames_idx, frame_num)
                if p and p.exists(): img, src_label, src_path_str = cv.imread(str(p)), "frames", str(p)

            if img is None:
                for c in _FRAME_NAME_COLS:
                    val = row.get(c)
                    if isinstance(val, str) and val:
                        base = os.path.basename(val)
                        cand = frames_idx.get(base) or (frames_dir / base)
                        if cand.exists(): img, src_label, src_path_str = cv.imread(str(cand)), "frames", str(
                            cand); break

            if img is None:
                for c in ["crop_path", "face_crop_path", "image_path", "filename", "image_name"]:
                    val = row.get(c)
                    if isinstance(val, str) and val:
                        base = os.path.basename(val)
                        cand = crops_idx.get(base) or (face_crops_root / movie_label / base)
                        if cand.exists():
                            shutil.copy2(cand, out_path)
                            meta_entries.append(
                                {"source": "crops", "src_path": str(cand), "out": out_name, "frame": frame_num,
                                 "track_id": int(row.get("track_id")) if pd.notna(row.get("track_id")) else None})
                            made_images += 1
                            src_label = None
                            break

            if src_label is None and not out_path.exists():
                if frame_num is not None:
                    for ext in (".mp4", ".mkv", ".mov", ".avi"):
                        cand = videos_root / f"{movie_label}{ext}"
                        if cand.exists():
                            fps = row.get("fps")
                            if not (isinstance(fps, (int, float)) and fps > 0):
                                try:
                                    meta = json.loads(Path(storage["metadata_json"]).read_text(encoding="utf-8"))
                                    fps_map = (meta.get("_generated") or {}).get("fps_map") or {}
                                    fps = fps_map.get(movie_label, 25.0)
                                except Exception:
                                    fps = 25.0
                            img, src_label, src_path_str = _read_frame_from_video(cand, int(frame_num),
                                                                                  float(fps)), "video", str(cand)
                            break

            if img is None and not out_path.exists():
                skipped_no_image += 1;
                continue

            if out_path.exists(): continue

            face = _crop_from_frame_or_video(img, row)
            if face is None: face, skipped_no_bbox = img, skipped_no_bbox + 1

            cv.imwrite(str(out_path), face)
            meta_entries.append(
                {"source": src_label or "unknown", "src_path": src_path_str, "out": out_name, "frame": frame_num,
                 "track_id": int(row.get("track_id")) if pd.notna(row.get("track_id")) else None})
            made_images += 1

        _atomic_write_json(cluster_dir / "metadata.json",
                           {"movie_id": str(movie_id), "movie": movie_label, "character_id": str(char_id),
                            "items": meta_entries})

    print(
        f"[Preview] Done: clusters={total_clusters}, images_written={made_images}, root={previews_root}; skipped_no_image={skipped_no_image}, skipped_no_bbox={skipped_no_bbox}")
    return str(previews_root)