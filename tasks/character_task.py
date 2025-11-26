# tasks/character_task.py
# !/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import re

import pandas as pd
from prefect import task

from utils.config_loader import load_config


# =====================================================================
# CÁC HÀM HELPER (Đã được cải tiến để ổn định hơn)
# =====================================================================

def _read_metadata(cfg: dict) -> Dict[str, Any]:
    storage = cfg.get("storage", {})
    meta_path = storage.get("metadata_json") or "Data/metadata.json"
    p = Path(meta_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_fps_for_title(meta: Dict[str, Any], title: str) -> float:
    info = meta.get(title) or {}
    fps = info.get("fps")
    try:
        return float(fps) if fps is not None else 24.0
    except Exception:
        return 24.0


def _frames_to_scenes(frames: List[int], fps: float, *, max_gap_s: float = 3.0, pad_s: float = 0.5) -> List[
    Tuple[float, float]]:
    if not frames: return []
    frames = sorted(frames)
    times = [f / max(fps, 1e-6) for f in frames]
    out: List[Tuple[float, float]] = []
    start = times[0]
    prev = times[0]
    for t in times[1:]:
        if (t - prev) > max_gap_s:
            s = max(0.0, start - pad_s)
            e = max(s, prev + pad_s)
            out.append((s, e))
            start = t
        prev = t
    s = max(0.0, start - pad_s)
    e = max(s, prev + pad_s)
    out.append((s, e))
    return out


def _merge_overlaps(intervals: List[Tuple[float, float]], *, tol: float = 0.25) -> List[Tuple[float, float]]:
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + tol:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _limit_scenes(scenes: List[Tuple[float, float]], *, max_scenes: int = 12, min_len_s: float = 1.2) -> List[
    Tuple[float, float]]:
    long_enough = [(s, e) for (s, e) in scenes if (e - s) >= min_len_s]
    long_enough.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    return long_enough[:max_scenes] if max_scenes and max_scenes > 0 else long_enough


def _list_previews(preview_root: Path, title: str, char_id: str, *, limit: int = 12) -> List[str]:
    root = preview_root / title / str(char_id)
    paths: List[str] = []
    if root.exists():
        for p in sorted(root.glob("*.jpg")):
            paths.append(p.as_posix())
            if len(paths) >= limit: break
    return paths


def _load_clusters_parquet(cfg: dict) -> Optional[pd.DataFrame]:
    # --- CẬP NHẬT: Đọc đường dẫn từ config ---
    storage_cfg = cfg.get("storage", {})
    candidates = [
        storage_cfg.get("clusters_merged_parquet"),
        storage_cfg.get("warehouse_clusters"),
    ]
    for p_str in candidates:
        if p_str and Path(p_str).exists():
            try:
                print(f"[Characters] Loading clusters from: {p_str}")
                return pd.read_parquet(p_str)
            except Exception as e:
                print(f"[Characters] Warning: Failed to read {p_str}: {e}")
                pass
    return None


def _coerce_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _extract_frame_int(v: Any) -> Optional[int]:
    if v is None: return None
    iv = _coerce_int(v)
    if iv is not None: return iv
    s = str(v)
    nums = re.findall(r"\d+", s)
    if not nums: return None
    try:
        return int(max(nums, key=len))
    except Exception:
        return None


# =====================================================================
# TASK CHÍNH (Đã sửa lỗi đồng bộ)
# =====================================================================

@task(name="Character Manifest Task")
def character_task(
        filtered_clusters_df: pd.DataFrame | None = None,
        cfg: dict | None = None  # <-- CẬP NHẬT 1: Thêm tham số cfg
) -> str:
    # CẬP NHẬT 2: Ưu tiên cfg được truyền vào
    if cfg is None:
        print("[Characters][WARN] Config không được truyền vào, đang tự đọc lại file config tĩnh.")
        cfg = load_config()

    storage_cfg = cfg.get("storage", {})
    meta = _read_metadata(cfg)

    # CẬP NHẬT 3: Đọc đường dẫn từ config để đảm bảo tính nhất quán
    preview_root = Path(storage_cfg.get("cluster_previews_root", "warehouse/cluster_previews"))
    out_path = Path(storage_cfg.get("characters_json", "warehouse/characters.json"))

    df = filtered_clusters_df
    if df is None or df.empty:
        df = _load_clusters_parquet(cfg)  # Truyền cfg vào helper

    if df is None or df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Characters] No clusters found. Wrote empty manifest -> {out_path}")
        return str(out_path)

    # --- Logic nghiệp vụ còn lại giữ nguyên, vì nó đã đủ mạnh mẽ ---
    char_col = next((c for c in ["final_character_id", "cluster_id"] if c in df.columns), None)
    movie_col = "movie" if "movie" in df.columns else None
    frame_col = next((c for c in ["frame", "image", "image_name"] if c in df.columns), None)

    if not all([char_col, movie_col, frame_col]):
        raise RuntimeError(
            f"[Characters] Thiếu các cột cần thiết. Cần: [movie, character_id, frame]. Có: {list(df.columns)}")

    manifest: Dict[str, Dict[str, Any]] = {}
    gb = df.groupby([movie_col, char_col], dropna=False)

    for (title, cluster_id), sub in gb:
        if not title or pd.isna(title):
            continue
        fps = _get_fps_for_title(meta, str(title))
        frames_int = sorted(set(iv for v in sub[frame_col] if (iv := _extract_frame_int(v)) is not None))
        if not frames_int:
            continue

        raw_scenes = _frames_to_scenes(frames_int, fps=fps, max_gap_s=3.0, pad_s=0.5)
        merged = _merge_overlaps(raw_scenes, tol=0.25)
        final_scenes = _limit_scenes(merged, max_scenes=12, min_len_s=1.2)

        movie_bucket = manifest.setdefault(str(title), {})
        char_id = str(cluster_id)
        previews = _list_previews(preview_root, str(title), char_id, limit=12)
        vurl = f"/videos/{title}"

        scene_payload = [
            {"start_time": float(round(s, 3)), "end_time": float(round(e, 3)), "video_url": vurl}
            for (s, e) in final_scenes
        ]
        movie_bucket[char_id] = {
            "rep_image": previews[0] if previews else None,
            "preview_paths": previews,
            "scenes": scene_payload
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    num_movies = len(manifest)
    num_chars = sum(len(v) for v in manifest.values())
    print(f"[Characters] Wrote {num_chars} characters across {num_movies} movies -> {out_path}")

    return str(out_path)