# tasks/character_task.py
# !/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import re
import os

import pandas as pd
from prefect import task

from utils.config_loader import load_config


# =====================================================================
# CÁC HÀM HELPER
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


def _list_previews(preview_root: Path, title: str, char_id: str, limit: int = 12) -> List[str]:
    root = preview_root / title / str(char_id)
    paths: List[str] = []
    if root.exists():
        files = sorted(list(root.glob("*.jpg")) + list(root.glob("*.png")))
        for p in files:
            paths.append(p.as_posix())
            if len(paths) >= limit: break
    return paths


def _load_clusters_parquet(cfg: dict) -> Optional[pd.DataFrame]:
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


def _extract_frame_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        s = str(v)
        nums = re.findall(r"\d+", s)
        if not nums: return None
        return int(max(nums, key=len))


# =====================================================================
# TASK CHÍNH
# =====================================================================

@task(name="Character Manifest Task")
def character_task(
        filtered_clusters_df: pd.DataFrame | None = None,
        cfg: dict | None = None
) -> str:
    """
    Tạo file characters.json từ dữ liệu cluster.
    [CRITICAL FIX]: Load file cũ trước khi update để tránh mất dữ liệu của các phim khác.
    """
    if cfg is None:
        cfg = load_config()

    storage_cfg = cfg.get("storage", {})
    meta = _read_metadata(cfg)

    preview_root = Path(storage_cfg.get("cluster_previews_root", "warehouse/cluster_previews"))
    out_path = Path(storage_cfg.get("characters_json", "warehouse/characters.json"))

    # 1. Load Data
    df = filtered_clusters_df
    if df is None or df.empty:
        df = _load_clusters_parquet(cfg)

    # Nếu không có dữ liệu input thì dừng sớm, nhưng KHÔNG được ghi đè file rỗng
    if df is None or df.empty:
        print(f"[Characters] No clusters found. Skipping manifest update.")
        return str(out_path)

    # 2. [FIXED] Load Manifest cũ (để giữ lại phim khác)
    manifest: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        try:
            manifest = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Characters] Warning: Could not read existing manifest ({e}). Starting fresh.")
            manifest = {}

    # 3. [FIXED] Backup nhãn của bộ phim HIỆN TẠI (nếu đã có trong manifest cũ)
    # Lý do: Ta sắp ghi đè dữ liệu của phim hiện tại (manifest[title] = ...), nên cần cứu lại tên diễn viên nếu có.
    preserved_labels = {}
    # Lấy danh sách phim có trong dataframe hiện tại
    active_movies_in_df = set()
    movie_col = "movie" if "movie" in df.columns else "movie_id"
    if movie_col in df.columns:
        active_movies_in_df = set(df[movie_col].astype(str).unique())

    for m_title in active_movies_in_df:
        if m_title in manifest:
            m_chars = manifest[m_title]
            for c_id, c_data in m_chars.items():
                if c_data.get("name") and c_data.get("name") != "Unknown":
                    key = (str(m_title), str(c_id))
                    preserved_labels[key] = {
                        "name": c_data.get("name"),
                        "label_status": c_data.get("label_status", "MANUAL"),
                        "label_confidence": c_data.get("label_confidence", 1.0)
                    }

    if preserved_labels:
        print(f"[Characters] Preserving {len(preserved_labels)} existing labels for re-processed movies.")

    # 4. Process Clusters
    char_col = next((c for c in ["final_character_id", "cluster_id"] if c in df.columns), None)
    frame_col = next((c for c in ["frame", "image", "image_name", "frame_idx"] if c in df.columns), None)

    if not all([char_col, movie_col, frame_col]):
        print(f"[Characters] Missing columns in dataframe: {df.columns}. Skipping.")
        return str(out_path)

    # Group by Movie & Character
    gb = df.groupby([movie_col, char_col], dropna=False)

    # Dictionary tạm để chứa dữ liệu mới của các phim đang xử lý
    new_data_buffer: Dict[str, Dict[str, Any]] = {}

    for (title_val, cluster_id), sub in gb:
        title = str(title_val)
        if not title or title == "nan": continue

        # Init bucket cho phim nếu chưa có trong buffer mới
        if title not in new_data_buffer:
            new_data_buffer[title] = {}

        fps = _get_fps_for_title(meta, title)
        frames_int = sorted(set(iv for v in sub[frame_col] if (iv := _extract_frame_int(v)) is not None))

        raw_scenes = _frames_to_scenes(frames_int, fps=fps, max_gap_s=3.0, pad_s=0.5)
        merged = _merge_overlaps(raw_scenes, tol=0.25)
        final_scenes = _limit_scenes(merged, max_scenes=12, min_len_s=1.2)

        c_id_str = str(cluster_id)
        previews = _list_previews(preview_root, title, c_id_str, limit=12)
        vurl = f"/videos/{title}"

        scene_payload = [
            {"start_time": float(round(s, 3)), "end_time": float(round(e, 3)), "video_url": vurl}
            for (s, e) in final_scenes
        ]

        char_data = {
            "name": "Unknown",
            "label_status": "UNLABELED",
            "label_confidence": 0.0,
            "rep_image": previews[0] if previews else None,
            "preview_paths": previews,
            "scenes": scene_payload
        }

        # Restore Preserved Label
        pres_key = (title, c_id_str)
        if pres_key in preserved_labels:
            saved = preserved_labels[pres_key]
            char_data["name"] = saved["name"]
            char_data["label_status"] = saved["label_status"]
            char_data["label_confidence"] = saved["label_confidence"]

        new_data_buffer[title][c_id_str] = char_data

    # 5. Merge & Write Atomic
    # Chỉ cập nhật những phim có trong new_data_buffer, giữ nguyên các phim khác trong manifest
    for m_title, m_data in new_data_buffer.items():
        manifest[m_title] = m_data
        print(f"[Characters] Updated manifest for movie '{m_title}' with {len(m_data)} characters.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, out_path)

    total_chars = sum(len(v) for v in manifest.values())
    print(f"[Characters] Total DB: {len(manifest)} movies, {total_chars} characters -> {out_path}")

    return str(out_path)