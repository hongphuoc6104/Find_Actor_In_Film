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
        # Lấy cả ảnh jpg và png
        files = sorted(list(root.glob("*.jpg")) + list(root.glob("*.png")))
        # Ưu tiên ảnh có metadata (nếu cần logic phức tạp hơn thì đọc metadata.json)
        for p in files:
            paths.append(p.as_posix())
            if len(paths) >= limit: break
    return paths


def _load_clusters_parquet(cfg: dict) -> Optional[pd.DataFrame]:
    storage_cfg = cfg.get("storage", {})
    # Ưu tiên file merged, sau đó đến file clusters gốc
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
    [FIXED] Có logic bảo lưu nhãn (Label Persistence) để tránh mất dữ liệu khi chạy lại.
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

    if df is None or df.empty:
        print(f"[Characters] No clusters found. Writing empty manifest.")
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("{}", encoding="utf-8")
        return str(out_path)

    # 2. [FIX] Đọc file cũ để backup nhãn (Label Persistence)
    preserved_labels = {}
    if out_path.exists():
        try:
            old_data = json.loads(out_path.read_text(encoding="utf-8"))
            for m_title, m_chars in old_data.items():
                for c_id, c_data in m_chars.items():
                    # Chỉ backup những nhân vật đã có tên (không phải Unknown hoặc rỗng)
                    if c_data.get("name") and c_data.get("name") != "Unknown":
                        key = (str(m_title), str(c_id))
                        preserved_labels[key] = {
                            "name": c_data.get("name"),
                            "label_status": c_data.get("label_status", "MANUAL"),
                            "label_confidence": c_data.get("label_confidence", 1.0)
                        }
            if preserved_labels:
                print(f"[Characters] Found {len(preserved_labels)} labeled characters to preserve.")
        except Exception as e:
            print(f"[Characters] Warning: Could not read existing manifest to preserve labels: {e}")

    # 3. Process Clusters
    char_col = next((c for c in ["final_character_id", "cluster_id"] if c in df.columns), None)
    movie_col = "movie" if "movie" in df.columns else "movie_id"
    frame_col = next((c for c in ["frame", "image", "image_name", "frame_idx"] if c in df.columns), None)

    if not all([char_col, movie_col, frame_col]):
        print(f"[Characters] Missing columns in dataframe: {df.columns}. Skipping.")
        return str(out_path)

    manifest: Dict[str, Dict[str, Any]] = {}

    # Group by Movie & Character
    gb = df.groupby([movie_col, char_col], dropna=False)

    for (title_val, cluster_id), sub in gb:
        title = str(title_val)
        if not title or title == "nan": continue

        # Lấy FPS
        fps = _get_fps_for_title(meta, title)

        # Lấy danh sách frame
        frames_int = sorted(set(iv for v in sub[frame_col] if (iv := _extract_frame_int(v)) is not None))

        # Tính toán Scenes
        raw_scenes = _frames_to_scenes(frames_int, fps=fps, max_gap_s=3.0, pad_s=0.5)
        merged = _merge_overlaps(raw_scenes, tol=0.25)
        final_scenes = _limit_scenes(merged, max_scenes=12, min_len_s=1.2)

        # Lấy Preview Images
        c_id_str = str(cluster_id)
        previews = _list_previews(preview_root, title, c_id_str, limit=12)

        # Cấu trúc Scene payload
        vurl = f"/videos/{title}"
        scene_payload = [
            {"start_time": float(round(s, 3)), "end_time": float(round(e, 3)), "video_url": vurl}
            for (s, e) in final_scenes
        ]

        # Init Character Data
        char_data = {
            "name": "Unknown",  # Default
            "label_status": "UNLABELED",
            "label_confidence": 0.0,
            "rep_image": previews[0] if previews else None,
            "preview_paths": previews,
            "scenes": scene_payload
        }

        # 4. [FIX] Restore Preserved Label if exists
        pres_key = (title, c_id_str)
        if pres_key in preserved_labels:
            saved = preserved_labels[pres_key]
            char_data["name"] = saved["name"]
            char_data["label_status"] = saved["label_status"]
            char_data["label_confidence"] = saved["label_confidence"]
            # Không print để tránh spam log

        manifest.setdefault(title, {})[c_id_str] = char_data

    # 5. Write Atomic
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    import os
    os.replace(tmp_path, out_path)

    num_movies = len(manifest)
    num_chars = sum(len(v) for v in manifest.values())
    print(f"[Characters] Wrote {num_chars} characters across {num_movies} movies -> {out_path}")

    return str(out_path)