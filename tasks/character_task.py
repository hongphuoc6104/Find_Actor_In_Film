#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from prefect import task

from utils.config_loader import load_config


# =========================
# Helpers
# =========================

def _read_metadata(cfg: dict) -> Dict[str, Any]:
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    meta_path = storage.get("metadata_json") or "Data/metadata.json"
    p = Path(meta_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _reverse_movie_id_map(meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Trả map id(str) -> title, dựa vào metadata['_generated']['movie_id_map'].
    """
    gen = meta.get("_generated") or {}
    id_map = gen.get("movie_id_map") or {}
    rev: Dict[str, str] = {}
    for title, mid in id_map.items():
        try:
            rev[str(int(mid))] = str(title)
        except Exception:
            pass
    return rev


def _get_fps_for_title(meta: Dict[str, Any], title: str) -> float:
    info = meta.get(title) or {}
    fps = info.get("fps")
    try:
        return float(fps) if fps is not None else 25.0
    except Exception:
        return 25.0


def _resolve_title_from_any(meta: Dict[str, Any], movie_or_id: Any) -> Optional[str]:
    """
    Nhận vào movie_id (số hoặc chuỗi) hoặc title → trả title.
    """
    key = str(movie_or_id).strip()
    if not key:
        return None
    if key in meta and key != "_generated":
        return key
    rev = _reverse_movie_id_map(meta)
    return rev.get(key)


def _coerce_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _frames_to_scenes(
    frames: List[int],
    fps: float,
    *,
    max_gap_s: float = 3.0,
    pad_s: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Gom danh sách frame (đã sort) thành các khoảng thời gian (start_s, end_s).
    - max_gap_s: nếu khoảng cách giữa 2 frame > max_gap_s thì cắt đoạn.
    - pad_s: nới nhẹ 2 đầu mỗi đoạn.
    """
    if not frames:
        return []
    frames = sorted(frames)
    # chuyển frame → giây
    times = [f / max(fps, 1e-6) for f in frames]
    out: List[Tuple[float, float]] = []
    start = times[0]
    prev = times[0]
    for t in times[1:]:
        if (t - prev) > max_gap_s:
            # kết thúc đoạn cũ
            s = max(0.0, start - pad_s)
            e = max(s, prev + pad_s)
            out.append((s, e))
            start = t
        prev = t
    # đoạn cuối
    s = max(0.0, start - pad_s)
    e = max(s, prev + pad_s)
    out.append((s, e))
    return out


def _merge_overlaps(intervals: List[Tuple[float, float]], *, tol: float = 0.25) -> List[Tuple[float, float]]:
    """
    Gộp các đoạn chồng lấn hoặc sát nhau trong ngưỡng tol (giây).
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + tol:  # chồng lấn/tiệm cận
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _limit_scenes(
    scenes: List[Tuple[float, float]],
    *,
    max_scenes: int = 10,
    min_len_s: float = 1.2,
) -> List[Tuple[float, float]]:
    """
    Giữ tối đa N đoạn, bỏ đoạn quá ngắn.
    Ưu tiên các đoạn dài hơn.
    """
    long_enough = [(s, e) for (s, e) in scenes if (e - s) >= min_len_s]
    long_enough.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    return long_enough[:max_scenes] if max_scenes and max_scenes > 0 else long_enough


def _list_previews(preview_root: Path, title: str, char_id: str, *, limit: int = 12) -> List[str]:
    """
    Thu thập đường dẫn preview (tương đối) cho FE (nhằm debug/QA).
    warehouse/cluster_previews/<title>/<char_id>/*.jpg
    """
    root = preview_root / title / char_id
    paths: List[str] = []
    if root.exists():
        for p in sorted(root.glob("*.jpg")):
            rel = p.as_posix()
            paths.append(rel)
            if len(paths) >= limit:
                break
    return paths


def _load_clusters_parquet() -> Optional[pd.DataFrame]:
    """
    Load thứ tự ưu tiên:
      - warehouse/parquet/clusters_merged.parquet
      - warehouse/parquet/clusters.parquet
    """
    candidates = [
        Path("warehouse/parquet/clusters_merged.parquet"),
        Path("warehouse/parquet/clusters.parquet"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
    return None


def _pick_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    Chuẩn hóa tên cột đầu vào (tùy data hiện có).
    Trả dict mapping:
      - "movie_id"
      - "cluster" (id cụm)
      - "frame"  (số frame hoặc tên file)
      - "score"  (điểm tin cậy, optional)
    """
    cols = {c.lower(): c for c in df.columns}
    mapping: Dict[str, str] = {}

    # movie id
    for key in ["movie_id", "movieid", "mid"]:
        if key in cols:
            mapping["movie_id"] = cols[key]
            break

    # cluster id
    for key in ["cluster", "cluster_id", "char_id", "final_character_id"]:
        if key in cols:
            mapping["cluster"] = cols[key]
            break

    # frame index / filename
    for key in ["frame", "int_frame", "frame_idx", "frame_index", "image", "image_name"]:
        if key in cols:
            mapping["frame"] = cols[key]
            break

    # score (optional)
    for key in ["score", "sim", "similarity", "conf", "confidence"]:
        if key in cols:
            mapping["score"] = cols[key]
            break

    return mapping


def _extract_frame_int(v: Any) -> Optional[int]:
    """
    Cố gắng lấy số frame từ nhiều kiểu:
      - int/float -> int
      - '558' -> 558
      - 'frame_0000558.jpg' -> 558
    """
    if v is None:
        return None
    iv = _coerce_int(v)
    if iv is not None:
        return iv
    s = str(v)
    # tìm chuỗi số dài nhất
    buff: List[str] = []
    cur = []
    for ch in s:
        if ch.isdigit():
            cur.append(ch)
        else:
            if cur:
                buff.append("".join(cur))
                cur = []
    if cur:
        buff.append("".join(cur))
    if not buff:
        return None
    try:
        return int(buff[-1])
    except Exception:
        return None


# =========================
# Main Prefect task
# =========================

@task(name="Character Manifest Task")
def character_task() -> str:
    """
    Tạo file manifest nhân vật cho FE:
      warehouse/characters.json

    Cấu trúc (rút gọn):
    {
      "<TITLE>": {
        "<CHAR_ID>": {
          "rep_image": "... (optional)",
          "preview_paths": [...],
          "scenes": [
            {"start_time": 12.3, "end_time": 18.9, "score": 0.87, "video_url": "/videos/<TITLE>"}
          ]
        },
        ...
      },
      ...
    }
    """
    cfg = load_config()
    meta = _read_metadata(cfg)
    rev_id_map = _reverse_movie_id_map(meta)  # "0" -> "EMCHUA18"
    preview_root = Path("warehouse/cluster_previews")

    # 1) Load clusters parquet (merged > raw)
    df = _load_clusters_parquet()
    if df is None or df.empty:
        # fallback: không có clusters → vẫn xuất file rỗng hợp lệ
        out_path = Path("warehouse/characters.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Characters] No clusters found. Wrote empty manifest → {out_path}")
        return str(out_path)

    # 2) Chuẩn tên cột
    colmap = _pick_cols(df)
    miss = [k for k in ["movie_id", "cluster", "frame"] if k not in colmap]
    if miss:
        raise RuntimeError(f"[Characters] Missing required columns in clusters parquet: {miss}")

    movie_col = colmap["movie_id"]
    cluster_col = colmap["cluster"]
    frame_col = colmap["frame"]
    score_col = colmap.get("score")  # optional

    # 3) Gom theo movie_id + cluster
    manifest: Dict[str, Dict[str, Any]] = {}

    gb = df.groupby([movie_col, cluster_col], dropna=False)
    for (movie_id, cluster_id), sub in gb:
        # → title
        title = _resolve_title_from_any(meta, movie_id) or str(movie_id)

        # lấy fps theo từng phim
        fps = _get_fps_for_title(meta, title)

        # frames
        frames_int = []
        for v in sub[frame_col].tolist():
            iv = _extract_frame_int(v)
            if iv is not None:
                frames_int.append(iv)
        frames_int = sorted(set(frames_int))  # unique + sort
        if not frames_int:
            continue

        # scenes từ frames
        raw_scenes = _frames_to_scenes(frames_int, fps=fps, max_gap_s=3.0, pad_s=0.5)
        merged = _merge_overlaps(raw_scenes, tol=0.25)
        final_scenes = _limit_scenes(merged, max_scenes=12, min_len_s=1.2)

        # score (nếu có)
        avg_score: Optional[float] = None
        if score_col and score_col in sub.columns:
            try:
                avg_score = float(pd.to_numeric(sub[score_col], errors="coerce").dropna().mean())
            except Exception:
                avg_score = None

        # chuẩn bị entry
        movie_bucket = manifest.setdefault(title, {})
        char_id = str(cluster_id)

        # previews
        previews = _list_previews(preview_root, title, char_id, limit=12)

        # video url cho FE
        vurl = f"/videos/{title}"

        scene_payload = [
            {
                "start_time": float(round(s, 3)),
                "end_time": float(round(e, 3)),
                **({"score": float(round(avg_score, 6))} if avg_score is not None else {}),
                "video_url": vurl,
            }
            for (s, e) in final_scenes
        ]

        movie_bucket[char_id] = {
            "rep_image": previews[0] if previews else None,
            "preview_paths": previews,
            "scenes": scene_payload,
        }

    # 4) Ghi file
    out_path = Path("warehouse/characters.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) Log tóm tắt
    num_movies = len(manifest)
    num_chars = sum(len(v) for v in manifest.values())
    print(f"[Characters] Wrote {num_chars} characters across {num_movies} movies → {out_path}")

    return str(out_path)
