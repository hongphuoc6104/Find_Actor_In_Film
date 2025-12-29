# services/scene_loader.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pandas là tuỳ chọn — chỉ cần khi fallback parquet
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

# ---------------------------
# Metadata helpers
# ---------------------------
def _read_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _reverse_movie_id_map(meta: Dict[str, Any]) -> Dict[str, str]:
    gen = meta.get("_generated") or {}
    id_map = gen.get("movie_id_map") or {}
    rev: Dict[str, str] = {}
    if isinstance(id_map, dict):
        for title, mid in id_map.items():
            try:
                rev[str(int(mid))] = str(title)
            except Exception:
                pass
    return rev


def _title_from_any(meta: Dict[str, Any], key: Any) -> Optional[str]:
    s = str(key).strip()
    if not s:
        return None
    if s in meta and s != "_generated":
        return s
    rev = _reverse_movie_id_map(meta)
    return rev.get(s)


# ---------------------------
# Safe utils
# ---------------------------
def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v or v == float("inf"):
            return None
        return v
    except Exception:
        return None


def _frame_to_time(fr: Any, fps: Optional[float]) -> Optional[float]:
    if not fps or fps <= 0:
        return None
    # hỗ trợ cả số thuần và chuỗi kiểu "frame_0000558.jpg"
    try:
        return float(int(fr)) / float(fps)
    except Exception:
        m = re.search(r"(\d+)", str(fr))
        if not m:
            return None
        try:
            return float(int(m.group(1))) / float(fps)
        except Exception:
            return None


def _merge_simple(
    items: List[Tuple[float, float, Optional[float]]],
    *,
    min_gap: float = 2.0,
    extend: float = 0.5,
    min_len: float = 1.0,
) -> List[Tuple[float, float, Optional[float]]]:
    if not items:
        return []
    items = sorted(items, key=lambda t: t[0])
    merged: List[Tuple[float, float, List[float]]] = []

    s0, e0, scores0 = items[0][0], items[0][1], []
    if items[0][2] is not None:
        scores0.append(float(items[0][2]))

    for s, e, sc in items[1:]:
        if s <= e0 + min_gap:
            if e > e0:
                e0 = e
            if sc is not None:
                scores0.append(float(sc))
        else:
            merged.append((s0, e0, scores0.copy()))
            s0, e0 = s, e
            scores0 = [float(sc)] if sc is not None else []

    merged.append((s0, e0, scores0.copy()))

    out: List[Tuple[float, float, Optional[float]]] = []
    for s, e, scores in merged:
        s2 = max(0.0, s - extend)
        e2 = e + extend
        if (e2 - s2) < min_len:
            continue
        rep = max(scores) if scores else None
        out.append((s2, e2, rep))
    return out


# ---------------------------
# Public API
# ---------------------------
def get_scenes_for_character(
    cfg: Dict[str, Any],
    movie_title: str,
    char_id: str,
    *,
    json_root: str = "warehouse/merged_scenes",
    parquet_candidates: Optional[List[str]] = None,
    min_gap_seconds: float = 2.0,
    extend_seconds: float = 0.5,
    min_len_seconds: float = 1.0,
    topk: int = 12,
) -> List[Dict[str, Any]]:
    """
    Trả về danh sách scene đã “sẵn sàng cho FE”:
      [{"start_time":..., "end_time":..., "score":..., "video_url":"/videos/<movie_title>"}]
    Ưu tiên đọc JSON đã merge; nếu không có sẽ đọc parquet (fallback).
    """
    movie_title = str(movie_title).strip()
    char_id = str(char_id).strip()

    # 1) Ưu tiên JSON đã merge
    json_path = Path(json_root) / movie_title / f"{char_id}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            intervals = data.get("intervals") or []
            scenes = []
            for it in intervals:
                st = _as_float((it or {}).get("start_time"))
                et = _as_float((it or {}).get("end_time"))
                sc = _as_float((it or {}).get("score"))
                if st is None or et is None:
                    continue
                scenes.append({
                    "start_time": float(st),
                    "end_time": float(et),
                    "score": (None if sc is None else float(sc)),
                    "video_url": f"/videos/{movie_title}",
                })
            # sort theo start_time
            scenes.sort(key=lambda x: x["start_time"])
            if topk and topk > 0 and len(scenes) > topk:
                # ưu tiên theo score giảm dần (nếu có), rồi giữ lại sớm nhất
                scenes.sort(key=lambda x: (x["score"] if x["score"] is not None else -1.0), reverse=True)
                scenes = scenes[:int(topk)]
                scenes.sort(key=lambda x: x["start_time"])
            return scenes
        except Exception:
            pass  # rơi xuống parquet

    # 2) Fallback parquet
    if parquet_candidates is None:
        parquet_candidates = [
            "warehouse/parquet/scenes.parquet",
            "warehouse/parquet/clusters.parquet",
            "warehouse/parquet/tracks.parquet",
            "warehouse/parquet/embeddings.parquet",
        ]

    # metadata để lấy fps nếu chỉ có frame
    meta_path = Path((cfg.get("storage") or {}).get("metadata_json") or "Data/metadata.json")
    meta = _read_metadata(meta_path)
    fps = None
    if movie_title in meta:
        fps = meta[movie_title].get("fps")

    items: List[Tuple[float, float, Optional[float]]] = []

    if pd is None:
        # Không có pandas thì chịu, trả rỗng
        return []

    for cand in parquet_candidates:
        p = Path(cand)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        # tìm cột linh hoạt
        col_movie = None
        for c in ["movie", "movie_title", "movie_id"]:
            if c in df.columns:
                col_movie = c
                break
        col_char = None
        for c in ["character_id", "cluster_id"]:
            if c in df.columns:
                col_char = c
                break
        if not col_movie or not col_char:
            continue

        # lọc theo movie title (chấp nhận df theo id)
        if col_movie == "movie_id":
            # map title->id string
            rev = {v: k for k, v in _reverse_movie_id_map(meta).items()}
            target = rev.get(movie_title, movie_title)
            sub = df[df[col_movie].astype(str) == str(target)].copy()
        else:
            sub = df[df[col_movie].astype(str) == str(movie_title)].copy()

        if sub.empty:
            continue

        sub = sub[sub[col_char].astype(str) == str(char_id)].copy()
        if sub.empty:
            continue

        # map cột thời gian/frame/điểm
        def pick(*names: str) -> Optional[str]:
            for n in names:
                if n in sub.columns:
                    return n
            return None

        c_st = pick("start_time", "ts_start")
        c_et = pick("end_time", "ts_end")
        c_sf = pick("start_frame", "rep_frame", "frame", "frame_idx")
        c_ef = pick("end_frame")
        c_sc = pick("score", "similarity", "sim", "distance")

        for _, r in sub.iterrows():
            score = _as_float(r[c_sc]) if c_sc else None

            st, et = None, None
            if c_st and r.get(c_st) is not None:
                st = _as_float(r[c_st])
            if c_et and r.get(c_et) is not None:
                et = _as_float(r[c_et])

            if st is None and c_sf and r.get(c_sf) is not None:
                st = _frame_to_time(r[c_sf], fps)
            if et is None:
                if c_ef and r.get(c_ef) is not None:
                    et = _frame_to_time(r[c_ef], fps)
                elif c_sf and r.get(c_sf) is not None:
                    et = _frame_to_time(r[c_sf], fps)
                    if et is not None and fps and fps > 0:
                        et += 1.0 / float(fps)

            if st is None or et is None:
                continue
            if et < st:
                st, et = et, st

            items.append((st, et, score))

        # nếu đã gom được từ 1 nguồn là đủ
        if items:
            break

    if not items:
        return []

    merged = _merge_simple(
        items,
        min_gap=min_gap_seconds,
        extend=extend_seconds,
        min_len=min_len_seconds,
    )

    if not merged:
        return []

    # Ưu tiên score cao rồi cắt topk, sau đó sắp theo thời gian
    merged.sort(key=lambda t: (t[2] if t[2] is not None else -1.0), reverse=True)
    if topk and topk > 0 and len(merged) > topk:
        merged = merged[: int(topk)]
    merged.sort(key=lambda t: t[0])

    scenes = [
        {
            "start_time": float(s),
            "end_time": float(e),
            "score": (None if sc is None else float(sc)),
            "video_url": f"/videos/{movie_title}",
        }
        for (s, e, sc) in merged
    ]
    return scenes
