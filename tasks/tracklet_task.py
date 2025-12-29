# tasks/tracklet_task.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from prefect import task


# --------------------------
# Geometry & similarity
# --------------------------
def _clean_bbox(b: Tuple[float, float, float, float]) -> Optional[Tuple[int, int, int, int]]:
    """Clamp & type-cast bbox; trả None nếu không hợp lệ."""
    try:
        x1, y1, x2, y2 = map(float, b)
        if not np.isfinite([x1, y1, x2, y2]).all():
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
            return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - ay1 if False else (by2 - by1))  # keep simple
    union = max(1.0, float(area_a + area_b - inter))
    return float(inter) / union


def _area(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, (x2 - x1)) * max(0, (y2 - y1))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """a, b đã/ sẽ được chuẩn hoá L2 bên ngoài; có guard zero."""
    if a is None or b is None:
        return -1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a / na, b / nb))


def _parse_frame_index(val) -> int:
    """
    Chuyển mọi kiểu frame về chỉ số int tăng dần:
      - int/float kiểu 1234 -> 1234
      - "1234" -> 1234
      - "frame_0000123.jpg" hay path tương tự -> 123
      - nếu không tách được số -> -1 (sẽ thay bằng index dòng)
    """
    if val is None:
        return -1
    # numeric trực tiếp
    try:
        return int(val)
    except Exception:
        pass
    # float string "123.0"
    try:
        f = float(str(val))
        if np.isfinite(f):
            return int(f)
    except Exception:
        pass
    # rút chuỗi số dài nhất
    try:
        s = str(val)
        nums = re.findall(r"\d+", s)
        if nums:
            longest = max(nums, key=len)
            return int(longest)
    except Exception:
        pass
    return -1


# --------------------------
# Track state
# --------------------------
@dataclass
class _Track:
    track_id: int
    last_frame: int
    bbox: Tuple[int, int, int, int]
    emb: np.ndarray
    length: int = 1


def _best_match(
    bbox: Tuple[int, int, int, int],
    emb: np.ndarray,
    frame_idx: int,
    active: List[_Track],
    iou_threshold: float,
    cos_threshold: float,
    max_age: int,
    area_ratio_min: float,
    area_ratio_max: float,
) -> Optional[_Track]:
    """Chọn track tốt nhất theo cosine trong các ứng viên đạt IoU, còn sống trong max_age, và hợp lý về diện tích."""
    best: Optional[_Track] = None
    best_cos = -1.0
    area_b = _area(bbox)

    for t in active:
        age = frame_idx - t.last_frame
        if age < 1 or age > max_age:
            continue
        iou = _iou(bbox, t.bbox)
        if iou < iou_threshold:
            continue

        # gating theo diện tích bbox để tránh ghép khi scale thay đổi đột ngột
        area_t = _area(t.bbox)
        if area_t <= 0 or area_b <= 0:
            continue
        ratio = area_b / max(1.0, float(area_t))
        if ratio < area_ratio_min or ratio > area_ratio_max:
            continue

        cos = _cosine_sim(emb, t.emb)
        if cos >= cos_threshold and cos > best_cos:
            best_cos = cos
            best = t

    return best


# --------------------------
# Public task
# --------------------------
@task(name="Tracklet Task")
def tracklet_task(
    detections: pd.DataFrame,
    iou_threshold: float = 0.3,
    cos_threshold: float = 0.6,
    max_age: int = 2,
    area_ratio_min: float = 0.5,
    area_ratio_max: float = 2.0,
) -> pd.DataFrame:
    """
    Nối detection theo thời gian thành tracklet.
    Yêu cầu các cột tối thiểu:
      - frame (int | str | path), bbox (x1,y1,x2,y2), emb (1D array/list)
    Tùy chọn: movie_id để xử lý từng phim độc lập.
    Trả về DataFrame có thêm: track_id, track_len.
    """

    if detections is None or detections.empty:
        return detections

    df = detections.copy()

    # Guard cột cơ bản
    if "frame" not in df.columns or "bbox" not in df.columns or "emb" not in df.columns:
        # Không đủ dữ liệu để track → trả như cũ
        return df

    # Nếu có movie_id → track riêng theo từng phim
    group_cols = ["movie_id"] if "movie_id" in df.columns else [None]
    outputs: List[pd.DataFrame] = []

    for movie_key, sub in (df.groupby(group_cols) if group_cols[0] is not None else [(None, df)]):

        # Sắp xếp theo "thời gian": tạo cột số nguyên _frame_idx
        tmp = sub["frame"].apply(_parse_frame_index)
        fallback_seq = np.arange(len(sub), dtype=int)
        frame_idx_series = pd.Series(np.where(tmp.values >= 0, tmp.values, fallback_seq), index=sub.index)

        sub = sub.assign(_frame_idx=frame_idx_series).sort_values(["_frame_idx"]).reset_index(drop=True)

        # Chuẩn bị embedding & bbox từ cột (không dùng getattr trên itertuples)
        frames_idx: List[int] = sub["_frame_idx"].astype(int).tolist()
        raw_bboxes = sub["bbox"].values.tolist()
        raw_embs = sub["emb"].values.tolist()

        # Làm sạch bbox và L2-normalize embedding
        bboxes: List[Optional[Tuple[int, int, int, int]]] = []
        embs: List[Optional[np.ndarray]] = []
        for bb, e in zip(raw_bboxes, raw_embs):
            bb2 = _clean_bbox(bb) if isinstance(bb, (list, tuple)) and len(bb) == 4 else None
            bboxes.append(bb2)
            if e is None:
                embs.append(None)
            else:
                arr = np.asarray(e, dtype=np.float32).reshape(-1)
                n = np.linalg.norm(arr)
                embs.append(arr / max(1.0, n) if n > 0 else None)

        # Track loop
        next_id = 0
        active: List[_Track] = []
        assigned_ids: List[Optional[int]] = [None] * len(sub)

        for i, (fidx, bb, e) in enumerate(zip(frames_idx, bboxes, embs)):
            # Giữ chỉ các track còn sống
            active = [t for t in active if fidx - t.last_frame <= max_age]

            if bb is None or e is None:
                assigned_ids[i] = None
                continue

            # Tìm best match
            t = _best_match(
                bbox=bb,
                emb=e,
                frame_idx=fidx,
                active=active,
                iou_threshold=iou_threshold,
                cos_threshold=cos_threshold,
                max_age=max_age,
                area_ratio_min=area_ratio_min,
                area_ratio_max=area_ratio_max,
            )

            if t is None:
                # Tạo track mới
                t = _Track(track_id=next_id, last_frame=fidx, bbox=bb, emb=e, length=1)
                active.append(t)
                assigned_ids[i] = next_id
                next_id += 1
            else:
                # Cập nhật track
                t.last_frame = fidx
                t.bbox = bb
                t.emb = e
                t.length += 1
                assigned_ids[i] = t.track_id

        # Ghi kết quả
        sub = sub.assign(track_id=assigned_ids)

        # Tính track_len cho mỗi detection (join từ bảng summary)
        track_len_map: Dict[int, int] = {}
        if "track_id" in sub.columns:
            tl = sub.dropna(subset=["track_id"]).groupby("track_id")["_frame_idx"].size()
            track_len_map = {int(k): int(v) for k, v in tl.to_dict().items()}
            sub["track_len"] = sub["track_id"].map(lambda x: track_len_map.get(int(x), 1) if pd.notna(x) else 0)
        else:
            sub["track_len"] = 0

        # bỏ cột tạm
        sub = sub.drop(columns=["_frame_idx"], errors="ignore")
        outputs.append(sub)

    out = pd.concat(outputs, axis=0, ignore_index=True) if outputs else df
    return out


# --------------------------
# Backward-compat alias
# --------------------------
def link_tracklets(
    detections: pd.DataFrame,
    iou_threshold: float = 0.3,
    cos_threshold: float = 0.6,
    max_age: int = 2,
    area_ratio_min: float = 0.5,
    area_ratio_max: float = 2.0,
) -> pd.DataFrame:
    """
    Giữ tương thích với code cũ: embedding_task.py import link_tracklets.
    Thực chất gọi vào hàm tracklet_task.fn(...) (Prefect task).
    """
    return tracklet_task.fn(
        detections=detections,
        iou_threshold=iou_threshold,
        cos_threshold=cos_threshold,
        max_age=max_age,
        area_ratio_min=area_ratio_min,
        area_ratio_max=area_ratio_max,
    )
