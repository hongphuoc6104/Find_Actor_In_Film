from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional


# ==============================
# Config helpers
# ==============================

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def get_highlight_cfg(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Đọc block cấu hình 'highlight' từ config.yaml (nếu có) và đưa về giá trị an toàn.
    Các khóa đang dùng:

      - MIN_HL_DURATION_SEC:   độ dài tối thiểu cho một cảnh sau khi gộp
      - MERGE_GAP_SEC:         nếu khoảng trống giữa 2 cảnh < giá trị này thì gộp lại
      - TOP_K_HL_PER_SCENE:    tối đa số cảnh trả về cho mỗi nhân vật
      - PAD_IF_FEW_SEC:        (mới) nới biên mỗi đầu nếu cảnh quá ít
      - TARGET_MIN_SCENES:     (mới) ngưỡng “cảnh ít” để kích hoạt PAD_IF_FEW_SEC
    """
    hl = dict(cfg.get("highlight", {})) if isinstance(cfg, dict) else {}

    # Giá trị chuẩn xác & mặc định hợp lý
    return {
        "MIN_HL_DURATION_SEC": _as_float(hl.get("MIN_HL_DURATION_SEC", 4.0), 4.0),
        "MERGE_GAP_SEC": _as_float(hl.get("MERGE_GAP_SEC", 6.0), 6.0),
        "TOP_K_HL_PER_SCENE": _as_int(hl.get("TOP_K_HL_PER_SCENE", 3), 3),
        # Thêm hai tham số mới cho việc nới biên khi quá ít cảnh
        "PAD_IF_FEW_SEC": _as_float(hl.get("PAD_IF_FEW_SEC", 1.5), 1.5),
        "TARGET_MIN_SCENES": _as_int(hl.get("TARGET_MIN_SCENES", 2), 2),
    }


# ==============================
# Core structures
# ==============================

@dataclass
class Scene:
    """
    Đại diện một đoạn xuất hiện (đơn vị: giây).
    Các trường phụ (score/video_url/…​) giữ nguyên nếu có trong input.
    """
    start_time: float
    end_time: float
    score: float = 1.0
    video_url: Optional[str] = None
    # các trường tự do khác
    extra: Dict[str, Any] | None = None

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


# ==============================
# Utilities
# ==============================

def _norm_scene_dict(d: Dict[str, Any]) -> Scene:
    """Chuyển mọi dict cảnh về `Scene` với mặc định an toàn."""
    st = _as_float(d.get("start_time"), 0.0)
    et = _as_float(d.get("end_time"), st)
    if et < st:
        st, et = et, st
    return Scene(
        start_time=st,
        end_time=et,
        score=_as_float(d.get("score"), 1.0),
        video_url=d.get("video_url"),
        extra={k: v for k, v in d.items() if k not in {"start_time", "end_time", "score", "video_url"}},
    )


def _effective_score(s: Scene) -> float:
    """
    Điểm xếp hạng cảnh: kết hợp score (nếu có) và độ dài (log1p).
    Mục tiêu: cảnh dài/ổn định sẽ ưu tiên hơn cảnh ngắn dù score cao.
    """
    dur = max(0.0, s.duration())
    base = float(s.score) if math.isfinite(s.score) else 1.0
    return base * math.log1p(dur + 1e-6)


def _merge_sorted(scenes: List[Scene], merge_gap: float) -> List[Scene]:
    """Gộp các cảnh đã **được sort theo start_time** nếu gap giữa chúng < merge_gap."""
    if not scenes:
        return []
    merged: List[Scene] = [scenes[0]]
    for sc in scenes[1:]:
        last = merged[-1]
        # Nếu chồng lấp hoặc gap nhỏ hơn ngưỡng, gộp lại
        if sc.start_time <= last.end_time + merge_gap:
            merged[-1] = Scene(
                start_time=last.start_time,
                end_time=max(last.end_time, sc.end_time),
                score=max(last.score, sc.score),  # giữ score cao nhất của 2 đoạn
                video_url=last.video_url or sc.video_url,
                extra=(last.extra or {}) | (sc.extra or {}),
            )
        else:
            merged.append(sc)
    return merged


def _pad_scenes_if_few(scenes: List[Scene], pad_each_side: float) -> List[Scene]:
    """
    Nới mỗi đầu của từng cảnh một lượng 'pad_each_side' giây để đỡ hụt khúc mở/đóng.
    Dùng khi tổng số cảnh quá ít (ví dụ 1 cảnh).
    """
    padded: List[Scene] = []
    for sc in scenes:
        padded.append(
            Scene(
                start_time=max(0.0, sc.start_time - pad_each_side),
                end_time=max(sc.end_time, sc.start_time + 0.01) + pad_each_side,
                score=sc.score,
                video_url=sc.video_url,
                extra=sc.extra,
            )
        )
    return padded


# ==============================
# Public API
# ==============================

def consolidate_scenes(
    raw_scenes: Iterable[Dict[str, Any] | Scene],
    *,
    min_duration: float = 4.0,
    merge_gap: float = 6.0,
    top_k: int = 3,
    pad_if_few: float = 1.5,
    target_min_scenes: int = 2,
    clamp_to: Optional[tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Nhận một danh sách cảnh rời rạc (thường là từ tracklets), trả về danh sách
    cảnh đã gộp & xếp hạng để hiển thị cho FE.

    Quy tắc:
      1) sort theo start_time
      2) gộp nếu gap < merge_gap
      3) loại các cảnh < min_duration
      4) nếu tổng số cảnh < target_min_scenes → nới biên mỗi cảnh thêm pad_if_few
      5) xếp hạng theo _effective_score() và cắt top_k
      6) clamp thời gian vào [lo, hi] nếu clamp_to được cung cấp

    Trả về: List[dict] có các khóa {start_time, end_time, score, video_url, ...}
    """
    # Chuẩn hóa input
    scenes: List[Scene] = [
        s if isinstance(s, Scene) else _norm_scene_dict(dict(s))
        for s in (raw_scenes or [])
    ]
    if not scenes:
        return []

    # Sort để gộp ổn định
    scenes.sort(key=lambda s: (s.start_time, s.end_time))

    # Gộp các cảnh gần nhau / chồng lấp
    scenes = _merge_sorted(scenes, merge_gap=merge_gap)

    # Loại cảnh quá ngắn
    scenes = [s for s in scenes if s.duration() >= min_duration]
    if not scenes:
        return []

    # Nếu quá ít cảnh → nới biên mỗi cảnh một chút
    if len(scenes) < max(1, int(target_min_scenes)):
        scenes = _pad_scenes_if_few(scenes, pad_each_side=max(0.0, pad_if_few))
        # gộp lại lần nữa vì có thể biên đè nhau
        scenes.sort(key=lambda s: (s.start_time, s.end_time))
        scenes = _merge_sorted(scenes, merge_gap=merge_gap)

    # Clamp theo [0, duration] nếu biết
    if clamp_to is not None:
        lo, hi = clamp_to
        lo = max(0.0, float(lo))
        hi = max(lo, float(hi))
        tmp: List[Scene] = []
        for s in scenes:
            st = min(max(s.start_time, lo), hi)
            et = min(max(s.end_time, lo), hi)
            if et - st > 1e-3:
                tmp.append(Scene(st, et, s.score, s.video_url, s.extra))
        scenes = tmp

    # Xếp hạng và lấy top_k
    scenes.sort(key=_effective_score, reverse=True)
    if top_k and top_k > 0 and len(scenes) > top_k:
        scenes = scenes[:top_k]

    # Xuất dict (giữ nguyên các field phụ)
    out: List[Dict[str, Any]] = []
    for s in scenes:
        d = asdict(s)
        # Flatten extra (nếu có)
        if d.get("extra"):
            for k, v in list(d["extra"].items()):
                if k not in d:
                    d[k] = v
        d.pop("extra", None)
        out.append(d)
    return out


def consolidate_from_cfg(
    raw_scenes: Iterable[Dict[str, Any] | Scene],
    cfg: Dict[str, Any] | None,
    *,
    clamp_to: Optional[tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Bọc tiện lợi: đọc thông số từ config.yaml (block `highlight`) rồi gọi consolidate_scenes.
    """
    hcfg = get_highlight_cfg(cfg)
    return consolidate_scenes(
        raw_scenes,
        min_duration=hcfg["MIN_HL_DURATION_SEC"],
        merge_gap=hcfg["MERGE_GAP_SEC"],
        top_k=hcfg["TOP_K_HL_PER_SCENE"],
        pad_if_few=hcfg["PAD_IF_FEW_SEC"],
        target_min_scenes=hcfg["TARGET_MIN_SCENES"],
        clamp_to=clamp_to,
    )
