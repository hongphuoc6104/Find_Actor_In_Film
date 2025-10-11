# tools/merge_scene_intervals.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---- Cấu hình/metadata ----
def _read_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        import json
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


# ---- DF helpers ----
def _first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _coerce_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v == float("inf") or v != v:
            return None
        return v
    except Exception:
        return None


def _frame_to_time(fr: Any, fps: Optional[float]) -> Optional[float]:
    if fps is None or fps <= 0:
        return None
    try:
        return float(int(fr)) / float(fps)
    except Exception:
        # fr có dạng "frame_0000558.jpg" -> lấy số
        import re
        m = re.search(r"(\d+)", str(fr))
        if not m:
            return None
        try:
            return float(int(m.group(1))) / float(fps)
        except Exception:
            return None


def _row_to_interval_seconds(row: pd.Series, fps: Optional[float], col_map: Dict[str, str]) -> Optional[Tuple[float, float, Optional[float]]]:
    """
    Trả (start_time, end_time, score)
    Ưu tiên cột time, rơi về frame nếu có fps.
    """
    score = None
    if col_map.get("score") and pd.notna(row.get(col_map["score"])):
        score = _coerce_float(row[col_map["score"]])

    st: Optional[float] = None
    et: Optional[float] = None

    # time trước
    if col_map.get("start_time") and pd.notna(row.get(col_map["start_time"])):
        st = _coerce_float(row[col_map["start_time"]])
    if col_map.get("end_time") and pd.notna(row.get(col_map["end_time"])):
        et = _coerce_float(row[col_map["end_time"]])

    # nếu thiếu time -> thử frame
    if st is None and col_map.get("start_frame") and pd.notna(row.get(col_map["start_frame"])):
        st = _frame_to_time(row[col_map["start_frame"]], fps)
    if et is None:
        # end_frame >; nếu không có, fallback về start_frame + 1/fps
        if col_map.get("end_frame") and pd.notna(row.get(col_map["end_frame"])):
            et = _frame_to_time(row[col_map["end_frame"]], fps)
        elif col_map.get("start_frame") and pd.notna(row.get(col_map["start_frame"])):
            # giả định 1 frame
            et = _frame_to_time(row[col_map["start_frame"]], fps)
            if et is not None and fps and fps > 0:
                et += 1.0 / float(fps)

    if st is None or et is None:
        return None
    if et < st:
        st, et = et, st

    return (st, et, score)


# ---- Merge intervals ----
def _merge_intervals(
    items: List[Tuple[float, float, Optional[float]]],
    min_gap: float,
    extend: float,
    min_len: float,
) -> List[Tuple[float, float, Optional[float]]]:
    """
    items: list of (start, end, score)
    - gộp nếu khoảng cách <= min_gap
    - nới mỗi đầu extend
    - bỏ đoạn < min_len
    """
    if not items:
        return []
    # sort by start
    items = sorted(items, key=lambda x: x[0])

    merged: List[Tuple[float, float, List[float]]] = []  # (s,e,[scores])
    cur_s, cur_e, cur_scores = items[0][0], items[0][1], []
    if items[0][2] is not None:
        cur_scores = [float(items[0][2])]
    else:
        cur_scores = []

    for s, e, sc in items[1:]:
        if s <= cur_e + min_gap:
            # merge
            if e > cur_e:
                cur_e = e
            if sc is not None:
                cur_scores.append(float(sc))
        else:
            merged.append((cur_s, cur_e, cur_scores.copy()))
            cur_s, cur_e = s, e
            cur_scores = [float(sc)] if sc is not None else []

    merged.append((cur_s, cur_e, cur_scores.copy()))

    # extend + filter
    out: List[Tuple[float, float, Optional[float]]] = []
    for s, e, scores in merged:
        s2 = max(0.0, s - extend)
        e2 = e + extend
        if (e2 - s2) < min_len:
            continue
        # score đại diện: max
        rep = max(scores) if scores else None
        out.append((s2, e2, rep))

    return out


def _save_json(dst: Path, data: Any):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Gộp & lọc đoạn xuất hiện theo nhân vật/film.")
    ap.add_argument("--input", type=str, required=True,
                    help="Đường dẫn parquet/csv chứa các phát hiện (tracks/scenes/…).")
    ap.add_argument("--out", type=str, default="warehouse/merged_scenes",
                    help="Thư mục output (mặc định warehouse/merged_scenes).")
    ap.add_argument("--movie", type=str, default="",
                    help="Giới hạn 1 phim (title hoặc id).")
    ap.add_argument("--min-gap-seconds", type=float, default=2.0)
    ap.add_argument("--extend-seconds", type=float, default=0.5)
    ap.add_argument("--min-len-seconds", type=float, default=1.0)
    ap.add_argument("--topk-per-char", type=int, default=12,
                    help="Giữ tối đa N đoạn/nhân vật theo điểm đại diện.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"❌ Input not found: {in_path}")
        return

    # load df
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    if df is None or df.empty:
        print("⚠️  Không có dữ liệu.")
        return

    # cột linh hoạt
    col_movie = _first_col(df, ["movie", "movie_title", "movie_id"])
    col_char = _first_col(df, ["character_id", "cluster_id"])
    col_score = _first_col(df, ["score", "similarity", "sim", "distance"])
    col_st = _first_col(df, ["start_time", "ts_start"])
    col_et = _first_col(df, ["end_time", "ts_end"])
    col_sf = _first_col(df, ["start_frame", "rep_frame", "frame", "frame_idx"])
    col_ef = _first_col(df, ["end_frame"])

    if not col_movie or not col_char:
        print("❌ Thiếu cột movie/character.")
        return

    col_map = {
        "start_time": col_st or "",
        "end_time": col_et or "",
        "start_frame": col_sf or "",
        "end_frame": col_ef or "",
        "score": col_score or "",
    }

    # metadata (để lấy fps)
    meta = _read_metadata(Path("Data/metadata.json"))

    # lọc theo movie nếu chỉ định
    if args.movie:
        sel_title = _title_from_any(meta, args.movie) or str(args.movie)
        if col_movie == "movie_id":
            # df theo id
            try:
                # map title -> id từ meta
                rev = {v: k for k, v in _reverse_movie_id_map(meta).items()}
                # rev: title->id string
                target = rev.get(sel_title, sel_title)
            except Exception:
                target = sel_title
            df = df[df[col_movie].astype(str) == str(target)].copy()
        else:
            # df theo title
            df = df[df[col_movie].astype(str) == str(sel_title)].copy()

    if df.empty:
        print("⚠️  Không có dòng nào sau khi lọc movie.")
        return

    # group theo movie-char
    groups = df.groupby([col_movie, col_char], dropna=False)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    total_written = 0

    for (mv_key, ch_key), g in groups:
        title = _title_from_any(meta, mv_key) or str(mv_key)
        char_id = str(ch_key)

        # fps
        fps = None
        if title in meta:
            fps = meta[title].get("fps")

        # build intervals list
        intervals: List[Tuple[float, float, Optional[float]]] = []
        for _, row in g.iterrows():
            item = _row_to_interval_seconds(row, fps, col_map)
            if item is not None:
                intervals.append(item)

        if not intervals:
            continue

        merged = _merge_intervals(
            items=intervals,
            min_gap=max(0.0, float(args.min_gap_seconds)),
            extend=max(0.0, float(args.extend_seconds)),
            min_len=max(0.0, float(args.min_len_seconds)),
        )

        if not merged:
            continue

        # sort theo điểm đại diện giảm dần để pick topk
        merged.sort(key=lambda t: (t[2] if t[2] is not None else -1.0), reverse=True)
        if args.topk_per_char and args.topk_per_char > 0:
            merged = merged[: int(args.topk_per_char)]

        # sort lại theo thời gian bắt đầu để UX tốt hơn
        merged.sort(key=lambda t: t[0])

        payload = {
            "movie": title,
            "character_id": char_id,
            "intervals": [
                {"start_time": float(s), "end_time": float(e), "score": (None if sc is None else float(sc))}
                for (s, e, sc) in merged
            ],
            "params": {
                "min_gap_seconds": float(args.min_gap_seconds),
                "extend_seconds": float(args.extend_seconds),
                "min_len_seconds": float(args.min_len_seconds),
                "topk_per_char": int(args.topk_per_char),
            },
            "source": str(in_path),
        }

        out_path = out_root / title / f"{char_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        total_chars += 1
        total_written += len(merged)

    print("✅ Done.")
    print(json.dumps({"characters": total_chars, "intervals_written": total_written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
