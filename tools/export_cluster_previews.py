# tools/export_cluster_previews.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ---- Optional: dùng utils có sẵn nếu project của bạn có ----
try:
    from utils.config_loader import load_config  # type: ignore
except Exception:
    load_config = None  # fallback sẽ đọc config.yaml trực tiếp


# ---------------------------
# Config & metadata helpers
# ---------------------------
def _read_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_cfg() -> Dict[str, Any]:
    """
    Ưu tiên utils.config_loader.load_config(); nếu không có, đọc config.yaml tại cwd.
    """
    if load_config is not None:
        try:
            cfg = load_config()
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
    # Fallback
    yml = Path("config.yaml")
    return _read_yaml_config(yml) if yml.exists() else {}


def _read_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    meta_path = storage.get("metadata_json") or "Data/metadata.json"
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _reverse_movie_id_map(meta: Dict[str, Any]) -> Dict[str, str]:
    """
    metadata._generated.movie_id_map: {title: id}  -> đảo thành {str(id): title}
    """
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
# DataFrame helpers
# ---------------------------
def _load_clusters_parquet(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    cands = [
        storage.get("clusters_merged_parquet") or "warehouse/parquet/clusters_merged.parquet",
        storage.get("clusters_parquet") or "warehouse/parquet/clusters.parquet",
    ]
    for p in cands:
        pth = Path(p)
        if pth.exists():
            try:
                return pd.read_parquet(pth)
            except Exception:
                continue
    return None


def _first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _extract_frame_idx(val: Any) -> Optional[int]:
    """
    'frame_0000558.jpg' -> 558 ; '558' -> 558 ; None -> None
    """
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        pass
    import re
    m = re.search(r"(\d+)", str(val))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# ---------------------------
# Image overlay
# ---------------------------
def _load_font(size: int = 16) -> Optional[ImageFont.FreeTypeFont]:
    # cố gắng lấy font monospace cơ bản có sẵn
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_label(im: Image.Image, lines: List[str]) -> Image.Image:
    """
    Vẽ nhãn semi-transparent ở dưới ảnh, chứa nhiều dòng text.
    """
    draw = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    font = _load_font(16)
    pad = 8
    line_h = 20
    box_h = pad * 2 + line_h * len(lines)

    # semi-transparent black box
    draw.rectangle([0, H - box_h, W, H], fill=(0, 0, 0, 150))

    y = H - box_h + pad
    for ln in lines:
        draw.text((pad, y), ln, fill=(255, 255, 255, 230), font=font)
        y += line_h
    return im


def _safe_open_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    try:
        im = Image.open(path).convert("RGB")
        return im
    except Exception:
        return None


# ---------------------------
# Main exporting
# ---------------------------
def export_previews(
    df: pd.DataFrame,
    out_root: Path,
    meta: Dict[str, Any],
    only_movie: Optional[str] = None,
    limit_per_char: int = 12,
) -> Dict[str, Any]:
    """
    Tạo preview theo cấu trúc:
      warehouse/cluster_previews/<movie_title>/<char_id>/{0001..}.jpg
      + mỗi thư mục có index.json tóm tắt.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    # Các map cột linh động
    col_movie = _first_col(df, ["movie", "movie_title", "movie_id"])
    col_char = _first_col(df, ["character_id", "cluster_id"])
    col_image = _first_col(df, ["rep_image", "face_path", "frame_image", "crop_path", "image"])
    col_score = _first_col(df, ["score", "distance", "sim", "similarity"])
    col_frame = _first_col(df, ["rep_frame", "frame", "frame_idx", "start_frame"])

    if not col_movie or not col_char:
        raise RuntimeError("Không tìm thấy cột movie/character trong parquet.")

    # Lọc theo movie nếu cần
    if only_movie:
        # Cho phép truyền id hoặc title
        # nếu là id, map sang title
        title = _title_from_any(meta, only_movie) or only_movie
        if col_movie == "movie_id":
            # nếu df lưu id dạng số
            try:
                mid = int((_reverse_movie_id_map(meta) or {}).get(title, title))
            except Exception:
                mid = title
            mask = (df[col_movie].astype(str) == str(mid))
        else:
            mask = (df[col_movie].astype(str) == str(title))
        df = df[mask].copy()

    # group theo movie/char
    summary: Dict[str, Any] = {"total_movies": 0, "total_characters": 0, "written": 0, "skipped": 0}
    groups = df.groupby([col_movie, col_char], dropna=False)

    for (mv_key, ch_key), g in groups:
        # Chuẩn hoá movie title
        movie_title = _title_from_any(meta, mv_key) or str(mv_key)
        char_id = str(ch_key)

        # Tạo thư mục đích
        dst_dir = out_root / movie_title / char_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        items = []

        # sắp theo score giảm dần nếu có
        if col_score in g.columns:
            g = g.sort_values(col_score, ascending=False).copy()

        for _, row in g.iterrows():
            if written >= limit_per_char:
                break

            score = float(row[col_score]) if col_score in g.columns and pd.notna(row[col_score]) else None
            # cố gắng lấy path ảnh đại diện
            img_path = None
            if col_image and pd.notna(row.get(col_image)):
                img_path = Path(str(row[col_image]))
            else:
                # fallback: nếu có frame thì có thể bạn có cây frames/ nào đó — ở đây không suy luận thêm
                img_path = None

            if img_path is None:
                summary["skipped"] += 1
                continue

            # nếu path tương đối -> convert tương đối từ project root
            if not img_path.is_absolute():
                img_path = Path(os.getcwd()) / img_path

            im = _safe_open_image(img_path)
            if im is None:
                summary["skipped"] += 1
                continue

            # dựng các dòng label
            lines = [f"{movie_title} • char {char_id}"]
            if score is not None:
                lines.append(f"score: {score:.3f}")
            if col_frame and pd.notna(row.get(col_frame)):
                fr = _extract_frame_idx(row[col_frame])
                if fr is not None:
                    # nếu metadata có fps, quy đổi thời gian
                    fps = None
                    mi = meta.get(movie_title) or {}
                    fps = mi.get("fps")
                    if fps:
                        try:
                            t = float(fr) / float(fps)
                            lines.append(f"frame: {fr} • t={t:.2f}s")
                        except Exception:
                            lines.append(f"frame: {fr}")
                    else:
                        lines.append(f"frame: {fr}")

            im = _draw_label(im, lines)

            out_name = f"{written:04d}.jpg"
            out_path = dst_dir / out_name
            try:
                im.save(out_path, quality=90)
                items.append({"file": out_name, "score": score})
                written += 1
            except Exception:
                summary["skipped"] += 1
                continue

        # ghi index.json cho mỗi char
        try:
            with open(dst_dir / "index.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "movie": movie_title,
                        "character_id": char_id,
                        "count": written,
                        "items": items,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

        summary["written"] += written

    # Tổng quan
    try:
        summary["total_movies"] = int(df[col_movie].nunique())
        summary["total_characters"] = int(df[col_char].nunique())
    except Exception:
        pass

    return summary


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Xuất ảnh preview debug cho từng cụm/nhân vật."
    )
    parser.add_argument(
        "--movie",
        type=str,
        default="",
        help="Giới hạn 1 phim (nhập title hoặc id). Bỏ trống để xuất tất cả.",
    )
    parser.add_argument(
        "--limit-per-char",
        type=int,
        default=12,
        help="Số ảnh tối đa / nhân vật (mặc định 12).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="warehouse/cluster_previews",
        help="Thư mục output (mặc định warehouse/cluster_previews).",
    )
    args = parser.parse_args()

    cfg = _load_cfg()
    meta = _read_metadata(cfg)
    df = _load_clusters_parquet(cfg)

    if df is None or df.empty:
        print("⚠️  Không tìm thấy hoặc không đọc được parquet cụm (clusters_merged.parquet / clusters.parquet).")
        return

    out_root = Path(args.out)
    summary = export_previews(
        df=df,
        out_root=out_root,
        meta=meta,
        only_movie=(args.movie or None),
        limit_per_char=max(1, int(args.limit_per_char)),
    )

    print("✅ Xuất preview xong.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
