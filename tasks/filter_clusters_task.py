from __future__ import annotations

import json
import os
from typing import Dict, Tuple, Optional

import pandas as pd
from prefect import task

from utils.config_loader import load_config


def _series_mean_safe(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else 0.0


def _to_key(x) -> str:
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _compute_stats(
    clusters: pd.DataFrame,
    frame_col: str | None,
) -> pd.DataFrame:
    """
    Trả về bảng thống kê theo (movie_id, final_character_id):
      - size: số bản ghi
      - mean_det: điểm detect trung bình (0 nếu không có cột)
      - frames: số frame/track duy nhất (fallback = size)
    """
    # pandas >= 2.0: dict-of-named-agg
    agg: Dict[str, Tuple[str, str] | Tuple[str, callable]] = {"size": ("final_character_id", "size")}
    if "det_score" in clusters.columns:
        agg["mean_det"] = ("det_score", _series_mean_safe)
    if frame_col:
        agg["frames"] = (frame_col, "nunique")

    stats = (
        clusters.groupby(["movie_id", "final_character_id"], as_index=False)
        .agg(**agg)
    )

    if "frames" not in stats.columns:
        stats["frames"] = stats["size"]
    if "mean_det" not in stats.columns:
        stats["mean_det"] = 0.0

    return stats


def _load_clusters_if_needed(cfg: dict, clusters: Optional[pd.DataFrame]) -> pd.DataFrame:
    if clusters is not None and not clusters.empty:
        return clusters
    path = cfg["storage"]["warehouse_clusters"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"[FilterClusters] Không tìm thấy clusters parquet: {path}")
    return pd.read_parquet(path)


def _characters_path_if_needed(cfg: dict, characters_path: Optional[str]) -> str:
    return characters_path or cfg["storage"]["characters_json"]


def _active_movie_name_to_id(cfg: dict, movie_name: Optional[str]) -> Optional[int]:
    if not movie_name:
        return None
    # Ưu tiên map sinh ra trong metadata
    meta_path = cfg["storage"].get("metadata_json")
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            m = (meta or {}).get("_generated", {}).get("movie_id_map") or {}
            if movie_name in m:
                return int(m[movie_name])
        except Exception:
            pass
    # Fallback: cố tìm qua warehouse embeddings/clusters
    # (không bắt buộc; chỉ khi map chưa có)
    try:
        wh = cfg["storage"]["warehouse_embeddings"]
        if os.path.exists(wh):
            df = pd.read_parquet(wh, columns=["movie", "movie_id"])
            tmp = df[df["movie"].astype(str) == movie_name]
            if not tmp.empty:
                return int(tmp["movie_id"].iloc[0])
    except Exception:
        pass
    return None


@task(name="Filter Clusters Task")
def filter_clusters_task(
    clusters: pd.DataFrame | None = None,
    characters_path: str | None = None,
    cfg: dict | None = None,
    movie: str | None = None,          # tuỳ chọn: chỉ lọc riêng 1 phim theo tên
):
    """
    Lọc cụm chất lượng thấp theo từng phim và cập nhật characters.json.

    Tương thích ngược:
      - Nếu flow cũ gọi task *không truyền gì*, task sẽ:
          * tự load clusters từ cfg.storage.warehouse_clusters
          * tự lấy đường dẫn characters_json từ cfg.storage.characters_json
      - Hỗ trợ chạy đơn-phim qua:
          * ENV FS_ACTIVE_MOVIE="TEN_PHIM"
          * hoặc tham số movie="TEN_PHIM"

    Tham số config liên quan:
      filter_clusters:
        min_size: int = 6
        min_det: float? = None
        min_frames: int? = None
        adaptive: bool = False
      adaptive: nâng ngưỡng per-movie theo phân vị (P50 size, P40 det/frames).
    """

    # --- Load config & input paths ---
    cfg = cfg or load_config()
    clusters_df = _load_clusters_if_needed(cfg, clusters)
    characters_path = _characters_path_if_needed(cfg, characters_path)

    if clusters_df.empty:
        print("[FilterClusters] clusters rỗng → bỏ qua.")
        return characters_path

    # --- Chuẩn hoá movie_id và lọc đơn-phim nếu được yêu cầu ---
    if "movie_id" not in clusters_df.columns:
        if "movie" in clusters_df.columns:
            # cố gắng map sang movie_id ổn định nếu có
            print("[FilterClusters] 'movie_id' không có; sẽ giữ nguyên theo 'movie'.")
        else:
            # ép toàn bộ về 0 để không crash
            clusters_df = clusters_df.copy()
            clusters_df["movie_id"] = 0

    active_movie_name = (movie or os.getenv("FS_ACTIVE_MOVIE", "")).strip()
    active_movie_id: Optional[int] = None
    if active_movie_name:
        active_movie_id = _active_movie_name_to_id(cfg, active_movie_name)

    if active_movie_name and active_movie_id is not None and "movie_id" in clusters_df.columns:
        before = len(clusters_df)
        clusters_df = clusters_df[clusters_df["movie_id"] == int(active_movie_id)]
        print(f"[FilterClusters] Single-movie mode: '{active_movie_name}' (id={active_movie_id}); "
              f"rows {before} → {len(clusters_df)}")
    elif active_movie_name and active_movie_id is None:
        print(f"[FilterClusters][WARN] Không tra được movie_id cho '{active_movie_name}'. "
              f"Sẽ lọc theo toàn bộ movies hiện có.")

    # --- Guard: bắt buộc phải có final_character_id để lọc ---
    if "final_character_id" not in clusters_df.columns:
        print("[FilterClusters] Thiếu cột 'final_character_id' → bỏ qua lọc, giữ nguyên characters.json.")
        return characters_path

    # --- Config thresholds ---
    fcfg = (cfg or {}).get("filter_clusters", {}) or {}
    min_size_cfg = int(fcfg.get("min_size", 6))
    min_det_cfg = fcfg.get("min_det", None)          # None → không xét
    min_frames_cfg = fcfg.get("min_frames", None)    # None → không xét
    adaptive = bool(fcfg.get("adaptive", False))

    df = clusters_df.copy()

    # Đảm bảo có movie_id (nếu chỉ có 'movie' thì convert tạm để join)
    if "movie_id" not in df.columns:
        # không có map tin cậy → rải movie_id=0; vẫn chạy được theo nhóm
        df["movie_id"] = 0

    # Cột frame tốt nhất hiện có
    frame_col = "frame" if "frame" in df.columns else ("track_id" if "track_id" in df.columns else None)

    # Thống kê per (movie_id, final_character_id)
    stats = _compute_stats(df, frame_col)

    # Ngưỡng mặc định (cố định)
    use_min_size = pd.Series([min_size_cfg] * len(stats))
    use_min_det = None
    use_min_frames = None
    if "mean_det" in stats.columns and (min_det_cfg is not None):
        use_min_det = pd.Series([float(min_det_cfg)] * len(stats))
    if "frames" in stats.columns and (min_frames_cfg is not None):
        use_min_frames = pd.Series([int(min_frames_cfg)] * len(stats))

    # Adaptive: nâng ngưỡng theo phân vị per-movie
    if adaptive:
        def per_movie_thresh(g: pd.DataFrame) -> pd.Series:
            t: Dict[str, float] = {}
            t["min_size"] = max(min_size_cfg, int(g["size"].quantile(0.50)))
            if "mean_det" in g.columns:
                base_det = float(min_det_cfg) if (min_det_cfg is not None) else 0.0
                t["min_det"] = max(base_det, float(g["mean_det"].quantile(0.40)))
            if "frames" in g.columns:
                base_frames = int(min_frames_cfg) if (min_frames_cfg is not None) else 0
                t["min_frames"] = max(base_frames, int(g["frames"].quantile(0.40)))
            return pd.Series(t)

        movie_th = stats.groupby("movie_id", as_index=False).apply(per_movie_thresh)
        stats = stats.merge(movie_th, on="movie_id", how="left")
        use_min_size = stats["min_size"]
        if "min_det" in stats.columns:
            use_min_det = stats["min_det"]
        if "min_frames" in stats.columns:
            use_min_frames = stats["min_frames"]

    # Điều kiện giữ
    cond = stats["size"] >= use_min_size
    if use_min_det is not None:
        cond &= stats["mean_det"] >= use_min_det
    if use_min_frames is not None:
        cond &= stats["frames"] >= use_min_frames

    keep_pairs = stats.loc[cond, ["movie_id", "final_character_id"]]
    valid_pairs = {( _to_key(r.movie_id), _to_key(r.final_character_id) ) for r in keep_pairs.itertuples()}

    # Đọc characters.json, lọc theo valid_pairs (và theo single-movie nếu có)
    if not os.path.exists(characters_path):
        print(f"[FilterClusters] Không tìm thấy characters.json tại {characters_path}. Bỏ qua ghi.")
        return characters_path

    with open(characters_path, "r", encoding="utf-8") as f:
        try:
            characters: Dict[str, Dict[str, dict]] = json.load(f)
        except json.JSONDecodeError:
            print("[FilterClusters] characters.json lỗi/không hợp lệ → giữ nguyên file.")
            return characters_path

    cleaned: Dict[str, Dict[str, dict]] = {}
    removed = 0
    kept = 0

    # Nếu single-movie: chỉ xét movie_id tương ứng
    allowed_movie_ids = None
    if active_movie_id is not None:
        allowed_movie_ids = {str(active_movie_id)}

    for movie_id, char_map in characters.items():
        if allowed_movie_ids is not None and str(movie_id) not in allowed_movie_ids:
            # Bỏ qua movies khác khi chạy single-movie
            continue
        if not isinstance(char_map, dict):
            continue
        retained: Dict[str, dict] = {}
        for char_id, data in char_map.items():
            if (str(movie_id), str(char_id)) in valid_pairs:
                retained[str(char_id)] = data
                kept += 1
            else:
                removed += 1
        if retained:
            cleaned[str(movie_id)] = retained

    # Ghi atomic
    _atomic_write_json(characters_path, cleaned)

    print(
        f"[FilterClusters] movies_kept={len(cleaned)} kept_chars={kept} removed_chars={removed} "
        f"(min_size={min_size_cfg}"
        f"{'' if min_det_cfg is None else f', min_det={min_det_cfg}'}"
        f"{'' if min_frames_cfg is None else f', min_frames={min_frames_cfg}'}"
        f", adaptive={adaptive}"
        f"{'' if not active_movie_name else f'; single_movie={active_movie_name}'}"
        f")"
    )

    return characters_path
