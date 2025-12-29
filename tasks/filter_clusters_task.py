# tasks/filter_clusters_task.py
from __future__ import annotations
import json
import os
from typing import Dict, Tuple

import pandas as pd
from prefect import task
from utils.config_loader import load_config


def _load_clusters_if_needed(cfg: dict, clusters: pd.DataFrame | None) -> pd.DataFrame:
    if clusters is not None and not clusters.empty: return clusters
    path = cfg["storage"]["clusters_merged_parquet"]
    if not os.path.exists(path): raise FileNotFoundError(f"Không tìm thấy clusters_merged_parquet: {path}")
    return pd.read_parquet(path)


def _active_movie_name_to_id(cfg: dict, movie_name: str | None) -> int | None:
    if not movie_name: return None
    meta_path = cfg["storage"].get("metadata_json")
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            m = (meta or {}).get("_generated", {}).get("movie_id_map") or {}
            if movie_name in m: return int(m[movie_name])
        except Exception:
            pass
    return None


@task(name="Filter Clusters Task")
def filter_clusters_task(
        clusters: pd.DataFrame | None = None,
        cfg: dict | None = None,
        movie: str | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    clusters_df = _load_clusters_if_needed(cfg, clusters)

    if clusters_df.empty:
        print("[FilterClusters] clusters rỗng → trả về dataframe rỗng.")
        return clusters_df

    if "cluster_id" in clusters_df.columns and "final_character_id" not in clusters_df.columns:
        clusters_df = clusters_df.rename(columns={"cluster_id": "final_character_id"})

    if "movie_id" not in clusters_df.columns:
        clusters_df["movie_id"] = 0

    active_movie_name = (movie or os.getenv("FS_ACTIVE_MOVIE", "")).strip()
    if active_movie_name:
        active_movie_id = _active_movie_name_to_id(cfg, active_movie_name)
        if active_movie_id is not None and "movie_id" in clusters_df.columns:
            before = len(clusters_df)
            clusters_df = clusters_df[clusters_df["movie_id"] == int(active_movie_id)]
            print(
                f"[FilterClusters] Single-movie mode: '{active_movie_name}' (id={active_movie_id}); rows {before} → {len(clusters_df)}")

    if "final_character_id" not in clusters_df.columns:
        print("[FilterClusters] Thiếu cột 'final_character_id' → Bỏ qua lọc.")
        return clusters_df

    fcfg = cfg.get("filter_clusters", {})
    min_size_cfg = int(fcfg.get("min_size", 10))

    # Lấy ngưỡng chất lượng cao từ config
    lq_filter_cfg = cfg.get("quality_filters", {}).get("landmark_quality_filter", {})
    min_score_for_core = lq_filter_cfg.get("min_score_for_core", 0.6)

    df = clusters_df.copy()

    # THAY ĐỔI CỐT LÕI: Tính toán số lượng ảnh chất lượng cao trong mỗi cụm
    if 'quality_score' in df.columns:
        high_quality_counts = df[df['quality_score'] >= min_score_for_core].groupby('final_character_id').size().rename(
            'high_quality_size')
    else:
        # Fallback nếu không có cột quality_score, dùng kích thước tổng
        print("[FilterClusters][WARN] Không tìm thấy cột 'quality_score'. Sẽ lọc theo kích thước tổng của cụm.")
        high_quality_counts = df.groupby('final_character_id').size().rename('high_quality_size')

    # Join thông tin này vào kích thước tổng của cụm
    cluster_stats = df.groupby('final_character_id').size().rename('total_size').to_frame()
    cluster_stats = cluster_stats.join(high_quality_counts).fillna(0)

    # Một cụm được chọn làm HẠT NHÂN nếu có đủ số lượng ảnh CHẤT LƯỢNG CAO
    core_cluster_ids = cluster_stats[cluster_stats['high_quality_size'] >= min_size_cfg].index

    # Giữ lại TẤT CẢ các ảnh (cả chất lượng cao và trung bình) của các cụm được chọn làm hạt nhân
    df_filtered = df[df['final_character_id'].isin(core_cluster_ids)]

    print(
        f"[FilterClusters] Hoàn thành. Chọn {len(core_cluster_ids)} cụm làm hạt nhân "
        f"(ngưỡng high_quality_size >= {min_size_cfg}, min_score_for_core={min_score_for_core}). "
        f"Giữ lại {len(df_filtered)} records."
    )

    return df_filtered

