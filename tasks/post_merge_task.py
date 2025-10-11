# tasks/post_merge_task.py
from __future__ import annotations

import numpy as np
import pandas as pd
from prefect import task
from scipy.spatial.distance import cdist

from utils.vector_utils import l2_normalize


@task(name="Post-Merge Task: Assimilate Satellites")
def post_merge_task(
        core_clusters_df: pd.DataFrame,
        all_merged_clusters_df: pd.DataFrame,
        cfg: dict,
) -> pd.DataFrame:
    """
    Hấp thụ các cụm vệ tinh (chất lượng thấp/góc cạnh) vào các cụm hạt nhân (chất lượng cao).

    Parameters:
    - core_clusters_df: Dataframe chỉ chứa các cụm hạt nhân (đã được lọc).
    - all_merged_clusters_df: Dataframe chứa TẤT CẢ các cụm sau bước merge đầu tiên.
    - cfg: File config.

    Returns:
    - Dataframe cuối cùng với các cụm vệ tinh đã được gán lại nhãn của cụm hạt nhân.
    """
    post_merge_cfg = cfg.get("post_merge", {})
    if not post_merge_cfg.get("enable", False) or core_clusters_df.empty:
        print("[PostMerge] Post-merge bị tắt hoặc không có cụm hạt nhân. Bỏ qua.")
        # Nếu bỏ qua, trả về các cụm hạt nhân đã được lọc
        return core_clusters_df

    print("\n--- Starting Post-Merge Task (Satellite Assimilation) ---")
    distance_threshold = float(post_merge_cfg.get("distance_threshold", 0.45))
    metric = post_merge_cfg.get("metric", "cosine")
    print(f"[PostMerge] Metric: {metric}, Distance Threshold: {distance_threshold}")

    # Đổi tên cột từ filter_task để nhất quán
    if "final_character_id" in core_clusters_df.columns:
        core_clusters_df = core_clusters_df.rename(columns={"final_character_id": "cluster_id"})

    # Xác định ID của các cụm hạt nhân và vệ tinh
    core_ids = set(core_clusters_df["cluster_id"].unique())
    all_ids = set(all_merged_clusters_df["cluster_id"].unique())
    satellite_ids = all_ids - core_ids

    if not satellite_ids:
        print("[PostMerge] Không có cụm vệ tinh nào để hấp thụ.")
        return core_clusters_df

    # Tính centroid cho TẤT CẢ các cụm
    print("[PostMerge] Calculating centroids for all clusters...")
    all_centroids = all_merged_clusters_df.groupby("cluster_id")["track_centroid"].apply(
        lambda x: l2_normalize(np.mean(np.stack(x), axis=0))
    )

    core_centroids_map = {cid: all_centroids[cid] for cid in core_ids if cid in all_centroids}
    satellite_centroids_map = {cid: all_centroids[cid] for cid in satellite_ids if cid in all_centroids}

    if not core_centroids_map or not satellite_centroids_map:
        print("[PostMerge] Thiếu centroids cho hạt nhân hoặc vệ tinh. Bỏ qua.")
        return core_clusters_df

    # Chuẩn bị ma trận để tính khoảng cách
    core_labels = list(core_centroids_map.keys())
    satellite_labels = list(satellite_centroids_map.keys())
    core_matrix = np.stack(list(core_centroids_map.values()))
    satellite_matrix = np.stack(list(satellite_centroids_map.values()))

    # Tính khoảng cách từ mỗi vệ tinh đến TẤT CẢ các hạt nhân
    print(f"[PostMerge] Calculating distances from {len(satellite_labels)} satellites to {len(core_labels)} cores...")
    dist_matrix = cdist(satellite_matrix, core_matrix, metric=metric)

    # Tìm hạt nhân gần nhất cho mỗi vệ tinh
    closest_core_indices = np.argmin(dist_matrix, axis=1)
    min_distances = np.min(dist_matrix, axis=1)

    # Tạo mapping để gộp vệ tinh vào hạt nhân nếu đủ gần
    assimilation_map = {}
    assimilated_count = 0
    for i, satellite_id in enumerate(satellite_labels):
        if min_distances[i] < distance_threshold:
            closest_core_id = core_labels[closest_core_indices[i]]
            assimilation_map[satellite_id] = closest_core_id
            assimilated_count += 1

    print(f"[PostMerge] Assimilated {assimilated_count} / {len(satellite_labels)} satellite clusters.")

    # Áp dụng mapping vào dataframe chứa TẤT CẢ các cụm
    final_df = all_merged_clusters_df.copy()
    final_df["cluster_id"] = final_df["cluster_id"].map(assimilation_map).fillna(final_df["cluster_id"])

    # Chỉ giữ lại các cụm thuộc về hạt nhân ban đầu hoặc đã được hấp thụ vào hạt nhân
    final_ids_to_keep = core_ids.union(set(assimilation_map.values()))
    final_df = final_df[final_df["cluster_id"].isin(final_ids_to_keep)]

    print(f"[PostMerge] Final cluster count: {len(final_df['cluster_id'].unique())}")

    return final_df
