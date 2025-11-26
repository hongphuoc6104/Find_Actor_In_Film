# tasks/post_merge_task.py
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
    Hấp thụ các cụm vệ tinh vào các cụm hạt nhân gần nhất.
    Phiên bản này đã được nâng cấp để linh hoạt hơn với tên cột.
    """
    print("\n--- Starting Post-Merge Task (Satellite Assimilation) ---")
    post_merge_cfg = cfg.get("post_merge", {})
    if not post_merge_cfg.get("enable", True):
        print("[PostMerge] Disabled by config. Returning core clusters only.")
        return core_clusters_df

    # --- CẬP NHẬT 1: Điều kiện bảo vệ ban đầu ---
    if core_clusters_df is None or core_clusters_df.empty:
        print("[PostMerge] No core clusters provided to assimilate into. Returning empty dataframe.")
        # Trả về DataFrame rỗng với đúng các cột để tránh lỗi ở các bước sau
        return pd.DataFrame(columns=all_merged_clusters_df.columns if all_merged_clusters_df is not None else None)

    if all_merged_clusters_df is None or all_merged_clusters_df.empty:
        print("[PostMerge] No merged clusters data available. Returning core clusters only.")
        return core_clusters_df

    # --- CẬP NHẬT 2: Tự động tìm tên cột ID và thống nhất nó ---
    # Tìm tên cột ID (ví dụ: 'final_character_id' hoặc 'cluster_id') trong DataFrame tổng
    char_col = next((c for c in ["final_character_id", "cluster_id"] if c in all_merged_clusters_df.columns), None)

    # Kiểm tra các cột thiết yếu
    if not char_col or "track_centroid" not in all_merged_clusters_df.columns:
        print(
            f"[PostMerge] Missing required columns ('final_character_id'/'cluster_id', 'track_centroid'). Cannot perform assimilation.")
        return core_clusters_df

    # Tạo bản sao và thống nhất tên cột thành 'final_character_id' để xử lý nội bộ
    # Điều này đảm bảo logic bên dưới luôn hoạt động với một tên cột duy nhất
    core_df = core_clusters_df.copy().rename(columns={char_col: "final_character_id"})
    all_df = all_merged_clusters_df.copy().rename(columns={char_col: "final_character_id"})
    # --- KẾT THÚC CẬP NHẬT ---

    metric = post_merge_cfg.get("metric", "cosine")
    distance_threshold = post_merge_cfg.get("distance_threshold", 0.7)
    print(f"[PostMerge] Metric: {metric}, Distance Threshold: {distance_threshold}")

    core_ids = core_df["final_character_id"].unique()
    satellite_df = all_df[~all_df["final_character_id"].isin(core_ids)]

    if satellite_df.empty:
        print("[PostMerge] No satellite clusters to assimilate. Returning core clusters.")
        return core_df

    print("[PostMerge] Calculating centroids for all clusters...")
    # Tính toán centroid một cách an toàn hơn, bỏ qua các giá trị None
    all_centroids = all_df.groupby("final_character_id")["track_centroid"].apply(
        lambda e: l2_normalize(np.mean(np.stack([v for v in e if v is not None]), axis=0))
    ).reset_index()

    core_centroids = all_centroids[all_centroids["final_character_id"].isin(core_ids)]
    satellite_centroids = all_centroids[~all_centroids["final_character_id"].isin(core_ids)]

    if core_centroids.empty or satellite_centroids.empty:
        print("[PostMerge] Not enough core or satellite centroids to perform assimilation.")
        return core_df

    print(
        f"[PostMerge] Calculating distances from {len(satellite_centroids)} satellites to {len(core_centroids)} cores...")
    core_matrix = np.stack(core_centroids["track_centroid"].values)
    satellite_matrix = np.stack(satellite_centroids["track_centroid"].values)
    dist_matrix = cdist(satellite_matrix, core_matrix, metric=metric)

    min_distances = dist_matrix.min(axis=1)
    closest_core_indices = dist_matrix.argmin(axis=1)

    assimilation_map = {}
    assimilated_count = 0
    for i, (satellite_id, dist) in enumerate(zip(satellite_centroids["final_character_id"], min_distances)):
        if dist <= distance_threshold:
            closest_core_id = core_centroids["final_character_id"].iloc[closest_core_indices[i]]
            assimilation_map[satellite_id] = closest_core_id
            assimilated_count += 1

    print(f"[PostMerge] Assimilated {assimilated_count} / {len(satellite_centroids)} satellite clusters.")

    # Áp dụng bản đồ hấp thụ vào DataFrame tổng (đã được thống nhất tên cột)
    final_df = all_df.copy()
    final_df["final_character_id"] = final_df["final_character_id"].replace(assimilation_map)

    final_cluster_count = final_df["final_character_id"].nunique()
    print(f"[PostMerge] Final cluster count: {final_cluster_count}")

    return final_df