# tasks/merge_clusters_task.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from prefect import task
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from utils.vector_utils import l2_normalize


@task(name="Merge Clusters Task")
def merge_clusters_task(
        sim_threshold: float = 0.75,
        clusters_parquet: Optional[str] = None,
        output_parquet: Optional[str] = None,
) -> str:
    """
    Gộp các cụm tương tự nhau trong phạm vi từng phim (within-movie merge).
    Sử dụng cosine similarity giữa các centroid của cụm.
    """
    # Đường dẫn mặc định
    wh_dir = Path("warehouse") / "parquet"
    wh_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(clusters_parquet) if clusters_parquet else (wh_dir / "clusters.parquet")
    dst_path = Path(output_parquet) if output_parquet else (wh_dir / "clusters_merged.parquet")

    if not src_path.exists():
        raise FileNotFoundError(f"[Merge] Không tìm thấy file cụm: {src_path}")

    print("\n--- Starting Merge Clusters Task (within-movie only) ---")
    print(f"[Merge] Input : {src_path}")
    print(f"[Merge] Output: {dst_path}")
    print(f"[Merge] Similarity Threshold: {sim_threshold}")

    # Đọc dữ liệu cụm
    df = pd.read_parquet(src_path)

    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        print(f"[Merge] Single-movie mode → '{active_movie}'")
        if "movie" in df.columns:
            df = df[df["movie"].astype(str) == active_movie]

    if df.empty:
        print("[Merge] No data to process. Saving empty dataframe.")
        df.to_parquet(dst_path, index=False)
        return str(dst_path)

    # --- LOGIC MERGE THỰC SỰ ---

    # 1. Tính toán centroid cho mỗi cụm ban đầu
    # Sử dụng track_centroid là hợp lý vì nó đã được tính toán và ổn định
    cluster_centroids = df.groupby("cluster_id")["track_centroid"].apply(
        lambda x: l2_normalize(np.mean(np.stack(x), axis=0))
    )

    # Chuyển thành dataframe để dễ xử lý
    centroids_df = cluster_centroids.reset_index()
    centroids_df.columns = ["cluster_id", "centroid"]

    # Tách movie_id từ cluster_id (ví dụ: "NHAGIATIEN_12" -> "NHAGIATIEN")
    centroids_df["movie"] = centroids_df["cluster_id"].apply(lambda x: x.split("_")[0])

    all_merged_dfs = []

    # 2. Xử lý gộp cụm cho từng phim một
    for movie, movie_group in centroids_df.groupby("movie"):
        print(f"[Merge] Processing clusters for movie: {movie}")

        if len(movie_group) <= 1:
            # Không có gì để merge, giữ nguyên cluster_id
            mapping = {row.cluster_id: row.cluster_id for _, row in movie_group.iterrows()}
        else:
            # Lấy ma trận centroid của các cụm trong phim
            centroid_matrix = np.stack(movie_group["centroid"].values)

            # Sử dụng Agglomerative Clustering để gộp các centroid
            # pdist tính khoảng cách, linkage thực hiện gom cụm
            # 'cosine' distance = 1 - cosine_similarity. Ngưỡng distance = 1 - sim_threshold
            distance_threshold = 1.0 - sim_threshold

            # Sử dụng linkage 'average' hoặc 'complete' thường cho kết quả tốt
            Z = linkage(pdist(centroid_matrix, metric='cosine'), method='average')

            # Lấy nhãn cụm mới dựa trên ngưỡng distance
            new_labels = fcluster(Z, t=distance_threshold, criterion='distance')

            # Tạo mapping từ cluster_id cũ sang cluster_id mới (merged)
            movie_group = movie_group.copy()
            movie_group["new_label"] = new_labels
            mapping = movie_group.set_index("cluster_id")["new_label"].to_dict()

            # Tạo cluster_id mới có định dạng "movie_merged_label"
            mapping = {old_id: f"{movie}_merged_{new_label}" for old_id, new_label in mapping.items()}

        # Map các cluster_id cũ sang cluster_id mới cho dữ liệu gốc
        original_movie_df = df[df["cluster_id"].isin(mapping.keys())].copy()
        original_movie_df["cluster_id"] = original_movie_df["cluster_id"].map(mapping)
        all_merged_dfs.append(original_movie_df)

    if not all_merged_dfs:
        print("[Merge] No clusters were processed. Output will be empty.")
        merged_df = pd.DataFrame(columns=df.columns)
    else:
        merged_df = pd.concat(all_merged_dfs, ignore_index=True)

    # Ghi kết quả
    merged_df.to_parquet(dst_path, index=False)

    num_clusters_before = len(df["cluster_id"].unique())
    num_clusters_after = len(merged_df["cluster_id"].unique())

    print(f"[Merge] Cluster count reduced from {num_clusters_before} to {num_clusters_after}")
    print(f"[Merge] Đã lưu {len(merged_df)} records -> {dst_path}")

    return str(dst_path)