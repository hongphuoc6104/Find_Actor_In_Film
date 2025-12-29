# tasks/post_merge_task.py
import os
import numpy as np
import pandas as pd
from prefect import task
from scipy.spatial.distance import cdist
from pathlib import Path
from utils.vector_utils import l2_normalize


@task(name="Post-Merge Task: Assimilate Satellites")
def post_merge_task(
        core_clusters_df: pd.DataFrame,
        all_merged_clusters_df: pd.DataFrame,
        cfg: dict,
) -> pd.DataFrame:
    """
    Hấp thụ các cụm vệ tinh và LƯU TRỮ PER-MOVIE PARQUET để phục vụ tìm kiếm lâu dài.
    """
    print("\n--- Starting Post-Merge Task (Satellite Assimilation) ---")
    post_merge_cfg = cfg.get("post_merge", {})
    storage = cfg.get("storage", {})

    # Setup thư mục lưu trữ Parquet riêng lẻ
    # Lấy thư mục cha của warehouse_clusters (thường là warehouse/parquet/)
    wh_cluster_path = Path(storage.get("warehouse_clusters", "warehouse/parquet/clusters.parquet"))
    wh_parquet_dir = wh_cluster_path.parent
    wh_parquet_dir.mkdir(parents=True, exist_ok=True)

    # 1. [CRITICAL] Lưu File Parquet Riêng Cho Từng Phim TRƯỚC KHI filter
    # Điều này đảm bảo per-movie parquet luôn được lưu, bất kể filter có pass hay không
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    
    if active_movie and all_merged_clusters_df is not None and not all_merged_clusters_df.empty:
        # Lưu tất cả clusters (không chỉ filtered) để search hoạt động
        if "movie" in all_merged_clusters_df.columns:
            movie_df = all_merged_clusters_df[all_merged_clusters_df["movie"] == active_movie]
        else:
            movie_df = all_merged_clusters_df
        
        if not movie_df.empty:
            per_movie_path = wh_parquet_dir / f"{active_movie}_clusters.parquet"
            movie_df.to_parquet(per_movie_path, index=False)
            print(f"[PostMerge] ✅ Saved PERSISTENT cluster data for '{active_movie}' -> {per_movie_path} ({len(movie_df)} rows)")
        else:
            print(f"[PostMerge] Warning: No data for movie '{active_movie}' in all_merged_clusters_df.")

    # 2. Bảo vệ đầu vào - nhưng vẫn return per-movie đã được lưu ở trên
    if core_clusters_df is None or core_clusters_df.empty:
        print("[PostMerge] core_clusters_df is empty, skipping satellite assimilation but per-movie parquet was saved.")
        return all_merged_clusters_df if all_merged_clusters_df is not None else pd.DataFrame()

    # 3. Chuẩn hóa tên cột
    char_col = next((c for c in ["final_character_id", "cluster_id"] if c in all_merged_clusters_df.columns), None)
    if not char_col:
        print("[PostMerge] Missing ID column.")
        return core_clusters_df

    core_df = core_clusters_df.copy().rename(columns={char_col: "final_character_id"})
    all_df = all_merged_clusters_df.copy().rename(columns={char_col: "final_character_id"})

    # 3. Satellite Assimilation Logic
    final_df = all_df.copy()

    if post_merge_cfg.get("enable", True):
        metric = post_merge_cfg.get("metric", "cosine")
        threshold = post_merge_cfg.get("distance_threshold", 0.60)
        min_core_size = 10  # Clusters >= 10 faces are "core", < 10 are "satellites"
        
        print(f"[PostMerge] Satellite assimilation: metric={metric}, threshold={threshold}")
        
        # Check if embeddings exist
        if "emb" not in all_df.columns:
            print("[PostMerge] No embeddings found, skipping assimilation")
        else:
            # Identify core and satellite clusters
            cluster_sizes = all_df.groupby("final_character_id").size()
            core_clusters = cluster_sizes[cluster_sizes >= min_core_size].index.tolist()
            satellite_clusters = cluster_sizes[cluster_sizes < min_core_size].index.tolist()
            
            if not core_clusters:
                print(f"[PostMerge] No core clusters found (min_size={min_core_size}), skipping")
            elif not satellite_clusters:
                print(f"[PostMerge] No satellite clusters to assimilate")
            else:
                print(f"[PostMerge] Found {len(core_clusters)} core clusters, {len(satellite_clusters)} satellites")
                
                # Compute core centroids
                core_centroids = {}
                for core_id in core_clusters:
                    core_faces = all_df[all_df["final_character_id"] == core_id]
                    embs = np.vstack(core_faces["emb"].values)
                    centroid = embs.mean(axis=0)
                    if metric == "cosine":
                        centroid = l2_normalize(centroid.reshape(1, -1)).flatten()
                    core_centroids[core_id] = centroid
                
                # Assimilate satellites
                assimilated_count = 0
                for sat_id in satellite_clusters:
                    sat_faces = all_df[all_df["final_character_id"] == sat_id]
                    sat_embs = np.vstack(sat_faces["emb"].values)
                    sat_centroid = sat_embs.mean(axis=0)
                    if metric == "cosine":
                        sat_centroid = l2_normalize(sat_centroid.reshape(1, -1)).flatten()
                    
                    # Find nearest core
                    min_dist = float("inf")
                    best_core = None
                    
                    for core_id, core_centroid in core_centroids.items():
                        if metric == "cosine":
                            # Cosine distance = 1 - cosine similarity
                            # Both vectors are already normalized
                            dist = 1.0 - np.dot(sat_centroid, core_centroid)
                        else:
                            dist = np.linalg.norm(sat_centroid - core_centroid)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_core = core_id
                    
                    # Assimilate if within threshold
                    if min_dist < threshold and best_core is not None:
                        final_df.loc[final_df["final_character_id"] == sat_id, "final_character_id"] = best_core
                        assimilated_count += 1
                
                print(f"[PostMerge] Assimilated {assimilated_count}/{len(satellite_clusters)} satellites into cores")


    # 5. Vẫn lưu file merged tổng tạm thời (cho các bước sau trong cùng pipeline dùng)
    merged_path = Path(storage.get("clusters_merged_parquet", "warehouse/parquet/clusters_merged.parquet"))
    final_df.to_parquet(merged_path, index=False)

    return final_df