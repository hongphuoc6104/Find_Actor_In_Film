# flows/pipeline.py
# !/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import json
import pandas as pd
from prefect import flow

# Import tasks
from tasks.analyze_video_task import analyze_video_task
from tasks.ingestion_task import ingestion_task
from tasks.embedding_task import embedding_task
from tasks.build_warehouse_task import build_warehouse_task
from tasks.cluster_task import cluster_task
from tasks.merge_clusters_task import merge_clusters_task
from tasks.filter_clusters_task import filter_clusters_task
from tasks.post_merge_task import post_merge_task
from tasks.preview_clusters_task import preview_clusters_task
from tasks.character_task import character_task
from tasks.validation_task import validation_task
from tasks.assign_labels_task import assign_labels_task

# Import utils
from utils.config_loader import load_config, apply_preset, load_movie_metadata, deep_merge


def _banner(msg: str):
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70 + "\n")


@flow(name="Face Clustering Pipeline")
def face_clustering_pipeline(
        movie: str,
        preset: str = None,
        skip_ingestion: bool = False,
        skip_embedding: bool = False,
) -> dict:
    # 0. Setup
    active_movie = movie.strip()
    os.environ["FS_ACTIVE_MOVIE"] = active_movie
    base_cfg = load_config()
    _banner(f"🎬 PIPELINE START: {active_movie}")

    # 1. Ingestion
    if not skip_ingestion:
        _banner("Stage 1: Ingestion")
        found = ingestion_task(movie=active_movie)
        if not found:
            print(f"[Error] Video '{active_movie}' not found.")
            return {"status": "FAILED"}

    # 1.5 Analysis
    _banner("Stage 1.5: Video Analysis")
    video_profile = analyze_video_task(movie_title=active_movie)

    # Apply Config
    movie_meta = load_movie_metadata(active_movie, base_cfg)
    profile_key, cfg = apply_preset(base_cfg, video_profile, movie_meta.get("custom_knobs"))
    print(f"[Info] Applied Profile: {profile_key}")

    # 2. Embedding
    if not skip_embedding:
        _banner("Stage 2: Embedding")
        embedding_task(cfg=cfg)

    # 3. Warehouse
    _banner("Stage 3: Build Warehouse")
    warehouse_path, row_count = build_warehouse_task()
    if row_count == 0:
        print("[Stop] No embeddings found.")
        return {"status": "SKIPPED"}

    # 4. Clustering
    _banner("Stage 4: Clustering")
    cluster_task(cfg=cfg)

    # 5. Merge
    _banner("Stage 5: Merge Clusters")
    merge_clusters_task(sim_threshold=cfg.get("merge", {}).get("within_movie_threshold"))

    # 6. Filter
    _banner("Stage 6: Filter Clusters")
    core_clusters_df = filter_clusters_task(cfg=cfg, movie=active_movie)

    # 7. Post-Merge
    _banner("Stage 7: Post-Merge (Assimilation)")
    all_merged_df = pd.read_parquet(cfg["storage"]["clusters_merged_parquet"])
    final_clusters_df = post_merge_task(core_clusters_df, all_merged_df, cfg)

    # 8. Previews
    _banner("Stage 8: Preview Generation")
    previews_root = preview_clusters_task(filtered_clusters_df=final_clusters_df, cfg=cfg)

    # 9. Manifest (Generation)
    # [FIX] Logic trong character_task đã được sửa để không overwrite nhãn cũ
    _banner("Stage 9: Character Manifest")
    manifest_path = character_task(filtered_clusters_df=final_clusters_df, cfg=cfg)

    # 10. Auto Labeling
    # Chạy sau Manifest để update tên vào file json
    _banner("Stage 10: Auto Labeling")
    assign_labels_task(cfg=cfg)

    # 11. Validation
    _banner("Stage 11: Validation")
    validation_task(cfg=cfg)

    _banner("✅ PIPELINE COMPLETED")
    return {"status": "SUCCESS", "manifest": manifest_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie", type=str, required=True)
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    args = parser.parse_args()

    face_clustering_pipeline(
        movie=args.movie,
        skip_ingestion=args.skip_ingestion,
        skip_embedding=args.skip_embedding
    )


if __name__ == "__main__":
    main()