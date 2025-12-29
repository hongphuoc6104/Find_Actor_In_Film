# flows/pipeline.py
# !/usr/bin/env python3
from __future__ import annotations
from typing import Optional


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
from tasks.evaluation_task import evaluation_task

# Import utils
from utils.config_loader import load_config, load_video_config, apply_preset, load_movie_metadata, deep_merge


def _banner(msg: str):
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70 + "\n")


@flow(name="Face Clustering Pipeline")
def face_clustering_pipeline(
        movie: str,
        preset: Optional[str] = None,
        params_override: Optional[dict] = None,  # For compatibility with celery_worker
        skip_ingestion: bool = False,
        skip_embedding: bool = False,
        skip_evaluation: bool = False,
        skip_preview: bool = False,
        skip_validation: bool = False,
        skip_labeling: bool = False,
        quiet: bool = False,
) -> dict:
    # 0. Setup
    active_movie = movie.strip()
    os.environ["FS_ACTIVE_MOVIE"] = active_movie
    
    # Helper to conditionally print banners
    def banner(msg: str):
        if not quiet:
            _banner(msg)
    
    # Load base config with video-specific overrides (if exists)
    base_cfg = load_video_config(active_movie)

    banner(f"🎬 PIPELINE START: {active_movie}")

    # 1. Ingestion
    if not skip_ingestion:
        banner("Stage 1: Ingestion")
        found = ingestion_task(movie=active_movie)
        if not found:
            print(f"[Error] Video '{active_movie}' not found.")

            return {"status": "FAILED"}

    # 1.5 Analysis
    banner("Stage 1.5: Video Analysis")
    video_profile = analyze_video_task(movie_title=active_movie)


    # Apply Config with duration-based auto-tuning
    movie_meta = load_movie_metadata(active_movie, base_cfg)
    
    # Get duration for auto-tuning presets
    duration_seconds = None
    try:
        from utils.config_loader import get_video_duration
        duration_seconds = get_video_duration(active_movie, base_cfg)
        if duration_seconds:
            print(f"[Info] Video duration: {int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}")
    except Exception:
        pass
    
    profile_key, cfg = apply_preset(
        base_cfg, 
        video_profile, 
        movie_meta.get("custom_knobs"),
        duration_seconds=duration_seconds
    )
    print(f"[Info] Applied Profile: {profile_key}")



    # 2. Embedding
    if not skip_embedding:
        banner("Stage 2: Embedding")
        embedding_task(cfg=cfg)

    # 3. Warehouse
    banner("Stage 3: Build Warehouse")
    warehouse_path, row_count = build_warehouse_task()
    if row_count == 0:
        print("[Stop] No embeddings found.")

        return {"status": "SKIPPED"}

    # 4. Clustering
    banner("Stage 4: Clustering")
    cluster_task(cfg=cfg)

    # 5. Merge
    banner("Stage 5: Merge Clusters")
    merge_clusters_task(sim_threshold=cfg.get("merge", {}).get("within_movie_threshold"))

    # 6. Filter
    banner("Stage 6: Filter Clusters")
    core_clusters_df = filter_clusters_task(cfg=cfg, movie=active_movie)

    # 7. Post-Merge
    banner("Stage 7: Post-Merge (Assimilation)")
    all_merged_df = pd.read_parquet(cfg["storage"]["clusters_merged_parquet"])
    final_clusters_df = post_merge_task(core_clusters_df, all_merged_df, cfg)

    # 8. Previews (optional)
    if not skip_preview:
        banner("Stage 8: Preview Generation")
        previews_root = preview_clusters_task(filtered_clusters_df=final_clusters_df, cfg=cfg)
    else:
        previews_root = cfg["storage"]["cluster_previews_root"]
        if not quiet: print("[Skip] Preview generation skipped")

    # 9. Manifest (Generation)
    # [FIX] Logic trong character_task đã được sửa để không overwrite nhãn cũ
    banner("Stage 9: Character Manifest")
    manifest_path = character_task(filtered_clusters_df=final_clusters_df, cfg=cfg)

    # 10. Auto Labeling (optional)
    if not skip_labeling:
        banner("Stage 10: Auto Labeling")
        assign_labels_task(cfg=cfg)
    else:
        if not quiet: print("[Skip] Auto labeling skipped")

    # 11. Validation (optional - writes reports/)
    if not skip_validation:
        banner("Stage 11: Validation")
        validation_task(cfg=cfg)
    else:
        if not quiet: print("[Skip] Validation skipped")

    # 12. Clustering Evaluation (for parameter tuning)
    if not skip_evaluation:
        banner("Stage 12: Clustering Evaluation")
        metrics = evaluation_task(cfg=cfg)
        if metrics and not quiet:
            print(f"[Info] Evaluation completed. Check warehouse/evaluation/ for detailed results.")
    else:
        if not quiet: print("[Skip] Evaluation skipped")


    banner("✅ PIPELINE COMPLETED")
    return {"status": "SUCCESS", "manifest": manifest_path}


def main():
    parser = argparse.ArgumentParser(description="Face Clustering Pipeline")
    parser.add_argument("--movie", type=str, required=True, help="Movie name to process")
    
    # Skip options
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip video ingestion")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip face embedding")
    parser.add_argument("--skip-preview", action="store_true", help="Skip preview image generation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation (reports/)")
    parser.add_argument("--skip-labeling", action="store_true", help="Skip auto-labeling")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip clustering evaluation")
    
    # Fast mode: skip all optional stages
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip preview, validation, labeling, evaluation")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Fast mode enables all skips
    if args.fast:
        args.skip_preview = True
        args.skip_validation = True
        args.skip_labeling = True
        args.skip_evaluation = True

    face_clustering_pipeline(
        movie=args.movie,
        skip_ingestion=args.skip_ingestion,
        skip_embedding=args.skip_embedding,
        skip_evaluation=args.skip_evaluation,
        skip_preview=args.skip_preview,
        skip_validation=args.skip_validation,
        skip_labeling=args.skip_labeling,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()