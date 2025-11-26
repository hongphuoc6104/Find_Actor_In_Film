# flows/pipeline.py
# !/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import time
import json
from typing import Optional, Dict, Any

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
# [Mới] Import task gán nhãn
from tasks.assign_labels_task import assign_labels_task

# Import utils
from utils.config_loader import load_config, apply_preset, load_movie_metadata, deep_merge


def _banner(msg: str):
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70 + "\n")


def _print_active_config(profile_key: str, cfg: dict, movie: str):
    print(f"\n📋 Active configuration for '{movie}':")
    print(f"  Profile strategy: {profile_key}")
    print(f"\n   FINAL TUNING PARAMETERS (after auto-tuning):")

    qf = cfg.get('quality_filters', {})
    lqf = qf.get('landmark_quality_filter', {})
    fc = cfg.get('filter_clusters', {})
    cl = cfg.get('clustering', {}).get('distance_threshold', {})
    m = cfg.get('merge', {})

    print(f"    - quality_filters.min_det_score:     {qf.get('min_det_score')}")
    print(f"    - quality_filters.min_face_size:     {qf.get('min_face_size')}")
    print(f"    - quality_filters.min_score_hard_cutoff: {lqf.get('min_score_hard_cutoff')}")
    print(f"    - filter_clusters.min_size:          {fc.get('min_size')}")
    print(f"    - clustering.distance_threshold:     {cl.get('default')}")
    print(f"    - merge.within_movie_threshold:      {m.get('within_movie_threshold')}")
    print()


@flow(name="Face Clustering Pipeline")
def face_clustering_pipeline(
        movie: Optional[str] = None,
        preset: Optional[str] = None,
        params_override: Optional[Dict[str, Any]] = None,
        skip_ingestion: bool = False,
        skip_embedding: bool = False,
) -> dict:
    base_cfg = load_config()
    active_movie = (movie or os.getenv("FS_ACTIVE_MOVIE", "")).strip()
    if not active_movie:
        raise ValueError("Single-movie mode is required.")

    _banner(f"🎬 SINGLE-MOVIE MODE: {active_movie}")
    os.environ["FS_ACTIVE_MOVIE"] = active_movie

    video_found = True
    if not skip_ingestion:
        _banner("Stage 1: Ingestion")
        video_found = ingestion_task(movie=active_movie)
    else:
        print("[SKIP] Ingestion stage.\n")

    if not video_found:
        message = f"Video file for '{active_movie}' not found. Stopping pipeline."
        print(f"\n {message}")
        _banner(" PIPELINE COMPLETED (Early Exit)")
        return {"status": "FAILED", "message": message}

    _banner("Stage 1.5: Video Analysis")
    video_profile = analyze_video_task(movie_title=active_movie)

    movie_meta = load_movie_metadata(active_movie, base_cfg)

    profile_key, cfg = apply_preset(
        base_cfg,
        video_profile=video_profile,
        custom_knobs=movie_meta.get("custom_knobs"),
    )

    if params_override:
        print(f"[INFO] Applying user-defined manual overrides: {params_override}")
        cfg = deep_merge(cfg, params_override)

    _print_active_config(profile_key, cfg, active_movie)

    if not skip_embedding:
        _banner("Stage 2: Embedding")
        embedding_task(cfg=cfg)
    else:
        print("[SKIP] Embedding stage.\n")

    _banner("Stage 3: Build Warehouse")
    warehouse_path, row_count = build_warehouse_task()
    if row_count == 0:
        message = f"No data processed for movie '{active_movie}'. Stopping pipeline."
        print(f"\n {message}")
        _banner(" PIPELINE COMPLETED (Early Exit)")
        return {"status": "SKIPPED", "message": message}

    _banner("Stage 4: Clustering");
    cluster_task(cfg=cfg)
    _banner("Stage 5: Merge Clusters");
    merge_clusters_task(sim_threshold=cfg.get("merge", {}).get("within_movie_threshold"))

    _banner("Stage 6: Filter Clusters");
    core_clusters_df = filter_clusters_task(clusters=None, cfg=cfg, movie=active_movie)

    _banner("Stage 7: Post-Merge");
    all_merged_df = pd.read_parquet(cfg["storage"]["clusters_merged_parquet"]);
    final_clusters_df = post_merge_task(core_clusters_df=core_clusters_df, all_merged_clusters_df=all_merged_df,
                                        cfg=cfg)

    _banner("Stage 8: Preview Generation");
    previews_root = preview_clusters_task(filtered_clusters_df=final_clusters_df, cfg=cfg)

    _banner("Stage 9: Character Manifest")
    manifest_path = character_task(filtered_clusters_df=final_clusters_df, cfg=cfg)

    # --- [Mới] Gán nhãn tự động ---
    _banner("Stage 11: Auto Labeling")
    assign_labels_task(cfg=cfg)

    _banner("Stage 10: Validation")
    reports = validation_task(cfg=cfg)

    try:
        metadata_filepath = base_cfg["storage"]["metadata_json"]
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, "r+", encoding="utf-8") as f:
                all_metadata = json.load(f)
                used_params = {
                    "min_det_score": cfg.get("quality_filters", {}).get("min_det_score"),
                    "min_size": cfg.get("filter_clusters", {}).get("min_size"),
                }
                all_metadata.setdefault(active_movie, {})
                all_metadata[active_movie]["last_run_params"] = used_params
                f.seek(0)
                json.dump(all_metadata, f, indent=4, ensure_ascii=False)
                f.truncate()
                print(f"[INFO] Updated last run parameters for '{active_movie}' in {metadata_filepath}")
    except Exception as e:
        print(f"[WARN] Could not save last run parameters to metadata.json: {e}")

    _banner(" PIPELINE COMPLETED")
    summary = {"movie": active_movie, "preset": profile_key, "characters": manifest_path, "previews": previews_root}
    print("📊 Outputs:")
    for k, v in summary.items(): print(f"  {k:15s}: {v}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Face clustering pipeline")
    parser.add_argument("--movie", type=str, required=True, help="Tên phim cần xử lý (bắt buộc)")
    parser.add_argument("--preset", type=str, default=None, help="Preset profile để override")
    parser.add_argument("--skip-ingestion", action="store_true", help="Bỏ qua trích xuất frames")
    parser.add_argument("--skip-embedding", action="store_true", help="Bỏ qua embedding")
    args = parser.parse_args()
    try:
        face_clustering_pipeline(
            movie=args.movie,
            preset=args.preset,
            skip_ingestion=args.skip_ingestion,
            skip_embedding=args.skip_embedding
        )
    except Exception as e:
        print(f"\n Pipeline failed: {e}", file=sys.stderr)
        import traceback;
        traceback.print_exc()


if __name__ == "__main__":
    main()