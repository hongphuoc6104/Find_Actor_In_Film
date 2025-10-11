# flows/pipeline.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import time  # THAY ĐỔI 1: Import module time
from typing import Optional

import pandas as pd
from prefect import flow

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
from utils.config_loader import load_config, apply_preset, load_movie_metadata, extract_knobs, deep_merge

def _banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70 + "\n")

def _print_active_config(profile_key: str, cfg: dict, movie: str) -> None:
    knobs = extract_knobs(cfg)
    print(f"\n📋 Active configuration for '{movie}':")
    print(f"  Profile: {profile_key}")
    print(f"\n   Tuning knobs:")
    print(f"    merge.within_movie_threshold: {knobs.get('within')}")
    print(f"    post_merge.distance_threshold: {cfg.get('post_merge', {}).get('distance_threshold')}")
    print(f"    filter_clusters.min_size:     {cfg['filter_clusters']['min_size']}")
    print(f"    clustering.distance_threshold: {cfg['clustering']['distance_threshold']['default']}")
    print(f"\n   Full clustering config:")
    print(f"    algo:     {cfg['clustering'].get('algo')}")
    print(f"    linkage:  {cfg['clustering'].get('linkage')}")
    print(f"    metric:   {cfg['clustering'].get('metric')}")
    print()

@flow(name="Face Clustering Pipeline")
def face_clustering_pipeline(
        movie: Optional[str] = None,
        preset: Optional[str] = None,
        min_size_override: Optional[int] = None,
        skip_ingestion: bool = False,
        skip_embedding: bool = False,
) -> dict:
    # --- Phần logic pipeline giữ nguyên ---
    base_cfg = load_config()
    env_movie = os.getenv("FS_ACTIVE_MOVIE", "").strip()
    active_movie = (movie or env_movie or "").strip()
    if active_movie:
        _banner(f"🎬 SINGLE-MOVIE MODE: {active_movie}")
        os.environ["FS_ACTIVE_MOVIE"] = active_movie
        movie_meta = load_movie_metadata(active_movie, base_cfg)
        profile_key, cfg = apply_preset(base_cfg, era=movie_meta.get("era"), genre=movie_meta.get("genre"), context_tags=movie_meta.get("context_tags"), profile=preset, custom_knobs=movie_meta.get("custom_knobs"))
        if min_size_override is not None and min_size_override > 0:
            print(f"[INFO] Overriding min_size from config with user value: {min_size_override}")
            override_cfg = {"filter_clusters": {"min_size": min_size_override}}
            cfg = deep_merge(cfg, override_cfg)
        _print_active_config(profile_key, cfg, active_movie)
    else:
        _banner("🎬 MULTI-MOVIE MODE (Legacy)")
        cfg = base_cfg; profile_key = "default"
        print("[INFO] Processing all movies with default config.\n")

    if not skip_ingestion: _banner("Stage 1: Ingestion"); ingestion_task(movie=active_movie or None)
    else: print("[SKIP] Ingestion stage (frames already exist)\n")
    if not skip_embedding: _banner("Stage 2: Embedding"); embedding_task()
    else: print("[SKIP] Embedding stage (embeddings already exist)\n")
    _banner("Stage 3: Build Warehouse"); warehouse_path = build_warehouse_task()
    _banner("Stage 4: Clustering"); clusters_path = cluster_task()
    _banner("Stage 5: Merge Clusters (Core Identification)"); merged_path = merge_clusters_task(sim_threshold=cfg["merge"].get("within_movie_threshold", 0.75))
    _banner("Stage 6: Filter Clusters (Select Cores)"); core_clusters_df = filter_clusters_task(clusters=None, cfg=cfg, movie=active_movie or None)
    _banner("Stage 7: Post-Merge (Satellite Assimilation)"); all_merged_df = pd.read_parquet(merged_path); final_clusters_df = post_merge_task(core_clusters_df=core_clusters_df, all_merged_clusters_df=all_merged_df, cfg=cfg)
    _banner("Stage 8: Preview Generation"); previews_root = preview_clusters_task(filtered_clusters_df=final_clusters_df, cfg=cfg)
    _banner("Stage 9: Character Manifest"); manifest_path = character_task(filtered_clusters_df=final_clusters_df)
    _banner("Stage 10: Validation"); reports = validation_task()

    _banner("✅ PIPELINE COMPLETED")
    summary = {"movie": active_movie or "all", "preset": profile_key, "warehouse": warehouse_path, "clusters": clusters_path, "characters": manifest_path, "previews": previews_root, "manifest": manifest_path, "reports": reports}
    print("📊 Outputs:")
    for k, v in summary.items(): print(f"  {k:15s}: {v}")
    print("\n💡 Next steps:")
    print(f"  1. Review metrics: cat {reports.get('cluster_metrics', 'reports/cluster_metrics.csv')}")
    print(f"  2. Check previews: ls {previews_root}")
    print(f"  3. Test search:    python -c 'from utils.search_actor import search_actor; print(search_actor(\"face.jpg\"))'")
    print()
    return summary

def main():
    parser = argparse.ArgumentParser(description="Face clustering pipeline", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--movie", type=str, default=None, help="Tên phim cần xử lý")
    parser.add_argument("--preset", type=str, default=None, help="Preset profile để override")
    parser.add_argument("--min-size", type=int, default=None, help="Override min_size cho việc lọc cụm")
    parser.add_argument("--skip-ingestion", action="store_true", help="Bỏ qua trích xuất frames")
    parser.add_argument("--skip-embedding", action="store_true", help="Bỏ qua embedding")
    args = parser.parse_args()
    try:
        face_clustering_pipeline(movie=args.movie, preset=args.preset, min_size_override=args.min_size, skip_ingestion=args.skip_ingestion, skip_embedding=args.skip_embedding)
        # THAY ĐỔI 2: Thêm một khoảng chờ ngắn trước khi thoát
        time.sleep(1)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        # THAY ĐỔI 3: Thêm khoảng chờ ngay cả khi có lỗi
        time.sleep(1)
        sys.exit(1)

if __name__ == "__main__":
    main()

