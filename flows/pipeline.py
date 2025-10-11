#!/usr/bin/env python3
"""
Pipeline chính cho face clustering theo từng phim.

Usage:
  # Xử lý 1 phim với preset tự động
  python -m flows.pipeline --movie NHAGIATIEN

  # Override preset thủ công
  python -m flows.pipeline --movie NHAGIATIEN --preset co_trang_dong_duc

  # Override knobs qua ENV
  FS_WITHIN=0.82 FS_MIN_SIZE=10 python -m flows.pipeline --movie NHAGIATIEN
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from prefect import flow

# Import tasks
from tasks.ingestion_task import ingestion_task
from tasks.embedding_task import embedding_task
from tasks.build_warehouse_task import build_warehouse_task
from tasks.cluster_task import cluster_task
from tasks.merge_clusters_task import merge_clusters_task
from tasks.filter_clusters_task import filter_clusters_task
from tasks.preview_clusters_task import preview_clusters_task
from tasks.character_task import character_task
from tasks.validation_task import validation_task

# Import utils
from utils.config_loader import (
    load_config,
    apply_preset,
    load_movie_metadata,
    extract_knobs,
)

# ============================================================================
# Helpers
# ============================================================================

def _banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70 + "\n")


def _print_active_config(profile_key: str, cfg: dict, movie: str) -> None:
    knobs = extract_knobs(cfg)

    print(f"\n📋 Active configuration for '{movie}':")
    print(f"  Profile: {profile_key}")
    print(f"\n  🎯 Tuning knobs:")
    print(f"    merge.within_movie_threshold: {knobs.get('within')}")
    print(f"    filter_clusters.min_size:     {knobs.get('min_size')}")
    print(f"    cluster.distance_threshold:   {knobs.get('dist_pca')}")
    print(f"    tracklet.max_age:             {knobs.get('max_age')}")
    print(f"    tracklet.iou_threshold:       {knobs.get('iou')}")

    print(f"\n  🔧 Full clustering config:")
    print(f"    algo:     {cfg['clustering'].get('algo')}")
    print(f"    linkage:  {cfg['clustering'].get('linkage')}")
    print(f"    metric:   {cfg['clustering'].get('metric')}")
    print()

# ============================================================================
# Main flow
# ============================================================================

@flow(name="Face Clustering Pipeline")
def face_clustering_pipeline(
    movie: Optional[str] = None,
    preset: Optional[str] = None,
    skip_ingestion: bool = False,
    skip_embedding: bool = False,
) -> dict:
    """
    Pipeline chính xử lý face clustering (KHÔNG dùng PCA).
    """

    # Load base config
    base_cfg = load_config()

    # Xác định active movie
    env_movie = os.getenv("FS_ACTIVE_MOVIE", "").strip()
    active_movie = (movie or env_movie or "").strip()

    if active_movie:
        _banner(f"🎬 SINGLE-MOVIE MODE: {active_movie}")
        os.environ["FS_ACTIVE_MOVIE"] = active_movie  # set ENV cho các task

        # Load movie metadata
        movie_meta = load_movie_metadata(active_movie, base_cfg)

        print(f"📂 Movie metadata:")
        print(f"  era:          {movie_meta.get('era') or 'N/A'}")
        print(f"  genre:        {movie_meta.get('genre') or 'N/A'}")
        print(f"  context_tags: {movie_meta.get('context_tags') or []}")
        if movie_meta.get('custom_knobs'):
            print(f"  custom_knobs: YES")

        # Apply preset
        profile_key, cfg = apply_preset(
            base_cfg,
            era=movie_meta.get("era"),
            genre=movie_meta.get("genre"),
            context_tags=movie_meta.get("context_tags"),
            profile=preset,  # manual override nếu có
            custom_knobs=movie_meta.get("custom_knobs"),
        )

        _print_active_config(profile_key, cfg, active_movie)

    else:
        _banner("🎬 MULTI-MOVIE MODE (Legacy)")
        cfg = base_cfg
        profile_key = "default"
        print("[INFO] Processing all movies with default config.\n")

    # ========================================================================
    # Stage 1: Ingestion (frames extraction)
    # ========================================================================
    if not skip_ingestion:
        _banner("Stage 1: Ingestion")
        ingestion_task(movie=active_movie if active_movie else None)
    else:
        print("[SKIP] Ingestion stage (frames already exist)\n")

    # ========================================================================
    # Stage 2: Embedding (face detection + embedding + tracklets)
    # ========================================================================
    if not skip_embedding:
        _banner("Stage 2: Embedding")
        embedding_task()
    else:
        print("[SKIP] Embedding stage (embeddings already exist)\n")

    # ========================================================================
    # Stage 3: Build Warehouse (aggregate per-movie → global)
    # ========================================================================
    _banner("Stage 3: Build Warehouse")
    warehouse_path = build_warehouse_task()

    # ========================================================================
    # Stage 4: Clustering (Agglomerative/HDBSCAN) - NO PCA
    # ========================================================================
    _banner("Stage 4: Clustering")
    clusters_path = cluster_task()

    # ========================================================================
    # Stage 5: Merge Clusters (within-movie only by default)
    # ========================================================================
    _banner("Stage 5: Merge Clusters")
    merged_path = merge_clusters_task(
        sim_threshold=cfg["merge"].get("within_movie_threshold", 0.75)
    )

    # ========================================================================
    # Stage 6: Filter Low-Quality Clusters
    # ========================================================================
    _banner("Stage 6: Filter Clusters")
    characters_path = filter_clusters_task(
        clusters=None,
        characters_path=None,
        cfg=cfg,
        movie=active_movie if active_movie else None,
    )

    # ========================================================================
    # Stage 7: Generate Preview Images
    # ========================================================================
    _banner("Stage 7: Preview Generation")
    previews_root = preview_clusters_task(cfg=cfg)

    # ========================================================================
    # Stage 8: Build Character Manifest (API/FE)
    # ========================================================================
    _banner("Stage 8: Character Manifest")
    manifest_path = character_task()

    # ========================================================================
    # Stage 9: Validation & Quality Reports
    # ========================================================================
    _banner("Stage 9: Validation")
    reports = validation_task()

    # ========================================================================
    # Summary
    # ========================================================================
    _banner("✅ PIPELINE COMPLETED")
    summary = {
        "movie": active_movie or "all",
        "preset": profile_key,
        "warehouse": warehouse_path,
        "clusters": clusters_path,
        "characters": characters_path,
        "previews": previews_root,
        "manifest": manifest_path,
        "reports": reports,
    }

    print("📊 Outputs:")
    for k, v in summary.items():
        print(f"  {k:15s}: {v}")

    print("\n💡 Next steps:")
    print(f"  1. Review metrics: cat {reports.get('cluster_metrics', 'reports/cluster_metrics.csv')}")
    print(f"  2. Check previews: ls {previews_root}")
    print(
        f"  3. Test search:    python -c 'from utils.search_actor import search_actor; print(search_actor(\"face.jpg\"))'")
    print()

    return summary


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Face clustering pipeline (no PCA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--movie",
        type=str,
        default=None,
        help="Tên phim cần xử lý (không đuôi). Vd: NHAGIATIEN",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset profile để override auto-detection. Vd: co_trang_dong_duc",
    )

    # CLI override cho metadata (optional)
    parser.add_argument("--era", type=str, default=None, help="Era: co_trang, hien_dai, xua")
    parser.add_argument("--genre", type=str, default=None, help="Thể loại: hanh_dong, tam_ly, kinh_di, tinh_cam, hai")
    parser.add_argument("--context", type=str, default=None, help="Context tags, ví dụ: toi,dong_duc,rung_lac")

    parser.add_argument("--skip-ingestion", action="store_true", help="Bỏ qua trích xuất frames (nếu đã có)")
    parser.add_argument("--skip-embedding", action="store_true", help="Bỏ qua embedding (nếu đã có)")

    args = parser.parse_args()

    # Run pipeline
    try:
        face_clustering_pipeline(
            movie=args.movie,
            preset=args.preset,
            skip_ingestion=args.skip_ingestion,
            skip_embedding=args.skip_embedding,
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
