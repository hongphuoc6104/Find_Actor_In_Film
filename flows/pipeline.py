# flows/pipeline.py
# !/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import time
import json  # <-- THÊM MỚI
from typing import Optional, Dict, Any  # <-- THÊM MỚI

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
    print(f"\n   TUNING KNOBS:")
    print(f"    - min_det_score:                {cfg.get('quality_filters', {}).get('min_det_score')}")
    print(f"    - min_size (for core clusters): {cfg.get('filter_clusters', {}).get('min_size')}")
    print(f"\n   Other important settings:")
    print(f"    - clustering.distance_threshold: {cfg['clustering']['distance_threshold']['default']}")
    print(f"    - merge.within_movie_threshold:  {knobs.get('within')}")
    print()


# --- THAY ĐỔI 1: Thêm tham số `params_override` ---
@flow(name="Face Clustering Pipeline")
def face_clustering_pipeline(
        movie: Optional[str] = None,
        preset: Optional[str] = None,
        params_override: Optional[Dict[str, Any]] = None,  # <-- THÊM MỚI
        skip_ingestion: bool = False,
        skip_embedding: bool = False,
) -> dict:
    base_cfg = load_config()
    env_movie = os.getenv("FS_ACTIVE_MOVIE", "").strip()
    active_movie = (movie or env_movie or "").strip()

    if not active_movie:
        raise ValueError("Single-movie mode is required. Please provide a movie title.")

    _banner(f"🎬 SINGLE-MOVIE MODE: {active_movie}")
    os.environ["FS_ACTIVE_MOVIE"] = active_movie
    movie_meta = load_movie_metadata(active_movie, base_cfg)

    profile_key, cfg = apply_preset(
        base_cfg,
        era=movie_meta.get("era"),
        genre=movie_meta.get("genre"),
        context_tags=movie_meta.get("context_tags"),
        profile=preset,
        custom_knobs=movie_meta.get("custom_knobs")
    )

    # --- THAY ĐỔI 2: Merge các tham số từ người dùng (nếu có) ---
    if params_override:
        print(f"[INFO] Applying user-defined parameters: {params_override}")
        cfg = deep_merge(cfg, params_override)

    _print_active_config(profile_key, cfg, active_movie)

    # --- Các bước pipeline giữ nguyên ---
    if not skip_ingestion:
        _banner("Stage 1: Ingestion"); ingestion_task(movie=active_movie)
    else:
        print("[SKIP] Ingestion stage.\n")
    if not skip_embedding:
        _banner("Stage 2: Embedding"); embedding_task()
    else:
        print("[SKIP] Embedding stage.\n")

    _banner("Stage 3: Build Warehouse");
    warehouse_path = build_warehouse_task()
    _banner("Stage 4: Clustering");
    clusters_path = cluster_task()
    _banner("Stage 5: Merge Clusters");
    merged_path = merge_clusters_task(sim_threshold=cfg["merge"].get("within_movie_threshold", 0.75))
    _banner("Stage 6: Filter Clusters");
    core_clusters_df = filter_clusters_task(clusters=None, cfg=cfg, movie=active_movie)
    _banner("Stage 7: Post-Merge");
    all_merged_df = pd.read_parquet(merged_path);
    final_clusters_df = post_merge_task(core_clusters_df=core_clusters_df, all_merged_clusters_df=all_merged_df,
                                        cfg=cfg)
    _banner("Stage 8: Preview Generation");
    previews_root = preview_clusters_task(filtered_clusters_df=final_clusters_df, cfg=cfg)
    _banner("Stage 9: Character Manifest");
    manifest_path = character_task(filtered_clusters_df=final_clusters_df)
    _banner("Stage 10: Validation");
    reports = validation_task()

    # --- THAY ĐỔI 3: Lưu lại các tham số đã dùng vào metadata.json ---
    try:
        metadata_filepath = base_cfg["storage"]["metadata_json"]
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, "r+", encoding="utf-8") as f:
                all_metadata = json.load(f)

                # Lấy các giá trị thực tế đã dùng
                used_min_det_score = cfg.get("quality_filters", {}).get("min_det_score")
                used_min_size = cfg.get("filter_clusters", {}).get("min_size")

                # Tạo mục để lưu
                all_metadata.setdefault(active_movie, {})
                all_metadata[active_movie]["last_run_params"] = {
                    "min_det_score": used_min_det_score,
                    "min_size": used_min_size,
                }

                # Ghi lại file
                f.seek(0)
                json.dump(all_metadata, f, indent=4, ensure_ascii=False)
                f.truncate()
                print(f"[INFO] Updated last run parameters for '{active_movie}' in {metadata_filepath}")
    except Exception as e:
        print(f"[WARN] Could not save last run parameters to metadata.json: {e}")

    _banner("✅ PIPELINE COMPLETED")
    summary = {"movie": active_movie, "preset": profile_key, "characters": manifest_path, "previews": previews_root}
    print("📊 Outputs:")
    for k, v in summary.items(): print(f"  {k:15s}: {v}")
    print()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Face clustering pipeline")
    parser.add_argument("--movie", type=str, required=True, help="Tên phim cần xử lý (bắt buộc)")
    parser.add_argument("--preset", type=str, default=None, help="Preset profile để override")
    parser.add_argument("--skip-ingestion", action="store_true", help="Bỏ qua trích xuất frames")
    parser.add_argument("--skip-embedding", action="store_true", help="Bỏ qua embedding")
    # Gỡ bỏ --min-size vì giờ sẽ truyền qua params_override
    args = parser.parse_args()
    try:
        face_clustering_pipeline(
            movie=args.movie,
            preset=args.preset,
            skip_ingestion=args.skip_ingestion,
            skip_embedding=args.skip_embedding
        )
        time.sleep(1)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback;
        traceback.print_exc()
        time.sleep(1)
        sys.exit(1)


if __name__ == "__main__":
    main()