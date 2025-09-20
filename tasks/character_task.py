from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from prefect import task
from sklearn.cluster import AgglomerativeClustering

from utils.config_loader import load_config
from utils.indexer import build_index
from utils.vector_utils import _mean_vector, l2_normalize
from tasks.filter_clusters_task import filter_clusters_task


def _frame_to_int(frame_name: Any) -> int:
    base = os.path.splitext(str(frame_name))[0]
    digits = "".join(ch for ch in base if ch.isdigit())
    try:
        return int(digits)
    except ValueError:
        return -1


def _timestamp_from_frame(frame_idx: int, fps: float | None) -> float | None:
    if fps is None or fps <= 0 or frame_idx < 0:
        return None
    return round(frame_idx / fps, 3)


def _as_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype="float32")
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1)


def _normalize_bbox(bbox: Any) -> List[int]:
    if isinstance(bbox, (list, tuple)):
        return [int(float(x)) for x in bbox]
    arr = np.asarray(bbox).astype("float32").tolist()
    return [int(float(x)) for x in arr]


@task(name="Build Character Profiles Task")
def character_task():
    """Xây dựng hồ sơ nhân vật riêng cho từng phim."""

    print("\n--- Starting Character Profile Task ---")
    cfg = load_config()
    storage_cfg = cfg.get("storage", {})
    post_merge_cfg = cfg.get("post_merge", {})

    clusters_path = storage_cfg["warehouse_clusters"]
    embeddings_path = storage_cfg.get("warehouse_embeddings")
    output_json_path = storage_cfg["characters_json"]
    merged_parquet_path = storage_cfg.get("clusters_merged_parquet")

    print(f"[Character] Loading clustered data from {clusters_path}...")

    clusters_df = pd.read_parquet(clusters_path)
    if clusters_df.empty:
        print("[Character] No clustered data to process. Skipping task.")
        return None

    if "movie_id" not in clusters_df.columns:
        if "movie" in clusters_df.columns:
            clusters_df["movie_id"] = (
                clusters_df["movie"].astype("category").cat.codes.astype(int)
            )
        else:
            clusters_df["movie_id"] = 0

    movie_name_by_id: Dict[int, str] = {}
    if "movie" in clusters_df.columns:
        movie_name_by_id = (
            clusters_df.groupby("movie_id")["movie"].apply(
                lambda values: next(
                    (str(v) for v in values if isinstance(v, str) and v),
                    str(values.iloc[0]) if len(values) else "",
                )
            )
        ).to_dict()

    fps_by_movie: Dict[str, float] = {}
    metadata_path = storage_cfg.get("metadata_json")
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for movie_name, info in metadata.items():
                fps = info.get("fps") or info.get("FPS")
                if fps:
                    try:
                        fps_by_movie[str(movie_name)] = float(fps)
                    except (TypeError, ValueError):
                        continue
        except (OSError, json.JSONDecodeError):
            print(f"[WARN] Could not read metadata file at {metadata_path}")

    full_embeddings: pd.DataFrame | None = None
    if embeddings_path and os.path.exists(embeddings_path):
        full_embeddings = pd.read_parquet(embeddings_path)
        if not full_embeddings.empty:
            if "movie_id" not in full_embeddings.columns:
                if "movie" in full_embeddings.columns:
                    full_embeddings["movie_id"] = (
                        full_embeddings["movie"].astype("category").cat.codes.astype(int)
                    )
                else:
                    full_embeddings["movie_id"] = 0
            if "frame_index" not in full_embeddings.columns:
                full_embeddings["frame_index"] = full_embeddings["frame"].apply(
                    _frame_to_int
                )
    else:
        print("[WARN] Full embeddings parquet not found – scene metadata will be limited.")

    per_movie_records: List[pd.DataFrame] = []
    centroid_rows: List[Dict[str, Any]] = []
    characters: Dict[str, Dict[str, Any]] = {}

    for movie_id, movie_group in clusters_df.groupby("movie_id"):
        movie_group = movie_group.copy()
        movie_name = movie_name_by_id.get(movie_id, str(movie_id))
        fps = fps_by_movie.get(movie_name)

        print(f"[Character] Processing movie_id={movie_id} ({movie_name})")
        centroids = (
            movie_group.groupby("cluster_id")["track_centroid"]
            .apply(
                lambda rows: l2_normalize(
                    _mean_vector([_as_array(v) for v in rows if v is not None])
                )
            )
            .reset_index()
        )

        if centroids.empty:
            continue

        centroid_vectors = np.stack(centroids["track_centroid"].to_list()).astype("float32")

        if post_merge_cfg.get("enable", False) and len(centroid_vectors) > 1:
            distance_th = float(post_merge_cfg.get("distance_threshold", 0.35))
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_th,
                metric=post_merge_cfg.get("metric", "cosine"),
                linkage=post_merge_cfg.get("linkage", "average"),
            )
            labels = clusterer.fit_predict(centroid_vectors)
            label_map = {
                lbl: idx for idx, lbl in enumerate(sorted(set(labels.tolist())))
            }
            mapping = {
                row.cluster_id: label_map[label]
                for row, label in zip(centroids.itertuples(), labels.tolist())
            }
        else:
            mapping = {
                row.cluster_id: idx
                for idx, row in enumerate(
                    centroids.sort_values("cluster_id").itertuples()
                )
            }

        movie_group["final_character_id"] = movie_group["cluster_id"].map(mapping).astype(str)
        per_movie_records.append(movie_group)

        movie_characters: Dict[str, Dict[str, Any]] = {}

        if full_embeddings is not None and not full_embeddings.empty:
            movie_embeddings = full_embeddings[full_embeddings["movie_id"] == movie_id]
        else:
            movie_embeddings = pd.DataFrame()

        for final_id, tracks in movie_group.groupby("final_character_id"):
            track_ids = (
                tracks.get("track_id")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            track_vectors = [
                _as_array(v) for v in tracks["track_centroid"] if v is not None
            ]
            if not track_vectors:
                continue
            centroid_vec = l2_normalize(_mean_vector(track_vectors))

            centroid_rows.append(
                {
                    "movie_id": int(movie_id),
                    "movie": movie_name,
                    "character_id": str(final_id),
                    "embedding": centroid_vec.tolist(),
                }
            )

            scenes: List[Dict[str, Any]] = []
            rep_row: pd.Series | None = None

            if not movie_embeddings.empty and track_ids:
                scene_rows = movie_embeddings["track_id"].isin(track_ids)
                scene_df = movie_embeddings[scene_rows].copy()
                if not scene_df.empty:
                    if "frame_index" not in scene_df.columns:
                        scene_df["frame_index"] = scene_df["frame"].apply(_frame_to_int)
                    scene_df.sort_values(["frame_index", "track_id"], inplace=True)
                    for order_idx, row in enumerate(scene_df.itertuples()):
                        frame_idx = int(row.frame_index) if row.frame_index == row.frame_index else -1
                        timestamp = _timestamp_from_frame(frame_idx, fps)
                        bbox = _normalize_bbox(row.bbox) if hasattr(row, "bbox") else []
                        scenes.append(
                            {
                                "order": order_idx,
                                "frame": row.frame,
                                "frame_index": frame_idx if frame_idx >= 0 else None,
                                "timestamp": timestamp,
                                "bbox": bbox,
                                "track_id": int(row.track_id)
                                if hasattr(row, "track_id") and row.track_id == row.track_id
                                else None,
                                "det_score": float(row.det_score)
                                if hasattr(row, "det_score")
                                else None,
                            }
                        )
                    rep_idx = scene_df["det_score"].idxmax() if "det_score" in scene_df else scene_df.index[0]
                    rep_row = scene_df.loc[rep_idx]

            if rep_row is None and not tracks.empty:
                rep_idx = tracks["det_score"].idxmax() if "det_score" in tracks else tracks.index[0]
                rep_row = tracks.loc[rep_idx]

            rep_image = {}
            if rep_row is not None:
                rep_bbox = (
                    rep_row["bbox"] if "bbox" in rep_row else []
                )
                rep_image = {
                    "movie": rep_row.get("movie", movie_name),
                    "frame": rep_row.get("frame"),
                    "bbox": _normalize_bbox(rep_bbox) if rep_bbox is not None else [],
                    "det_score": float(rep_row.get("det_score", 0.0)),
                }

            previews_root = storage_cfg.get("cluster_previews_root")
            preview_paths: List[str] = []
            preview_entries: List[Dict[str, Any]] = []
            if previews_root:
                for raw_cluster_id in sorted(tracks["cluster_id"].astype(str).unique()):
                    cluster_dir = os.path.join(previews_root, f"cluster_{raw_cluster_id}")
                    if not os.path.isdir(cluster_dir):
                        continue
                    meta_file = os.path.join(cluster_dir, "metadata.json")
                    if os.path.exists(meta_file):
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta_entries = json.load(f)
                            for entry in meta_entries:
                                entry = dict(entry)
                                entry["cluster_id"] = raw_cluster_id
                                preview_img = entry.get("preview_image")
                                annotated_img = entry.get("annotated_image")
                                if preview_img and not os.path.isabs(preview_img):
                                    entry["preview_image"] = os.path.join(cluster_dir, preview_img)
                                if annotated_img and not os.path.isabs(annotated_img):
                                    entry["annotated_image"] = os.path.join(
                                        cluster_dir, annotated_img
                                    )
                                if entry.get("preview_image"):
                                    preview_paths.append(entry["preview_image"])
                                preview_entries.append(entry)
                        except (OSError, json.JSONDecodeError):
                            pass
                    else:
                        images = [
                            os.path.join(cluster_dir, f)
                            for f in sorted(os.listdir(cluster_dir))
                            if f.lower().endswith((".jpg", ".png"))
                        ]
                        preview_paths.extend(images)

            movie_characters[str(final_id)] = {
                "movie": movie_name,
                "movie_id": int(movie_id),
                "count": int(len(scenes)) if scenes else int(len(tracks)),
                "track_count": len(track_ids),
                "rep_image": rep_image,
                "preview_paths": preview_paths,
                "previews": preview_entries,
                "scenes": scenes,
                "embedding": centroid_vec.tolist(),
                "raw_cluster_ids": sorted(tracks["cluster_id"].astype(str).unique().tolist()),
            }

        if movie_characters:
            characters[str(int(movie_id))] = movie_characters

    merged_df = pd.concat(per_movie_records, ignore_index=True)

    if merged_parquet_path:
        os.makedirs(os.path.dirname(merged_parquet_path), exist_ok=True)
        pd.DataFrame(centroid_rows).to_parquet(merged_parquet_path, index=False)
        print(f"[Character] Saved centroid data to {merged_parquet_path}")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2, ensure_ascii=False)

    before_clusters = clusters_df["cluster_id"].nunique()
    after_clusters = sum(len(v) for v in characters.values())
    print(
        f"[Character] Clusters before merge: {before_clusters}, after per-movie merge: {after_clusters}"
    )
    print(
        f"[Character] Saved {after_clusters} character profiles grouped across {len(characters)} movies to {output_json_path}"
    )

    filter_clusters_task(merged_df, output_json_path, cfg)

    index_path = storage_cfg.get("index_path")
    if index_path:
        print(f"[Character] Building index at {index_path}...")
        build_index(output_json_path, index_path)

    print("[Character] Task completed successfully ✅")
    return output_json_path