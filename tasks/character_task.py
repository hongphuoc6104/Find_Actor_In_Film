from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from prefect import task
from sklearn.cluster import AgglomerativeClustering

from utils.config_loader import load_config
from utils.indexer import build_index
from utils.vector_utils import _mean_vector, l2_normalize
from tasks.filter_clusters_task import filter_clusters_task


DEFAULT_CLIP_FPS = 8.0


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


def _safe_slug(value: Any, fallback: str = "scene") -> str:
    """Return a filesystem-friendly slug for ``value``."""

    if value is None:
        text = fallback
    else:
        try:
            if isinstance(value, (int, np.integer)):
                text = str(int(value))
            else:
                text = str(value)
        except Exception:
            text = fallback

    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    sanitized = sanitized.strip("_") or fallback
    return sanitized[:80]


def _prepare_track_timeline(
    track_df: pd.DataFrame,
    frames_dir: str | None,
    fps: float | None,
    track_id: Any,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]]]:
    """Create timeline entries and frame records for a given track."""

    timeline: List[Dict[str, Any]] = []
    frame_records: List[Tuple[int, str]] = []

    base_dir = frames_dir if frames_dir and os.path.isdir(frames_dir) else None

    for order_idx, row in enumerate(track_df.itertuples()):
        frame_name = getattr(row, "frame", None)
        if not isinstance(frame_name, str) or not frame_name:
            continue

        raw_idx = getattr(row, "frame_index", None)
        frame_idx = None
        if raw_idx is not None and raw_idx == raw_idx:
            try:
                frame_idx = int(raw_idx)
            except (TypeError, ValueError):
                frame_idx = _frame_to_int(frame_name)
        else:
            frame_idx = _frame_to_int(frame_name)

        timestamp = _timestamp_from_frame(frame_idx, fps)
        bbox = _normalize_bbox(getattr(row, "bbox", [])) if hasattr(row, "bbox") else []
        det_score = getattr(row, "det_score", None)
        det_score_val = None
        if det_score is not None and det_score == det_score:
            try:
                det_score_val = float(det_score)
            except (TypeError, ValueError):
                det_score_val = None

        entry = {
            "order": order_idx,
            "track_id": int(track_id) if track_id == track_id else None,
            "frame": frame_name,
            "frame_index": frame_idx if frame_idx is not None and frame_idx >= 0 else None,
            "timestamp": timestamp,
            "bbox": bbox,
        }
        if det_score_val is not None:
            entry["det_score"] = det_score_val

        entry_index = len(timeline)
        timeline.append(entry)

        if base_dir:
            frame_path = os.path.join(base_dir, frame_name)
            if os.path.exists(frame_path):
                frame_records.append((entry_index, frame_path))

    return timeline, frame_records


def _export_track_clip(
    frame_records: List[Tuple[int, str]],
    clips_root: str | None,
    movie_name: str,
    character_id: str,
    track_id: Any,
    clip_fps: float | None,
) -> Tuple[str | None, int | None, int | None, float | None, List[int]]:
    """Persist a short MP4 clip for the provided frames."""

    if not clips_root or not frame_records:
        return None, None, None, None, []

    clips_root_abs = os.path.abspath(clips_root)
    os.makedirs(clips_root_abs, exist_ok=True)

    movie_slug = _safe_slug(movie_name, "movie")
    char_slug = _safe_slug(character_id, "character")
    track_slug = _safe_slug(track_id, "track") if track_id is not None else "track"

    clip_dir = os.path.join(clips_root_abs, movie_slug, char_slug)
    os.makedirs(clip_dir, exist_ok=True)

    filename = f"{movie_slug}_{char_slug}_{track_slug}.mp4"
    output_path = os.path.join(clip_dir, filename)

    fps_value = clip_fps if clip_fps and clip_fps > 0 else DEFAULT_CLIP_FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = None
    width = None
    height = None
    used_indices: List[int] = []

    try:
        for entry_index, frame_path in frame_records:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            if width is None or height is None:
                height, width = frame.shape[:2]
                temp_writer = cv2.VideoWriter(
                    output_path, fourcc, fps_value, (int(width), int(height))
                )
                if not temp_writer.isOpened():
                    temp_writer.release()
                    writer = None
                    break
                writer = temp_writer
            elif writer is not None:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (int(width), int(height)))
            else:
                break

            if writer is None:
                break

            writer.write(frame)
            used_indices.append(entry_index)
    finally:
        if writer is not None:
            writer.release()

    if not used_indices or width is None or height is None:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return None, None, None, None, []

    rel_path = os.path.relpath(output_path, clips_root_abs)
    return rel_path.replace(os.sep, "/"), int(width), int(height), float(fps_value), used_indices


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
    frames_root = storage_cfg.get("frames_root")
    clips_root = storage_cfg.get("scene_clips_root")
    clips_root_abs = os.path.abspath(clips_root) if clips_root else None
    if clips_root_abs:
        os.makedirs(clips_root_abs, exist_ok=True)

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
            scene_frame_count = 0

            movie_frames_dir = None
            if frames_root:
                potential_dir = os.path.join(frames_root, movie_name)
                if os.path.isdir(potential_dir):
                    movie_frames_dir = potential_dir

            if not movie_embeddings.empty and track_ids:
                scene_rows = movie_embeddings["track_id"].isin(track_ids)
                scene_df = movie_embeddings[scene_rows].copy()
                if not scene_df.empty:
                    if "frame_index" not in scene_df.columns:
                        scene_df["frame_index"] = scene_df["frame"].apply(_frame_to_int)
                    scene_df.sort_values(["track_id", "frame_index"], inplace=True)

                    grouped_tracks: List[Tuple[float, Any, pd.DataFrame]] = []
                    for track_key, track_df in scene_df.groupby("track_id"):
                        track_df = track_df.sort_values("frame_index")
                        first_idx = None
                        if "frame_index" in track_df.columns:
                            first_val = track_df["frame_index"].dropna().min()
                            if first_val == first_val:
                                try:
                                    first_idx = float(first_val)
                                except (TypeError, ValueError):
                                    first_idx = None
                        order_value = first_idx if first_idx is not None else float("inf")
                        grouped_tracks.append((order_value, track_key, track_df))

                    grouped_tracks.sort(
                        key=lambda item: (
                            item[0],
                            _safe_slug(item[1], "track"),
                        )
                    )

                    for order_idx, (_, track_key, track_df) in enumerate(grouped_tracks):
                        timeline_entries, frame_records = _prepare_track_timeline(
                            track_df,
                            movie_frames_dir,
                            fps,
                            track_key,
                        )
                        if not timeline_entries:
                            continue

                        clip_fps_value = float(fps) if fps else DEFAULT_CLIP_FPS
                        clip_rel_path = None
                        clip_width = None
                        clip_height = None
                        used_entry_indices: List[int] = []

                        if clips_root_abs and frame_records:
                            (
                                clip_rel_path,
                                clip_width,
                                clip_height,
                                clip_fps_result,
                                used_entry_indices,
                            ) = _export_track_clip(
                                frame_records,
                                clips_root_abs,
                                movie_name,
                                str(final_id),
                                track_key,
                                clip_fps_value,
                            )
                            if clip_fps_result:
                                clip_fps_value = clip_fps_result

                        if clip_rel_path and used_entry_indices:
                            unique_indices = sorted(dict.fromkeys(used_entry_indices))
                            timeline_to_store = [
                                timeline_entries[idx]
                                for idx in unique_indices
                                if 0 <= idx < len(timeline_entries)
                            ]
                        else:
                            timeline_to_store = timeline_entries

                        if not timeline_to_store:
                            continue

                        if clip_width is None or clip_height is None:
                            sample_path = None
                            if frame_records:
                                sample_path = frame_records[0][1]
                            if (
                                sample_path
                                and os.path.exists(sample_path)
                            ):
                                sample_img = cv2.imread(sample_path)
                                if sample_img is not None:
                                    clip_height, clip_width = sample_img.shape[:2]

                        for seq_idx, entry in enumerate(timeline_to_store):
                            entry["order"] = seq_idx
                            if clip_rel_path:
                                entry["clip_offset"] = round(
                                    seq_idx / clip_fps_value, 3
                                )

                        scene_frame_count += len(timeline_to_store)

                        first_entry = timeline_to_store[0]
                        last_entry = timeline_to_store[-1]

                        try:
                            track_int = int(track_key)
                        except (TypeError, ValueError):
                            track_int = None

                        scene_entry = {
                            "order": order_idx,
                            "track_id": track_int,
                            "frame": first_entry.get("frame"),
                            "frame_index": first_entry.get("frame_index"),
                            "timestamp": first_entry.get("timestamp"),
                            "bbox": first_entry.get("bbox"),
                            "det_score": first_entry.get("det_score"),
                            "timeline": timeline_to_store,
                            "frame_count": len(timeline_to_store),
                            "clip_path": clip_rel_path,
                            "clip_fps": clip_fps_value if clip_rel_path else None,
                            "duration": round(
                                len(timeline_to_store) / clip_fps_value, 3
                            )
                            if clip_rel_path
                            else None,
                            "end_frame": last_entry.get("frame"),
                            "end_frame_index": last_entry.get("frame_index"),
                            "end_timestamp": last_entry.get("timestamp"),
                            "width": clip_width,
                            "height": clip_height,
                        }

                        scenes.append(scene_entry)

                    rep_idx = (
                        scene_df["det_score"].idxmax()
                        if "det_score" in scene_df
                        else scene_df.index[0]
                    )
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
                "count": int(scene_frame_count)
                if scene_frame_count
                else int(len(track_ids)),
                "track_count": len(track_ids),
                "rep_image": rep_image,
                "preview_paths": preview_paths,
                "previews": preview_entries,
                "scenes": scenes,
                "embedding": centroid_vec.tolist(),
                "raw_cluster_ids": sorted(
                    tracks["cluster_id"].astype(str).unique().tolist()
                ),
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
