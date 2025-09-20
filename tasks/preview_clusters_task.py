import json
import os
import shutil
from typing import Optional

import cv2
import pandas as pd
from prefect import task

from utils.config_loader import load_config


def _frame_to_int(frame_name: str) -> int:
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


@task(name="Preview Clusters Task")
def preview_clusters_task(max_images_per_cluster: Optional[int] = 3):
    """Tạo thư mục preview cho từng cụm ảnh.

    Đọc file ``clusters.parquet`` và với mỗi ``cluster_id`` tạo một thư mục
    ``cluster_{id}`` nằm trong ``storage.cluster_previews_root``. Trong mỗi thư
    mục con lưu tối đa ``max_images_per_cluster`` khung hình đại diện. Mỗi khung
    hình được lưu dưới dạng ảnh gốc và ảnh có vẽ bbox.

    Args:
        max_images_per_cluster: Số khung hình đại diện sẽ lưu cho mỗi cụm. Mặc
            định là 3.
    Returns:
        Đường dẫn gốc nơi chứa các thư mục preview.
    """
    print("\n--- Starting Preview Clusters Task ---")
    cfg = load_config()
    storage_cfg = cfg["storage"]
    clusters_path = storage_cfg["warehouse_clusters"]
    previews_root = storage_cfg["cluster_previews_root"]
    frames_root = storage_cfg["frames_root"]

    os.makedirs(previews_root, exist_ok=True)
    df = pd.read_parquet(clusters_path)

    fps_by_movie = {}
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
            print(f"[WARN] Unable to parse metadata file: {metadata_path}")

    for cluster_id, group in df.groupby("cluster_id"):
        cluster_dir = os.path.join(previews_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        subset = (
            group.sort_values("det_score", ascending=False)
            .drop_duplicates(subset=["frame"])
            .head(max_images_per_cluster)
        )
        metadata_entries = []
        movie_name = None
        if "movie" in group.columns:
            movie_values = group["movie"].dropna().astype(str)
            if not movie_values.empty:
                movie_name = movie_values.iloc[0]
        fps = fps_by_movie.get(movie_name)
        for idx, row in subset.iterrows():
            frame_path = os.path.join(frames_root, row["movie"], row["frame"])
            if not os.path.exists(frame_path):
                continue

            base_name = f"{idx:02d}_{os.path.splitext(row['frame'])[0]}"
            orig_dst = os.path.join(cluster_dir, f"{base_name}.jpg")
            bbox_dst = os.path.join(cluster_dir, f"{base_name}_bbox.jpg")

            if not os.path.exists(orig_dst):
                try:
                    os.symlink(os.path.abspath(frame_path), orig_dst)
                except OSError:
                    try:
                        shutil.copy(frame_path, orig_dst)
                    except Exception as e:  # noqa: BLE001
                        print(f"[WARN] Could not copy {frame_path} -> {orig_dst}: {e}")

            img = cv2.imread(frame_path)
            if img is None:
                continue
            bbox = row.get("bbox")
            if bbox is None:
                continue
            if not isinstance(bbox, (list, tuple)):
                bbox = bbox.tolist()
            x1, y1, x2, y2 = map(int, bbox)
            annotated = img.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(bbox_dst, annotated)

            frame_idx = _frame_to_int(row.get("frame"))
            timestamp = _timestamp_from_frame(frame_idx, fps)
            metadata_entries.append(
                {
                    "order": len(metadata_entries),
                    "movie": row.get("movie"),
                    "movie_id": int(row.get("movie_id", 0)) if not pd.isna(row.get("movie_id", 0)) else None,
                    "frame": row.get("frame"),
                    "frame_index": frame_idx if frame_idx >= 0 else None,
                    "timestamp": timestamp,
                    "bbox": [x1, y1, x2, y2],
                    "det_score": float(row.get("det_score", 0.0)),
                    "track_id": int(row.get("track_id")) if not pd.isna(row.get("track_id", float("nan"))) else None,
                    "preview_image": os.path.abspath(orig_dst),
                    "annotated_image": os.path.abspath(bbox_dst),
                }
            )

        if metadata_entries:
            meta_path = os.path.join(cluster_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata_entries, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Cluster previews generated at {previews_root}")
    return previews_root