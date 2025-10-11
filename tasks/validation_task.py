# tasks/validation_task.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prefect import task

from utils.config_loader import load_config


# -------------------------------
# CSV helpers (atomic + append)
# -------------------------------
def _atomic_write_csv(df: pd.DataFrame, path: str, mode: str = "w") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    header = not (mode == "a" and os.path.exists(path))
    df.to_csv(tmp, index=False, header=header, mode="w", encoding="utf-8")
    if mode == "a" and os.path.exists(path):
        # append by concatenating files (cheap + safe)
        with open(path, "ab") as out, open(tmp, "rb") as src:
            if header:  # shouldn't happen in append branch
                pass
            out.write(src.read())
        os.remove(tmp)
    else:
        os.replace(tmp, path)


def _append_report(df: pd.DataFrame, path: str) -> None:
    """Append with timestamp column; create file if missing."""
    if df.empty:
        return
    mode = "a" if os.path.exists(path) else "w"
    _atomic_write_csv(df, path, mode=mode)


# -------------------------------
# Metrics computations
# -------------------------------
def _safe_uniques(s: pd.Series) -> int:
    try:
        return int(s.nunique())
    except Exception:
        return int(len(s))


def _mean_l2_to_centroid(vectors: List[np.ndarray]) -> float:
    try:
        if not vectors:
            return 0.0
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in vectors], axis=0)
        c = arr.mean(axis=0)
        d = np.linalg.norm(arr - c, axis=1)
        return float(np.mean(d))
    except Exception:
        return 0.0


def _ensure_movie_id(df: pd.DataFrame) -> pd.DataFrame:
    if "movie_id" not in df.columns:
        return df.assign(movie_id=df.get("movie", 0))
    return df


# -------------------------------
# Public task
# -------------------------------
@task(name="Validation Task")
def validation_task() -> Dict[str, str]:
    """
    Sinh báo cáo chất lượng:
      - cluster_metrics.csv: theo (movie_id, final_character_id)
      - track_quality.csv: theo (movie_id, track_id)
    Ghi append (thêm cột timestamp) để theo dõi theo thời gian.
    """
    print("\n--- Running Validation Task ---")
    cfg = load_config()
    storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}

    # Đường vào/ra
    clusters_path = storage.get("warehouse_clusters", "warehouse/parquet/clusters.parquet")
    embeds_path = storage.get("warehouse_embeddings", "warehouse/parquet/embeddings.parquet")

    # Đường ra báo cáo (định nghĩa mềm: dùng 'reports/' mặc định)
    reports_dir = "reports"
    cluster_metrics_csv = os.path.join(reports_dir, "cluster_metrics.csv")
    track_quality_csv = os.path.join(reports_dir, "track_quality.csv")

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------- Load data -----------------
    if not os.path.exists(clusters_path):
        print(f"[Validation] Missing clusters parquet: {clusters_path}")
        return {"cluster_metrics": cluster_metrics_csv, "track_quality": track_quality_csv}

    try:
        clu = pd.read_parquet(clusters_path)
    except Exception as e:
        print(f"[Validation] Failed to read clusters parquet: {e}")
        return {"cluster_metrics": cluster_metrics_csv, "track_quality": track_quality_csv}

    if clu.empty:
        print("[Validation] Clusters parquet is empty.")
        return {"cluster_metrics": cluster_metrics_csv, "track_quality": track_quality_csv}

    clu = _ensure_movie_id(clu)

    # ----------------- Cluster metrics -----------------
    # Ưu tiên final_character_id; fallback cluster_id
    id_col = "final_character_id" if "final_character_id" in clu.columns else "cluster_id"
    keep_cols = [c for c in [id_col, "movie_id", "frame", "det_score", "track_centroid"] if c in clu.columns]
    gdf = clu[keep_cols].copy()

    # group theo (movie_id, id)
    rows: List[Dict[str, object]] = []
    for (movie_id, cid), g in gdf.groupby(["movie_id", id_col], dropna=False):
        size = int(len(g))
        frames = _safe_uniques(g["frame"]) if "frame" in g.columns else size
        mean_det = float(pd.to_numeric(g["det_score"], errors="coerce").mean()) if "det_score" in g.columns else 0.0
        # compactness: mean L2 distance đến centroid
        mean_l2 = 0.0
        if "track_centroid" in g.columns:
            vecs = []
            for v in g["track_centroid"].tolist():
                try:
                    vecs.append(np.asarray(v, dtype=np.float32))
                except Exception:
                    pass
            mean_l2 = _mean_l2_to_centroid(vecs)

        rows.append(
            {
                "timestamp": ts,
                "movie_id": str(movie_id),
                "character_id": str(cid),
                "cluster_size": size,
                "unique_frames": frames,
                "mean_det": round(mean_det, 6),
                "mean_l2_to_centroid": round(mean_l2, 6),
            }
        )

    cluster_metrics = pd.DataFrame(rows, columns=[
        "timestamp", "movie_id", "character_id",
        "cluster_size", "unique_frames", "mean_det", "mean_l2_to_centroid"
    ])
    _append_report(cluster_metrics, cluster_metrics_csv)
    print(f"[Validation] Wrote/append cluster metrics → {cluster_metrics_csv}")

    # ----------------- Track quality -----------------
    # Từ embeddings parquet: (movie_id, track_id) → length, frames, mean_det (nếu có)
    if os.path.exists(embeds_path):
        try:
            emb = pd.read_parquet(embeds_path)
            emb = _ensure_movie_id(emb)
        except Exception as e:
            print(f"[Validation] Failed to read embeddings parquet: {e}")
            emb = pd.DataFrame()
    else:
        emb = pd.DataFrame()

    rows2: List[Dict[str, object]] = []
    if not emb.empty and "track_id" in emb.columns:
        base = emb[["movie_id", "track_id"] + ([ "frame" ] if "frame" in emb.columns else []) + ([ "det_score" ] if "det_score" in emb.columns else [])].copy()
        for (movie_id, tid), g in base.groupby(["movie_id", "track_id"], dropna=False):
            length = int(len(g))
            frames = _safe_uniques(g["frame"]) if "frame" in g.columns else length
            mean_det = float(pd.to_numeric(g["det_score"], errors="coerce").mean()) if "det_score" in g.columns else 0.0
            rows2.append(
                {
                    "timestamp": ts,
                    "movie_id": str(movie_id),
                    "track_id": str(tid),
                    "track_len": length,
                    "unique_frames": frames,
                    "mean_det": round(mean_det, 6),
                }
            )
    else:
        print("[Validation] No track_id column in embeddings parquet; skip track_quality report.")

    track_quality = pd.DataFrame(rows2, columns=[
        "timestamp", "movie_id", "track_id", "track_len", "unique_frames", "mean_det"
    ])
    if not track_quality.empty:
        _append_report(track_quality, track_quality_csv)
        print(f"[Validation] Wrote/append track quality → {track_quality_csv}")

    return {"cluster_metrics": cluster_metrics_csv, "track_quality": track_quality_csv}
