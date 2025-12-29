# tasks/validation_task.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from prefect import task

from utils.config_loader import load_config


# =====================================================================
# CÁC HÀM HELPER (Không thay đổi)
# =====================================================================

def _atomic_write_csv(df: pd.DataFrame, path: str, mode: str = "w") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    header = not (mode == "a" and os.path.exists(path))
    df.to_csv(tmp, index=False, header=header, mode="w", encoding="utf-8")
    if mode == "a" and os.path.exists(path):
        with open(path, "ab") as out, open(tmp, "rb") as src:
            out.write(src.read())
        os.remove(tmp)
    else:
        os.replace(tmp, path)


def _append_report(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    mode = "a" if os.path.exists(path) else "w"
    _atomic_write_csv(df, path, mode=mode)


def _safe_uniques(s: pd.Series) -> int:
    try:
        return int(s.nunique())
    except Exception:
        return int(len(s))


def _mean_l2_to_centroid(vectors: List[np.ndarray]) -> float:
    try:
        if not vectors:
            return 0.0
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in vectors if v is not None], axis=0)
        if arr.size == 0:
            return 0.0
        c = arr.mean(axis=0)
        d = np.linalg.norm(arr - c, axis=1)
        return float(np.mean(d))
    except Exception:
        return 0.0


def _ensure_movie_id(df: pd.DataFrame) -> pd.DataFrame:
    if "movie_id" not in df.columns:
        return df.assign(movie_id=df.get("movie", 0))
    return df


# =====================================================================
# TASK CHÍNH (Đã sửa lỗi đồng bộ)
# =====================================================================

@task(name="Validation Task")
def validation_task(cfg: dict | None = None) -> Dict[str, str]:
    """
    Sinh báo cáo chất lượng:
      - cluster_metrics.csv: theo (movie_id, final_character_id)
      - track_quality.csv: theo (movie_id, track_id)
    Sử dụng config được truyền vào để đảm bảo đồng bộ.
    """
    print("\n--- Running Validation Task ---")
    # CẬP NHẬT 1: Ưu tiên cfg được truyền vào
    if cfg is None:
        print("[Validation][WARN] Config không được truyền vào, đang tự đọc lại file config tĩnh.")
        cfg = load_config()

    storage = cfg.get("storage", {})

    # CẬP NHẬT 2: Đọc đường dẫn từ config đã được đồng bộ
    clusters_path = storage.get("clusters_merged_parquet")  # Ưu tiên file đã merge
    if not clusters_path or not os.path.exists(clusters_path):
        clusters_path = storage.get("warehouse_clusters", "warehouse/parquet/clusters.parquet")

    embeds_path = storage.get("warehouse_embeddings", "warehouse/parquet/embeddings.parquet")

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    cluster_metrics_csv = os.path.join(reports_dir, "cluster_metrics.csv")
    track_quality_csv = os.path.join(reports_dir, "track_quality.csv")

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # --- Logic tính toán còn lại giữ nguyên ---
    if not os.path.exists(clusters_path):
        print(f"[Validation] Missing final clusters parquet: {clusters_path}")
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

    id_col = "final_character_id" if "final_character_id" in clu.columns else "cluster_id"
    keep_cols = [c for c in [id_col, "movie_id", "frame", "det_score", "track_centroid"] if c in clu.columns]
    gdf = clu[keep_cols].copy()

    rows: List[Dict[str, object]] = []
    for (movie_id, cid), g in gdf.groupby(["movie_id", id_col], dropna=False):
        size = int(len(g))
        frames = _safe_uniques(g["frame"]) if "frame" in g.columns else size
        mean_det = float(pd.to_numeric(g["det_score"], errors="coerce").mean()) if "det_score" in g.columns else 0.0
        mean_l2 = 0.0
        if "track_centroid" in g.columns:
            vecs = [np.asarray(v, dtype=np.float32) for v in g["track_centroid"].tolist() if v is not None]
            if vecs:
                mean_l2 = _mean_l2_to_centroid(vecs)

        rows.append({
            "timestamp": ts, "movie_id": str(movie_id), "character_id": str(cid),
            "cluster_size": size, "unique_frames": frames, "mean_det": round(mean_det, 6),
            "mean_l2_to_centroid": round(mean_l2, 6),
        })

    cluster_metrics = pd.DataFrame(rows)
    _append_report(cluster_metrics, cluster_metrics_csv)
    print(f"[Validation] Wrote/append cluster metrics → {cluster_metrics_csv}")

    try:
        emb = pd.read_parquet(embeds_path)
        emb = _ensure_movie_id(emb)
    except Exception:
        emb = pd.DataFrame()

    rows2: List[Dict[str, object]] = []
    if not emb.empty and "track_id" in emb.columns:
        base_cols = ["movie_id", "track_id"]
        if "frame" in emb.columns: base_cols.append("frame")
        if "det_score" in emb.columns: base_cols.append("det_score")
        base = emb[base_cols].copy()

        for (movie_id, tid), g in base.groupby(["movie_id", "track_id"], dropna=False):
            length = int(len(g))
            frames = _safe_uniques(g["frame"]) if "frame" in g.columns else length
            mean_det = float(pd.to_numeric(g["det_score"], errors="coerce").mean()) if "det_score" in g.columns else 0.0
            rows2.append({
                "timestamp": ts, "movie_id": str(movie_id), "track_id": str(tid),
                "track_len": length, "unique_frames": frames, "mean_det": round(mean_det, 6),
            })
    else:
        print("[Validation] No track_id column in embeddings parquet; skip track_quality report.")

    track_quality = pd.DataFrame(rows2)
    if not track_quality.empty:
        _append_report(track_quality, track_quality_csv)
        print(f"[Validation] Wrote/append track quality → {track_quality_csv}")

    return {"cluster_metrics": cluster_metrics_csv, "track_quality": track_quality_csv}