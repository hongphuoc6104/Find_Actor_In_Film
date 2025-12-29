from __future__ import annotations
import itertools
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from prefect import task
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from utils.config_loader import load_config


# ----------------------------- Small helpers ----------------------------- #

def _score_candidate(
    silhouette: Optional[float],
    n_clusters: int,
    *,
    min_clusters: int,
    max_clusters: Optional[int],
    cluster_weight: float,
) -> float:
    """Comparable score for auto-tuning candidates."""
    if max_clusters is not None and n_clusters > max_clusters:
        return float("-inf")
    if n_clusters < min_clusters:
        return float("-inf")
    base = -1.0 if (silhouette is None or math.isnan(silhouette)) else float(silhouette)
    return base + cluster_weight * float(n_clusters)


def _auto_optimize_agglomerative(
    emb_matrix: np.ndarray,
    movie_key: Any,
    *,
    base_candidate: Dict[str, Any],
    auto_cfg: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Grid-search quanh base candidate, chọn cấu hình có score tốt nhất."""

    thresholds: Iterable[float] = auto_cfg.get("distance_thresholds") or []
    metrics: Iterable[str] = auto_cfg.get("metrics") or []
    linkages: Iterable[str] = auto_cfg.get("linkages") or []

    candidate_thresholds = set(float(t) for t in thresholds)
    candidate_metrics = set(m.lower() for m in metrics)
    candidate_linkages = set(l.lower() for l in linkages)

    candidate_thresholds.add(float(base_candidate["distance_threshold"]))
    candidate_metrics.add(base_candidate["metric"].lower())
    candidate_linkages.add(base_candidate["linkage"].lower())

    min_clusters = int(auto_cfg.get("min_clusters", 1))
    max_clusters = auto_cfg.get("max_clusters")
    if max_clusters is not None:
        max_clusters = int(max_clusters)

    cluster_weight = float(auto_cfg.get("cluster_weight", 0.0))

    candidates: List[Dict[str, Any]] = []
    base_candidate = {
        **base_candidate,
        "score": _score_candidate(
            base_candidate.get("silhouette"),
            int(base_candidate.get("num_clusters", 0)),
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            cluster_weight=cluster_weight,
        ),
        "error": None,
        "is_base": True,
    }
    candidates.append(base_candidate)

    for metric, linkage, threshold in itertools.product(
        sorted(candidate_metrics),
        sorted(candidate_linkages),
        sorted(candidate_thresholds),
    ):
        if (
            metric == base_candidate["metric"]
            and linkage == base_candidate["linkage"]
            and math.isclose(threshold, float(base_candidate["distance_threshold"]))
        ):
            continue
        if linkage == "ward" and metric != "euclidean":
            continue

        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=float(threshold),
                metric=metric,
                linkage=linkage,
            )
            labels = clusterer.fit_predict(emb_matrix)
            n_clusters = int(len(set(labels)))
            sil = None
            if n_clusters > 1:
                try:
                    sil = float(silhouette_score(emb_matrix, labels, metric=metric))
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Silhouette failed (metric={metric}, linkage={linkage}): {exc}")
                    sil = None
        except Exception as exc:  # noqa: BLE001
            candidates.append(
                {
                    "movie_key": movie_key,
                    "metric": metric,
                    "linkage": linkage,
                    "distance_threshold": float(threshold),
                    "num_clusters": None,
                    "silhouette": None,
                    "score": float("nan"),
                    "error": str(exc),
                    "labels": None,
                    "is_base": False,
                }
            )
            continue

        score = _score_candidate(
            sil,
            n_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            cluster_weight=cluster_weight,
        )

        candidates.append(
            {
                "movie_key": movie_key,
                "metric": metric,
                "linkage": linkage,
                "distance_threshold": float(threshold),
                "num_clusters": n_clusters,
                "silhouette": sil,
                "score": score,
                "error": None,
                "labels": labels,
                "is_base": False,
            }
        )

    best_candidate: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for cand in candidates:
        score = cand.get("score")
        labels = cand.get("labels")
        if labels is None:
            continue
        if score is None or math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_candidate = cand

    report_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        report_rows.append(
            {
                "movie_key": movie_key,
                "metric": cand.get("metric"),
                "linkage": cand.get("linkage"),
                "distance_threshold": cand.get("distance_threshold"),
                "num_clusters": cand.get("num_clusters"),
                "silhouette": cand.get("silhouette"),
                "score": cand.get("score"),
                "error": cand.get("error"),
                "is_base": cand.get("is_base", False),
                "chosen": cand is best_candidate,
            }
        )

    return best_candidate, report_rows


def _filter_clusters_sizewise(
    df: pd.DataFrame,
    *,
    min_track_size: int = 1,
    min_cluster_size: int = 1,
) -> pd.DataFrame:
    """Bỏ track/cụm quá nhỏ theo config."""
    original_len = len(df)

    if min_track_size > 1 and "track_size" in df.columns:
        df = df[df["track_size"] >= min_track_size]

    if min_cluster_size > 1 and "cluster_id" in df.columns:
        counts = df.groupby("cluster_id")["track_id"].count()
        valid_clusters = counts[counts >= min_cluster_size].index
        df = df[df["cluster_id"].isin(valid_clusters)]

    removed = original_len - len(df)
    if removed > 0:
        print(f"[INFO] Filtered out {removed} rows by size constraints.")
    return df.copy()


# --------------------------------- Task --------------------------------- #

@task(name="Cluster Faces Task")
def cluster_task(cfg: Dict[str, Any] | None = None) -> str:
    """
    Gom cụm embeddings (mặc định Agglomerative).
    **Bảo đảm chạy một-phim-đơn:** nếu ENV `FS_ACTIVE_MOVIE` được set, chỉ gom cụm của phim đó.
    """
    # [FIX] Nhận tham số cfg để dùng cấu hình đã được Auto-Tuning
    if cfg is None:
        cfg = load_config()

    storage_cfg = cfg["storage"]
    clustering_cfg = cfg.get("clustering", {})
    pca_cfg = cfg.get("pca", {})

    algo = (clustering_cfg.get("algo", "agglomerative") or "agglomerative").lower()
    default_metric = (clustering_cfg.get("metric", "euclidean") or "euclidean").lower()
    default_linkage = (clustering_cfg.get("linkage", "complete") or "complete").lower()

    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        print(f"[Cluster] Single-movie mode → '{active_movie}'")

    print(f"\n--- Starting Cluster Task ({algo.capitalize()}) ---")
    if pca_cfg.get("enable", False):
        embeddings_path = storage_cfg["warehouse_embeddings_pca"]
        print("[INFO] Clustering on PCA-reduced data.")
    else:
        embeddings_path = storage_cfg["warehouse_embeddings"]
        print("[INFO] Clustering on original 512-D data.")

    # Load embeddings
    df = pd.read_parquet(embeddings_path)

    # --- hard guard: one-movie only if requested
    if active_movie:
        if "movie" in df.columns:
            df = df[df["movie"] == active_movie]
        # Nếu parquet chỉ có movie_id, vẫn giữ; build_warehouse_task đã lọc theo movie.
        if df.empty:
            raise RuntimeError(f"[Cluster] No rows for movie='{active_movie}' in {embeddings_path}")

    if "track_centroid" not in df.columns:
        raise ValueError("[Cluster] Input parquet must contain column: track_centroid")

    # Chọn cột group theo phim
    if "movie_id" in df.columns:
        group_col = "movie_id"
    elif "movie" in df.columns:
        group_col = "movie"
        print("[INFO] movie_id column not found. Grouping by movie.")
    else:
        print("[WARN] No movie info found. Treating entire dataset as one movie.")
        group_col = "_movie_tmp"
        df[group_col] = 0

    # Track-level dedup
    df_tracks = df.drop_duplicates(subset=[group_col, "track_id"])
    print(f"[INFO] Đã load {len(df_tracks)} track centroids để gom cụm.")

    tuning_report_rows: List[Dict[str, Any]] = []
    results = []

    # Gom cụm theo từng phim (thường chỉ 1 nhóm khi bật single-movie)
    for movie_key, group in df_tracks.groupby(group_col):
        print(f"Processing {group_col}={movie_key}...")
        metric = default_metric
        linkage = default_linkage
        emb_matrix = np.asarray(group["track_centroid"].tolist(), dtype=np.float32)

        if len(group) > 1:
            # Xác định ngưỡng
            auto_percentile = clustering_cfg.get("auto_distance_percentile")
            if auto_percentile is not None:
                dist_matrix = pairwise_distances(emb_matrix, metric=metric)
                triu = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
                dist_th = float(np.percentile(triu, auto_percentile))
                print(f"[INFO] Inferred distance_threshold={dist_th:.4f} at p{auto_percentile}")
            else:
                if pca_cfg.get("enable", False):
                    dist_cfg = clustering_cfg.get("distance_threshold_pca",
                                                  clustering_cfg.get("distance_threshold", 0.7))
                else:
                    dist_cfg = clustering_cfg.get("distance_threshold", 0.7)
                if isinstance(dist_cfg, dict):
                    default_th = float(dist_cfg.get("default", 0.7))
                    per_movie = dist_cfg.get("per_movie", {})
                    dist_th = float(per_movie.get(str(movie_key), default_th))
                else:
                    dist_th = float(dist_cfg)
                print(f"[INFO] Using configured distance_threshold={dist_th:.4f}")

            # Baseline: Agglomerative
            aggl = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_th,
                metric=metric,
                linkage=linkage,
            )
            aggl_labels = aggl.fit_predict(emb_matrix)
            n_aggl = len(set(aggl_labels))
            sil = float(silhouette_score(emb_matrix, aggl_labels, metric=metric)) if n_aggl > 1 else -1.0
            print(f"[DEBUG] Agglomerative produced {n_aggl} clusters (silhouette={sil:.3f})")

            chosen_labels = aggl_labels
            final_clusters = n_aggl
            final_algo = "agglomerative"

            # Auto-opt (tuỳ config)
            auto_cfg = clustering_cfg.get("auto_optimize", {})
            auto_enabled = bool(auto_cfg.get("enable", False))
            min_track_count = int(auto_cfg.get("min_track_count", 2))
            if auto_enabled and len(group) >= min_track_count:
                best, report = _auto_optimize_agglomerative(
                    emb_matrix,
                    movie_key,
                    base_candidate={
                        "movie_key": movie_key,
                        "metric": metric,
                        "linkage": linkage,
                        "distance_threshold": dist_th,
                        "num_clusters": n_aggl,
                        "silhouette": float(sil),
                        "labels": aggl_labels,
                    },
                    auto_cfg=auto_cfg,
                )
                tuning_report_rows.extend(report)
                if best is not None and not best.get("is_base", False):
                    chosen_labels = best["labels"]
                    final_clusters = int(best.get("num_clusters", final_clusters))
                    metric = best.get("metric", metric)
                    linkage = best.get("linkage", linkage)
                    dist_th = float(best.get("distance_threshold", dist_th))
                    sil_disp = "nan" if best.get("silhouette") is None else f"{best['silhouette']:.3f}"
                    final_algo = "agglomerative(auto)"
                    print(f"[INFO] Auto-optimized: metric={metric}, linkage={linkage}, "
                          f"distance={dist_th:.4f} (silhouette={sil_disp})")
            elif auto_enabled:
                print(f"[INFO] Auto optimize skipped for {movie_key}: insufficient tracks (n={len(group)})")

            # HDBSCAN (tuỳ chọn)
            if algo in {"auto", "hdbscan"}:
                import hdbscan  # lazy import

                hdb_cfg = clustering_cfg.get("hdbscan", {})
                if hdb_cfg.get("use_precomputed", False):
                    dist_hdb = 1 - cosine_similarity(emb_matrix)
                    hdb = hdbscan.HDBSCAN(
                        min_cluster_size=int(hdb_cfg.get("min_cluster_size", 5)),
                        min_samples=int(hdb_cfg.get("min_samples", 5)),
                        metric="precomputed",
                    )
                    hdb_labels = hdb.fit_predict(dist_hdb)
                else:
                    hdb = hdbscan.HDBSCAN(
                        min_cluster_size=int(hdb_cfg.get("min_cluster_size", 5)),
                        min_samples=int(hdb_cfg.get("min_samples", 5)),
                        metric=metric,
                    )
                    hdb_labels = hdb.fit_predict(emb_matrix)

                n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
                outlier_ratio = float(np.mean(hdb_labels == -1))
                print(f"[DEBUG] HDBSCAN produced {n_hdb} clusters (outliers={outlier_ratio:.2%})")

                if algo == "hdbscan" or (
                    algo == "auto"
                    and (n_aggl <= 1 or sil < 0.2)
                    and n_hdb > 0
                    and outlier_ratio < 0.5
                ):
                    chosen_labels = hdb_labels
                    final_clusters = n_hdb
                    final_algo = "hdbscan"

            if algo == "auto":
                print(f"[INFO] Cluster count change: {n_aggl} -> {final_clusters} using {final_algo}")
        else:
            chosen_labels = np.array([0])
            final_clusters = 1
            final_algo = algo

        group = group.copy()
        group["cluster_id"] = [f"{movie_key}_{lbl}" for lbl in chosen_labels]
        results.append(group)

    # Kết quả & filter kích thước
    clusters_df = pd.concat(results, ignore_index=True)
    original_count = len(clusters_df)
    filter_cfg = cfg.get("filter", {})
    min_track = int(filter_cfg.get("min_track_size", 1))
    min_cluster = int(filter_cfg.get("min_cluster_size", 1))
    
    clusters_df = _filter_clusters_sizewise(
        clusters_df,
        min_track_size=min_track,
        min_cluster_size=min_cluster,
    )

    # Lưu báo cáo tuning nếu có
    tuning_report_path = cfg["storage"].get("cluster_tuning_report")
    if tuning_report_rows and tuning_report_path:
        os.makedirs(os.path.dirname(tuning_report_path), exist_ok=True)
        pd.DataFrame(tuning_report_rows).to_csv(tuning_report_path, index=False)
        print(f"[INFO] Saved tuning report to {tuning_report_path}")

    # Kiểm tra nếu không còn cluster nào sau filter
    if clusters_df.empty:
        print("\n" + "="*60)
        print("⚠️  CẢNH BÁO: Không đủ dữ liệu để tạo cluster!")
        print("="*60)
        print(f"   - Số faces trước filter: {original_count}")
        print(f"   - Yêu cầu min_track_size: {min_track}")
        print(f"   - Yêu cầu min_cluster_size: {min_cluster}")
        print("")
        print("💡 Gợi ý:")
        print("   1. Video quá ngắn - thử video dài hơn (>10 phút)")
        print("   2. Giảm min_cluster_size xuống 2-5 trong ProcessTab")
        print("   3. Chọn preset 'short' hoặc 'very_short' nếu có")
        print("="*60 + "\n")
    else:
        # Thống kê
        unique_labels, counts = np.unique(clusters_df["cluster_id"], return_counts=True)
        print("\n=== KẾT QUẢ CLUSTERING ===")
        print(f"Số cụm nhân vật tìm được: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            print(f"  → Cụm {label}: {count} khuôn mặt")
        print("==========================")

    # Lưu (dù rỗng vẫn lưu để không gây lỗi downstream)
    clusters_path = cfg["storage"]["warehouse_clusters"]
    clusters_df.to_parquet(clusters_path, index=False)
    print(f"\n[INFO] Đã lưu {len(clusters_df)} cluster records -> {clusters_path}")
    return clusters_path