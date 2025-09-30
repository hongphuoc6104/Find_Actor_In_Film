"""Utilities for searching characters using face embeddings."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from utils.config_loader import load_config
from utils.indexer import load_index
from utils.vector_utils import l2_normalize


def _query_index(index: Any, emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Query a FAISS or Annoy index."""
    if hasattr(index, "search"):
        distances, indices = index.search(emb, k)
        return distances[0], indices[0]
    if hasattr(index, "get_nns_by_vector"):
        indices, distances = index.get_nns_by_vector(
            emb[0], k, include_distances=True
        )
        return np.array(distances), np.array(indices)
    raise TypeError("Unsupported index type")


def search_actor(
    image_path: str,
    k: int | None = 5,
    min_count: int = 0,
    return_emb: bool = False,
    *,
    score_floor: float | None = None,
    max_results: int | None = None,
) -> Union[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Find the closest characters in the index based on an image.

    ``score_floor`` and ``max_results`` allow callers to request a wider
    neighbourhood from the vector index so that all matches above the desired
    similarity threshold can be considered.
    """
    cfg = load_config()
    emb_cfg = cfg["embedding"]
    storage_cfg = cfg["storage"]
    pca_cfg = cfg.get("pca", {})
    index_cfg = cfg.get("index", {})
    search_cfg = cfg.get("search", {})

    index, id_map = load_index()
    index_type = str(index_cfg.get("type", "")).lower()
    is_similarity_index = "ip" in index_type or "cos" in index_type

    default_floor = float(search_cfg.get("min_score", 0.0))
    floor_threshold = default_floor if score_floor is None else float(score_floor)

    default_max_results = int(
        search_cfg.get("max_results", search_cfg.get("top_k", 0)) or 0
    )
    if k is not None:
        default_max_results = max(default_max_results, int(k))
    if max_results is None:
        base_query_limit = default_max_results if default_max_results > 0 else 50
    else:
        base_query_limit = int(max_results)
    base_query_limit = max(1, base_query_limit)

    app = FaceAnalysis(name=emb_cfg["model"], providers=emb_cfg["providers"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        return {} if return_emb else []

    faces = app.get(img)
    if not faces:
        return {} if return_emb else []

    # lấy mặt có det_score cao nhất
    faces.sort(key=lambda f: f.det_score, reverse=True)
    emb = faces[0].embedding.astype("float32")

    # normalize trước PCA (embedding gốc thường đã chuẩn)
    if emb_cfg.get("l2_normalize", True):
        emb = l2_normalize(emb)

    # nếu index đang ở chiều khác, cố gắng áp PCA để khớp
    index_dim = getattr(index, "d", emb.shape[0])  # FAISS có .d; Annoy không có
    if pca_cfg.get("enable", False) or emb.shape[0] != index_dim:
        try:
            from joblib import load
            pca_path = storage_cfg.get("pca_model", "models/pca_model.joblib")
            if os.path.exists(pca_path):
                pca = load(pca_path)
                emb = pca.transform(emb.reshape(1, -1)).astype("float32")[0]
        except Exception:
            # nếu có lỗi khi load/transform PCA, giữ nguyên emb
            pass

    # sau PCA, norm bị thay đổi -> normalize lại nếu dùng IP/cosine
    if "ip" in index_type:
        emb = l2_normalize(emb)

    emb = emb.reshape(1, -1).astype("float32")

    with open(storage_cfg["characters_json"], "r", encoding="utf-8") as f:
        characters = json.load(f)

    def _search_func(
        query_emb: np.ndarray,
        top_k: int | None = k,
        min_count: int = min_count,
        *,
        score_floor: float | None = None,
        max_results: int | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search the loaded index using the provided embedding."""
        q = np.asarray(query_emb, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]

        # đảm bảo normalize khi index là inner-product (cosine)
        if "ip" in index_type:
            # normalize theo hàng
            norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
            q = q / norms

        effective_floor = (
            float(floor_threshold)
            if score_floor is None
            else float(score_floor)
        )

        if max_results is None:
            effective_limit = base_query_limit
        else:
            effective_limit = max(1, int(max_results))
        if top_k is not None:
            effective_limit = max(effective_limit, int(top_k))

        distances, indices = _query_index(index, q, effective_limit)

        per_movie: Dict[str, List[Dict[str, Any]]] = {}
        previews_root = storage_cfg.get("cluster_previews_root", "")

        for dist, idx in zip(distances, indices):
            score = float(dist)
            threshold_active = (
                score_floor is not None
                or floor_threshold not in (0.0, float("-inf"))
            )
            if threshold_active:
                if is_similarity_index:
                    if score < effective_floor:
                        continue
                else:
                    if score > effective_floor:
                        continue

            meta = id_map.get(int(idx))
            if meta is None:
                continue
            if isinstance(meta, dict):
                movie_id = str(meta.get("movie_id"))
                character_id = str(meta.get("character_id"))
            else:
                movie_id = "0"
                character_id = str(meta)

            movie_data = characters.get(movie_id, {})
            if isinstance(movie_data, dict) and character_id in movie_data:
                char_info = movie_data.get(character_id)
            else:
                char_info = characters.get(character_id) if movie_id == "0" else None
            if not char_info:
                continue

            count = int(char_info.get("count", 0))
            if count < min_count:
                continue

            preview_paths = char_info.get("preview_paths", [])
            normalized_previews = [
                p if os.path.isabs(p) else os.path.join(previews_root, p)
                for p in preview_paths
            ]
            result = {
                "movie_id": movie_id,
                "movie": char_info.get("movie"),
                "character_id": character_id,
                "distance": score,
                "count": count,
                "track_count": int(char_info.get("track_count", count)),
                "rep_image": char_info.get("rep_image", {}),
                "preview_paths": normalized_previews,
                "previews": char_info.get("previews", []),
                "scenes": char_info.get("scenes", []),
                "raw_cluster_ids": char_info.get("raw_cluster_ids", []),
                "movies": [char_info.get("movie")] if char_info.get("movie") else [],
            }

            per_movie.setdefault(movie_id, []).append(result)

        for movie_results in per_movie.values():
            # Ensure the best match is always first regardless of index metric:
            # similarity indexes prefer higher scores, distance indexes prefer
            # lower scores.
            movie_results.sort(
                key=lambda item: item.get("distance", 0.0),
                reverse=is_similarity_index,
            )
            if top_k is not None and len(movie_results) > int(top_k):
                del movie_results[int(top_k) :]

        return per_movie

    floor_argument = (
        floor_threshold
        if (score_floor is not None or floor_threshold not in (0.0, float("-inf")))
        else None
    )

    if return_emb:
        return {
            "embedding": emb[0].tolist(),
            "search_func": _search_func,
        }

    return _search_func(
        emb,
        min_count=min_count,
        score_floor=floor_argument,
        max_results=base_query_limit,
    )
