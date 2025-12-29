# utils/indexer.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Helpers: paths & metadata
# ------------------------------------------------------------

PROJECT_ROOT = Path(os.getcwd())
DATA_DIR = PROJECT_ROOT / "Data"
WAREHOUSE_DIR = PROJECT_ROOT / "warehouse"
PARQUET_DIR = WAREHOUSE_DIR / "parquet"
CHARACTERS_JSON = WAREHOUSE_DIR / "characters.json"

# cache in-memory
__INDEX_MATRIX: Optional[np.ndarray] = None
__INDEX_META: Optional[List[Dict]] = None


def _read_json(p: Path) -> dict:
    if not p.exists(): return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _load_per_movie_embeddings(movie_title: str) -> Optional[pd.DataFrame]:
    """Load embeddings gốc (Data/embeddings/...)"""
    emb_path = DATA_DIR / "embeddings" / f"{movie_title}.parquet"
    if not emb_path.exists(): return None
    df = pd.read_parquet(emb_path)

    # Normalize
    if 'embedding' not in df.columns and 'emb' in df.columns:
        df.rename(columns={'emb': 'embedding'}, inplace=True)
    if "embedding" not in df.columns: return None
    if "frame" not in df.columns and "image" in df.columns:
        df['frame'] = df['image']
    return df[["frame", "embedding"]].dropna().copy()


def _load_per_movie_clusters(movie_title: str) -> Optional[pd.DataFrame]:
    """Load clusters đã xử lý (warehouse/parquet/..._clusters.parquet)"""
    # Đây là điểm mấu chốt: Load file riêng của từng phim
    cluster_path = PARQUET_DIR / f"{movie_title}_clusters.parquet"

    if not cluster_path.exists():
        # Fallback nhẹ: nếu không thấy file riêng, thử load file chung (cho backward compatible)
        # nhưng tốt nhất là nên có file riêng.
        return None

    try:
        df = pd.read_parquet(cluster_path)
    except Exception:
        return None

    # Normalize ID
    char_id_col = next((c for c in ["final_character_id", "cluster_id", "character_id"] if c in df.columns), None)
    if not char_id_col: return None
    df.rename(columns={char_id_col: "character_id"}, inplace=True)
    return df


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


# ------------------------------------------------------------
# Build index: centroid 512-D per character
# ------------------------------------------------------------

def build_character_index(force_rebuild: bool = False) -> Tuple[np.ndarray, List[Dict]]:
    global __INDEX_MATRIX, __INDEX_META
    if __INDEX_MATRIX is not None and __INDEX_META is not None and not force_rebuild:
        return __INDEX_MATRIX, __INDEX_META

    print("[Indexer] Rebuilding Search Index (Multi-Movie)...")
    char_manifest = _read_json(CHARACTERS_JSON)

    rows: List[np.ndarray] = []
    metas: List[Dict] = []
    movies_indexed = 0

    # Duyệt qua tất cả các phim đang có trong JSON
    for movie_title, chars_in_movie in char_manifest.items():
        if movie_title.startswith("_"): continue

        # 1. Load Data
        emb_df = _load_per_movie_embeddings(movie_title)
        cluster_df = _load_per_movie_clusters(movie_title)

        if emb_df is None or cluster_df is None:
            continue

        movies_indexed += 1

        # 2. Compute Centroids
        for char_id, char_data in chars_in_movie.items():
            # Lấy các frame thuộc nhân vật này trong cluster file
            frames_for_char = cluster_df[cluster_df['character_id'] == char_id]
            if frames_for_char.empty: continue

            # Join với embedding gốc
            join_df = pd.merge(frames_for_char, emb_df, on='frame', how='inner')
            if join_df.empty: continue

            # Tính vector trung bình
            vecs = np.array(join_df["embedding"].tolist(), dtype=np.float32)
            if vecs.ndim != 2: continue

            centroid = _l2_normalize(vecs.mean(axis=0, keepdims=True))[0]

            rows.append(centroid)
            metas.append({
                "character_id": str(char_id),
                "name": char_data.get("name", "Unknown"),
                "movie": movie_title,
                "rep_image": char_data.get("rep_image"),
                "preview_paths": char_data.get("preview_paths", []),
                "scenes": char_data.get("scenes", []),
            })

    if not rows:
        __INDEX_MATRIX = np.zeros((0, 512), dtype=np.float32)
        __INDEX_META = []
    else:
        __INDEX_MATRIX = _l2_normalize(np.vstack(rows).astype(np.float32))
        __INDEX_META = metas

    print(f"[Indexer] Indexed {len(metas)} characters from {movies_indexed} movies.")
    return __INDEX_MATRIX, __INDEX_META


def search_by_embedding(query_vec: np.ndarray, top_k: int = 5, min_score: float = 0.25) -> List[Dict]:
    M, metas = build_character_index()
    if M.shape[0] == 0: return []

    q = _l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
    sims = (M @ q.T).ravel()
    order = np.argsort(-sims)[: max(top_k * 2, 20)]

    results: List[Dict] = []
    for idx in order:
        score = float(sims[idx])
        if score < float(min_score): continue
        meta = metas[idx].copy()
        meta["score"] = score
        results.append(meta)

    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]


def clear_index_cache():
    global __INDEX_MATRIX, __INDEX_META
    __INDEX_MATRIX = None
    __INDEX_META = None