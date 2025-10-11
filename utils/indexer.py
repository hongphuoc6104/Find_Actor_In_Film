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

METADATA_JSON = DATA_DIR / "metadata.json"
CHARACTERS_JSON = WAREHOUSE_DIR / "characters.json"
CLUSTERS_PARQUET = PARQUET_DIR / "clusters.parquet"      # per-detection with cluster keys
MERGED_PARQUET = PARQUET_DIR / "clusters_merged.parquet" # optional; same schema

# cache in-memory
__INDEX_MATRIX: Optional[np.ndarray] = None
__INDEX_META: Optional[List[Dict]] = None


def _read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _reverse_movie_id_map(meta: dict) -> Dict[str, str]:
    """return map: movie_id(str) -> title(str)"""
    gen = meta.get("_generated") or {}
    id_map = gen.get("movie_id_map") or {}
    rev: Dict[str, str] = {}
    for title, mid in id_map.items():
        try:
            rev[str(int(mid))] = str(title)
        except Exception:
            pass
    return rev


def _load_per_movie_embeddings(movie_title: str) -> pd.DataFrame:
    """
    Read 512-D embeddings from Data/embeddings/<title>.parquet.
    Expected columns at least: frame(int), embedding(list[float]).
    """
    emb_path = DATA_DIR / "embeddings" / f"{movie_title}.parquet"
    if not emb_path.exists():
        # Some older runs store under metadata key
        meta = _read_json(METADATA_JSON)
        p = (meta.get(movie_title) or {}).get("embedding_file_path")
        if p:
            emb_path = Path(p)
            if not emb_path.is_absolute():
                emb_path = PROJECT_ROOT / p

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found for movie '{movie_title}' at {emb_path}")

    df = pd.read_parquet(emb_path)
    # Normalize column names
    if "frame" not in df.columns:
        # try to recover from 'fr' or 'image' patterns
        if "fr" in df.columns:
            df = df.rename(columns={"fr": "frame"})
        elif "image" in df.columns:
            # try to extract numeric frame if stored as 'frame_0000123.jpg'
            def _to_frame(x):
                s = str(x)
                m = [c for c in s if c.isdigit()]
                return int("".join(m)) if m else None
            df["frame"] = df["image"].map(_to_frame)
        else:
            raise ValueError("Per-movie embeddings is missing 'frame' column.")

    # ensure embedding is list-like and 512-D
    if "embedding" not in df.columns:
        raise ValueError("Per-movie embeddings is missing 'embedding' column.")
    # filter valid rows
    df = df[df["embedding"].map(lambda v: isinstance(v, (list, tuple)) and len(v) >= 128)]
    return df[["frame", "embedding"]].copy()


def _load_clusters_df() -> pd.DataFrame:
    """
    Load clusters assignment (per detection). We need (movie_id, cluster_key, frame).
    """
    p = CLUSTERS_PARQUET if CLUSTERS_PARQUET.exists() else MERGED_PARQUET
    if not p or not Path(p).exists():
        raise FileNotFoundError("clusters.parquet (or clusters_merged.parquet) does not exist.")

    df = pd.read_parquet(p)

    # try to normalize columns
    # cluster key format expected like "2_15"
    if "cluster_key" not in df.columns:
        # attempt build from movie_id + cluster_id
        if "movie_id" in df.columns and "cluster_id" in df.columns:
            df["cluster_key"] = df["movie_id"].astype(str) + "_" + df["cluster_id"].astype(str)
        elif "cluster" in df.columns:
            df["cluster_key"] = df["cluster"].astype(str)
        else:
            raise ValueError("clusters parquet missing 'cluster_key' or ('movie_id','cluster_id').")

    # frame column
    if "frame" not in df.columns:
        # try common alternatives
        for c in ["fr", "image_id", "img_idx", "frame_idx"]:
            if c in df.columns:
                df = df.rename(columns={c: "frame"})
                break
    if "frame" not in df.columns:
        # try to parse from filename if available
        if "image" in df.columns:
            def _to_frame(x):
                s = str(x)
                num = "".join([ch for ch in s if ch.isdigit()])
                return int(num) if num else None
            df["frame"] = df["image"].map(_to_frame)
        else:
            raise ValueError("clusters parquet missing 'frame' information.")

    # movie_id as str for mapping
    if "movie_id" in df.columns:
        df["movie_id"] = df["movie_id"].astype(str)
    else:
        # try infer from cluster_key prefix
        df["movie_id"] = df["cluster_key"].astype(str).str.split("_").str[0]

    return df[["movie_id", "cluster_key", "frame"]].copy()


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


# ------------------------------------------------------------
# Build index: centroid 512-D per character (cluster_key)
# ------------------------------------------------------------

def build_character_index(force_rebuild: bool = False) -> Tuple[np.ndarray, List[Dict]]:
    """
    Return (matrix, meta_list)
    - matrix: shape (N_char, 512), L2-normalized
    - meta_list: [{cluster_key, movie_id, movie, rep_image, preview_paths, scenes}, ...]
    """
    global __INDEX_MATRIX, __INDEX_META
    if __INDEX_MATRIX is not None and __INDEX_META is not None and not force_rebuild:
        return __INDEX_MATRIX, __INDEX_META

    meta = _read_json(METADATA_JSON)
    char_manifest = _read_json(CHARACTERS_JSON)
    rev_id_map = _reverse_movie_id_map(meta)

    clusters_df = _load_clusters_df()

    # Prepare per-movie embedding tables cached
    per_movie_cache: Dict[str, pd.DataFrame] = {}  # title -> df(frame, embedding)

    rows: List[np.ndarray] = []
    metas: List[Dict] = []

    # If we have characters.json we iterate by characters there (stable & has previews/scenes)
    # Otherwise, fallback to unique cluster_key in clusters_df
    if char_manifest:
        # characters.json structure: { "<MOVIE_TITLE>": { "<cluster_key>": {...}, ... }, ...}
        for movie_title, chars in char_manifest.items():
            if movie_title.startswith("_"):
                continue
            # find movie_id via reverse map
            # rev map is id->title, we need title->id: invert quickly
            title_to_id = {t: mid for mid, t in rev_id_map.items()}
            movie_id = str(title_to_id.get(movie_title, ""))

            # restrict cluster_df for this movie_id if present
            sub = clusters_df[clusters_df["movie_id"] == movie_id] if movie_id else clusters_df

            # load per-movie 512D embeddings
            if movie_title not in per_movie_cache:
                per_movie_cache[movie_title] = _load_per_movie_embeddings(movie_title)
            emb_df = per_movie_cache[movie_title]

            for cluster_key, payload in (chars or {}).items():
                # frames that belong to this cluster
                frames = sub[sub["cluster_key"].astype(str) == str(cluster_key)]["frame"].dropna().astype(int)
                if frames.empty:
                    # cannot compute centroid; skip
                    continue

                # join to get 512D vectors
                join = emb_df.merge(frames.to_frame("frame"), on="frame", how="inner")
                if join.empty:
                    continue

                vecs = np.array(join["embedding"].tolist(), dtype=np.float32)
                if vecs.ndim != 2 or vecs.shape[1] < 128:
                    # invalid
                    continue

                vecs = _l2_normalize(vecs.astype(np.float32))
                centroid = _l2_normalize(vecs.mean(axis=0, keepdims=True))[0]

                rows.append(centroid)
                metas.append({
                    "cluster_key": str(cluster_key),
                    "movie_id": movie_id,
                    "movie": movie_title,
                    "rep_image": (payload or {}).get("rep_image"),
                    "preview_paths": (payload or {}).get("preview_paths", []),
                    "scenes": (payload or {}).get("scenes", []),
                })
    else:
        # Fallback: build by unique cluster_key in clusters_df, group by movie
        grouped = clusters_df.groupby(["movie_id", "cluster_key"])
        # need id->title
        id_to_title = rev_id_map
        # pre-load all movies that appear
        for (movie_id, cluster_key), g in grouped:
            title = id_to_title.get(str(movie_id))
            if not title:
                continue
            if title not in per_movie_cache:
                per_movie_cache[title] = _load_per_movie_embeddings(title)
            emb_df = per_movie_cache[title]
            frames = g["frame"].dropna().astype(int)
            join = emb_df.merge(frames.to_frame("frame"), on="frame", how="inner")
            if join.empty:
                continue
            vecs = np.array(join["embedding"].tolist(), dtype=np.float32)
            if vecs.ndim != 2 or vecs.shape[1] < 128:
                continue
            vecs = _l2_normalize(vecs)
            centroid = _l2_normalize(vecs.mean(axis=0, keepdims=True))[0]
            rows.append(centroid)
            metas.append({
                "cluster_key": str(cluster_key),
                "movie_id": str(movie_id),
                "movie": title,
                "rep_image": None,
                "preview_paths": [],
                "scenes": [],
            })

    if not rows:
        # no data
        __INDEX_MATRIX = np.zeros((0, 512), dtype=np.float32)
        __INDEX_META = []
        return __INDEX_MATRIX, __INDEX_META

    M = np.vstack(rows).astype(np.float32)
    M = _l2_normalize(M)

    __INDEX_MATRIX = M
    __INDEX_META = metas
    return __INDEX_MATRIX, __INDEX_META


# ------------------------------------------------------------
# Search API: cosine similarity
# ------------------------------------------------------------

def search_by_embedding(
    query_vec: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.25
) -> List[Dict]:
    """
    query_vec: shape (512,), raw 512-D; will be L2-normalized internally
    return: [{score, cluster_key, movie, movie_id, rep_image, preview_paths, scenes}, ...]
    """
    M, metas = build_character_index()
    if M.shape[0] == 0:
        return []

    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    if q.shape[1] < 128:
        # guard against wrong dimension (e.g., PCA 6-D)
        return []
    q = _l2_normalize(q)

    # cosine similarity = dot since both are L2-normalized
    sims = (M @ q.T).ravel()
    order = np.argsort(-sims)[: max(top_k, 1)]
    results: List[Dict] = []
    for idx in order:
        score = float(sims[idx])
        if score < float(min_score):
            continue
        meta = metas[idx].copy()
        meta["score"] = score
        results.append(meta)
    return results


# small utility for recognition service to clear cache when warehouse updates
def clear_index_cache():
    global __INDEX_MATRIX, __INDEX_META
    __INDEX_MATRIX = None
    __INDEX_META = None
