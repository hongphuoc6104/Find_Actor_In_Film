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
CLUSTERS_FINAL_PARQUET = PARQUET_DIR / "clusters_merged.parquet"
CLUSTERS_BASE_PARQUET = PARQUET_DIR / "clusters.parquet"

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


def _load_per_movie_embeddings(movie_title: str) -> Optional[pd.DataFrame]:
    """
    Đọc embeddings 512-D từ Data/embeddings/<title>.parquet.
    Hàm này giờ đây sẽ tìm cả cột 'embedding' và 'emb'.
    """
    emb_path = DATA_DIR / "embeddings" / f"{movie_title}.parquet"
    if not emb_path.exists():
        print(f"[Indexer][WARN] Không tìm thấy file embedding cho phim '{movie_title}' tại {emb_path}")
        return None

    df = pd.read_parquet(emb_path)

    # --- THAY ĐỔI 1: Tìm kiếm tên cột embedding một cách linh hoạt ---
    emb_col_name = None
    if 'embedding' in df.columns:
        emb_col_name = 'embedding'
    elif 'emb' in df.columns:
        emb_col_name = 'emb'

    if not emb_col_name:
        # Lỗi nghiêm trọng, không tìm thấy cột nào
        raise ValueError(f"File {emb_path} thiếu cột 'embedding' hoặc 'emb'. Hãy chạy lại embedding_task.")

    # --- THAY ĐỔI 2: Chuẩn hóa tên cột để các hàm sau sử dụng ---
    # Đổi tên cột tìm thấy (vd: 'emb') thành 'embedding' để xử lý nhất quán
    df.rename(columns={emb_col_name: 'embedding'}, inplace=True)

    # Chuẩn hóa cột frame để join
    if "frame" not in df.columns and "image" in df.columns:
        df['frame'] = df['image']

    return df[["frame", "embedding"]].dropna().copy()


def _load_final_clusters_df() -> Optional[pd.DataFrame]:
    """
    Tải dataframe chứa kết quả gom cụm cuối cùng.
    """
    p = CLUSTERS_FINAL_PARQUET if CLUSTERS_FINAL_PARQUET.exists() else CLUSTERS_BASE_PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file clusters: '{CLUSTERS_FINAL_PARQUET}' hoặc '{CLUSTERS_BASE_PARQUET}'.")

    df = pd.read_parquet(p)

    char_id_col = None
    for col_name in ["final_character_id", "cluster_id", "character_id"]:
        if col_name in df.columns:
            char_id_col = col_name
            break

    if not char_id_col:
        raise ValueError(
            "Không tìm thấy cột định danh nhân vật (vd: final_character_id, cluster_id) trong file clusters.")

    df.rename(columns={char_id_col: "character_id"}, inplace=True)

    if 'movie' not in df.columns:
        df['movie'] = df['character_id'].apply(lambda x: str(x).split('_')[0])

    return df


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


# ------------------------------------------------------------
# Build index: centroid 512-D per character
# ------------------------------------------------------------

def build_character_index(force_rebuild: bool = False) -> Tuple[np.ndarray, List[Dict]]:
    """
    Xây dựng index tìm kiếm từ vector trung tâm của mỗi nhân vật.
    """
    global __INDEX_MATRIX, __INDEX_META
    if __INDEX_MATRIX is not None and __INDEX_META is not None and not force_rebuild:
        return __INDEX_MATRIX, __INDEX_META

    print("[Indexer] Bắt đầu xây dựng hoặc làm mới index nhân vật...")
    char_manifest = _read_json(CHARACTERS_JSON)
    if not char_manifest:
        raise FileNotFoundError(f"Không tìm thấy file manifest nhân vật: {CHARACTERS_JSON}. Hãy chạy pipeline.")

    clusters_df = _load_final_clusters_df()
    if clusters_df is None: return np.array([]), []

    per_movie_emb_cache: Dict[str, pd.DataFrame] = {}
    rows: List[np.ndarray] = []
    metas: List[Dict] = []

    for movie_title, chars_in_movie in char_manifest.items():
        if movie_title.startswith("_"): continue

        if movie_title not in per_movie_emb_cache:
            emb_df = _load_per_movie_embeddings(movie_title)
            if emb_df is None: continue
            per_movie_emb_cache[movie_title] = emb_df

        emb_df = per_movie_emb_cache[movie_title]

        movie_clusters_df = clusters_df[clusters_df['movie'] == movie_title]

        for char_id, char_data in chars_in_movie.items():
            frames_for_char = movie_clusters_df[movie_clusters_df['character_id'] == char_id]
            if frames_for_char.empty: continue

            # Join bằng cột 'frame'
            join_df = pd.merge(frames_for_char, emb_df, on='frame', how='inner')
            if join_df.empty: continue

            vecs = np.array(join_df["embedding"].tolist(), dtype=np.float32)
            if vecs.ndim != 2: continue

            centroid = _l2_normalize(vecs.mean(axis=0, keepdims=True))[0]
            rows.append(centroid)

            # --- [FIX] Thêm name vào metadata để hiển thị ---
            metas.append({
                "character_id": str(char_id),
                "name": char_data.get("name"),  # <-- Thêm dòng này để lấy tên
                "movie": movie_title,
                "rep_image": char_data.get("rep_image"),
                "preview_paths": char_data.get("preview_paths", []),
                "scenes": char_data.get("scenes", []),
            })

    if not rows:
        print("[Indexer][WARN] Không có nhân vật nào được đưa vào index.")
        __INDEX_MATRIX = np.zeros((0, 512), dtype=np.float32)
        __INDEX_META = []
        return __INDEX_MATRIX, __INDEX_META

    M = np.vstack(rows).astype(np.float32)
    __INDEX_MATRIX = _l2_normalize(M)
    __INDEX_META = metas
    print(f"[Indexer] Xây dựng index hoàn tất với {len(metas)} nhân vật.")
    return __INDEX_MATRIX, __INDEX_META


# ------------------------------------------------------------
# Search API
# ------------------------------------------------------------

def search_by_embedding(
        query_vec: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.25
) -> List[Dict]:
    """
    Tìm kiếm vector truy vấn trong index đã xây dựng.
    """
    M, metas = build_character_index()
    if M.shape[0] == 0:
        return []

    q = _l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
    sims = (M @ q.T).ravel()

    order = np.argsort(-sims)[: max(top_k * 2, 10)]

    results: List[Dict] = []
    for idx in order:
        score = float(sims[idx])
        if score < float(min_score):
            continue
        meta = metas[idx].copy()
        meta["score"] = score
        results.append(meta)

    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]


def clear_index_cache():
    global __INDEX_MATRIX, __INDEX_META
    __INDEX_MATRIX = None
    __INDEX_META = None
    print("[Indexer] Cache của index đã được xóa.")