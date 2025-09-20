# utils/indexer.py
"""Tiện ích xây dựng và tải index tìm kiếm cho các nhân vật."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.config_loader import load_config
from utils.vector_utils import l2_normalize


def build_index(characters_json: str, index_path: str) -> None:
    """Xây dựng FAISS/Annoy index cho các nhân vật.

    Parameters
    ----------
    characters_json : str
        Đường dẫn tới file JSON chứa hồ sơ nhân vật.
    index_path : str
        Đường dẫn nơi lưu index đã tạo.
    """
    # Đọc danh sách nhân vật
    with open(characters_json, "r", encoding="utf-8") as f:
        characters = json.load(f)

    cfg = load_config()
    storage_cfg = cfg.get("storage", {})
    merged_path = storage_cfg.get("clusters_merged_parquet")

    vectors: List[np.ndarray] = []
    char_refs: List[Dict[str, str]] = []

    centroid_df = None
    if merged_path and os.path.exists(merged_path):
        centroid_df = pd.read_parquet(merged_path)

    def _lookup_centroid(movie_id: str, character_id: str) -> np.ndarray | None:
        if centroid_df is None or centroid_df.empty:
            return None
        mask = (
            centroid_df["movie_id"].astype(str) == movie_id
        ) & (centroid_df["character_id"].astype(str) == character_id)
        rows = centroid_df[mask]
        if rows.empty:
            return None
        value = rows.iloc[0].get("embedding")
        if value is None:
            return None
        arr = np.asarray(value, dtype="float32")
        if arr.ndim == 1:
            return arr
        return arr.reshape(-1)

    for movie_id, char_map in characters.items():
        for character_id, info in char_map.items():
            emb = info.get("embedding")
            if emb is None:
                emb = _lookup_centroid(movie_id, character_id)
            if emb is None:
                continue
            vector = l2_normalize(np.asarray(emb, dtype="float32"))
            vectors.append(vector)
            char_refs.append({"movie_id": str(movie_id), "character_id": str(character_id)})

    if not vectors:
        print("[Indexer] Không tìm thấy vector để xây index.")
        return

    vecs = np.stack(vectors).astype("float32")
    dim = vecs.shape[1]
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    use_faiss_gpu = False
    try:
        import faiss  # type: ignore

        use_faiss_gpu = faiss.get_num_gpus() > 0
    except Exception:
        faiss = None  # type: ignore

    if use_faiss_gpu and faiss is not None:
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        faiss.write_index(index, index_path)
        print(f"[Indexer] Đã lưu FAISS index vào {index_path}")
    else:
        try:
            from annoy import AnnoyIndex  # type: ignore

            index = AnnoyIndex(dim, "angular")
            for i, v in enumerate(vecs):
                index.add_item(i, v)
            index.build(10)
            index.save(index_path)
            print(f"[Indexer] Đã lưu Annoy index vào {index_path}")
        except Exception:
            if faiss is None:
                raise
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            faiss.write_index(index, index_path)
            print(f"[Indexer] Đã lưu FAISS CPU index vào {index_path}")

    # Lưu mapping index -> char_id nếu có cấu hình
    map_path = storage_cfg.get("index_map")
    if map_path:
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        mapping = {i: ref for i, ref in enumerate(char_refs)}
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"[Indexer] Đã lưu index map vào {map_path}")


def _infer_dim(cfg: Dict[str, Any], id_map: Dict[int, Dict[str, str]]) -> int:
    """Suy ra số chiều embedding để nạp Annoy index."""
    emb_cfg = cfg.get("embedding", {})
    dim = emb_cfg.get("dim") or emb_cfg.get("size")
    if dim:
        return int(dim)

    storage_cfg = cfg.get("storage", {})
    characters_json = storage_cfg.get("characters_json")
    if characters_json and os.path.exists(characters_json):
        with open(characters_json, "r", encoding="utf-8") as f:
            characters = json.load(f)
        for meta in id_map.values():
            movie_id = str(meta.get("movie_id"))
            character_id = str(meta.get("character_id"))
            info = characters.get(movie_id, {}).get(character_id)
            if info and info.get("embedding"):
                return len(info["embedding"])
    raise RuntimeError("Không xác định được số chiều embedding cho Annoy index")


def load_index() -> Tuple[Any, Dict[int, int]]:
    """Tải index và mapping từ file cấu hình.

    Hỗ trợ cả FAISS lẫn Annoy tuỳ thuộc vào loại index đã lưu.
    """

    cfg = load_config()
    storage_cfg = cfg.get("storage", {})
    index_path = storage_cfg.get("index_path")
    map_path = storage_cfg.get("index_map")

    if not index_path or not os.path.exists(index_path):
        raise FileNotFoundError("Không tìm thấy index: %s" % index_path)
    if not map_path or not os.path.exists(map_path):
        raise FileNotFoundError("Không tìm thấy index map: %s" % map_path)

    with open(map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    id_map: Dict[int, Dict[str, str]] = {}
    for key, value in raw_map.items():
        idx = int(key)
        if isinstance(value, dict):
            movie_id = str(value.get("movie_id"))
            character_id = str(value.get("character_id"))
        else:
            movie_id = "0"
            character_id = str(value)
        id_map[idx] = {"movie_id": movie_id, "character_id": character_id}

    # Thử đọc FAISS trước
    try:
        import faiss  # type: ignore

        index = faiss.read_index(index_path)
        return index, id_map
    except Exception:
        pass

    # Fallback Annoy
    try:
        from annoy import AnnoyIndex  # type: ignore

        dim = _infer_dim(cfg, id_map)
        index = AnnoyIndex(dim, "angular")
        index.load(index_path)
        return index, id_map
    except Exception as exc:  # pragma: no cover - runtime dependence
        raise RuntimeError(f"Không thể tải index: {exc}")