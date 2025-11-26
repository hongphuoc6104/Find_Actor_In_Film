# tasks/assign_labels_task.py
from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import faiss
from prefect import task
from insightface.app import FaceAnalysis

from utils.config_loader import load_config
from utils.vector_utils import l2_normalize

# --- Helper: Lấy model InsightFace ---
_APP = None


def _get_app(ctx_id: int = 0):
    global _APP
    if _APP is None:
        try:
            _APP = FaceAnalysis(name="buffalo_l")
            _APP.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except Exception as e:
            print(f"[Labeling] Init InsightFace error: {e}")
            return None
    return _APP


# --- Helper: Trích xuất embedding từ ảnh mẫu ---
def _extract_embedding_from_file(app: FaceAnalysis, path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        # Đọc ảnh, convert sang RGB để đảm bảo đúng format màu
        img = cv2.imread(str(path))
        if img is None:
            return None
        # InsightFace expect BGR (mặc định cv2) hoặc RGB tùy config, nhưng thường code mẫu dùng trực tiếp cv2.imread
        faces = app.get(img)
        if not faces:
            return None

        # Lấy khuôn mặt to nhất (ưu tiên det_score)
        best_face = max(faces, key=lambda f: f.det_score)

        # Ưu tiên lấy embedding đã chuẩn hóa nếu có
        emb = best_face.normed_embedding if hasattr(best_face, 'normed_embedding') else best_face.embedding

        if emb is None:
            return None

        # Đảm bảo trả về float32
        return l2_normalize(np.array(emb, dtype=np.float32))
    except Exception as e:
        print(f"[Labeling] Error processing {path.name}: {e}")
        return None


# --- Helper: Xây dựng FAISS Index từ folder ---
def _build_reference_index(labeled_root: Path) -> Tuple[Any, Dict[int, str], int, int]:
    """
    Quét thư mục labeled_root, tạo index FAISS.
    Trả về: (Index, Map{id->name}, count, dimension)
    """
    if not labeled_root.exists():
        return None, {}, 0, 0

    app = _get_app()
    if app is None:
        return None, {}, 0, 0

    embeddings = []
    ids = []
    id_to_name = {}
    current_id = 0

    # Duyệt qua từng thư mục con (tên diễn viên)
    actor_folders = [f for f in labeled_root.iterdir() if f.is_dir()]

    if not actor_folders:
        print(f"[Labeling] No actor folders found in {labeled_root}")
        return None, {}, 0, 0

    print(f"[Labeling] Found {len(actor_folders)} actor folders in {labeled_root}")

    for folder in actor_folders:
        actor_name = folder.name
        # Hỗ trợ nhiều định dạng ảnh
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(list(folder.glob(ext)))
            image_files.extend(list(folder.glob(ext.upper())))

        valid_embs_for_actor = 0
        for img_path in image_files:
            emb = _extract_embedding_from_file(app, img_path)
            if emb is not None:
                embeddings.append(emb)
                ids.append(current_id)
                id_to_name[current_id] = actor_name
                current_id += 1
                valid_embs_for_actor += 1

        if valid_embs_for_actor > 0:
            print(f"  + Indexed '{actor_name}': {valid_embs_for_actor} images")

    if not embeddings:
        return None, {}, 0, 0

    # QUAN TRỌNG: Chuyển sang numpy array float32 và C-contiguous
    emb_matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
    d = emb_matrix.shape[1]  # Dimension (thường là 512)

    try:
        # IndexFlatIP: Inner Product (tương đương Cosine Similarity vì vectors đã normalized)
        index = faiss.IndexFlatIP(d)
        index.add(emb_matrix)
        return index, id_to_name, len(embeddings), d
    except Exception as e:
        print(f"[Labeling] FAISS Index build failed: {e}")
        return None, {}, 0, 0


# --- Task Chính ---
@task(name="Assign Labels Task")
def assign_labels_task(cfg: dict | None = None) -> str:
    """
    Tự động gán nhãn cho các cụm nhân vật dựa trên ảnh mẫu.
    """
    print("\n--- Starting Auto-Labeling Task ---")
    if cfg is None:
        cfg = load_config()

    label_cfg = cfg.get("labeling", {})
    if not label_cfg.get("enable", False):
        print("[Labeling] Feature is disabled in config.")
        return "Disabled"

    storage = cfg["storage"]
    labeled_root = Path(storage.get("labeled_faces_root", "warehouse/labeled_faces"))
    clusters_path = Path(storage.get("clusters_merged_parquet", "warehouse/parquet/clusters_merged.parquet"))
    characters_json_path = Path(storage.get("characters_json", "warehouse/characters.json"))

    sim_threshold = float(label_cfg.get("similarity_threshold", 0.55))

    # 1. Build Index từ ảnh mẫu
    index, id_map, ref_count, ref_dim = _build_reference_index(labeled_root)
    if index is None or ref_count == 0:
        print("[Labeling] No reference images indexed. Skipping.")
        return "NoRefImages"

    print(f"[Labeling] Reference Index Dimension: {ref_dim}")

    # 2. Load Clusters
    if not clusters_path.exists():
        print(f"[Labeling] Clusters file not found: {clusters_path}")
        return "NoClusters"

    df = pd.read_parquet(clusters_path)
    if df.empty:
        print("[Labeling] Clusters data is empty.")
        return "EmptyClusters"

    # --- LOGIC MỚI: Lọc theo Active Movie ---
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie:
        print(f"[Labeling] Single-movie mode -> Filtering for '{active_movie}'")
        # Tìm cột movie (có thể là 'movie' hoặc 'movie_id')
        if "movie" in df.columns:
            df = df[df["movie"].astype(str) == active_movie]

    if df.empty:
        print(f"[Labeling] No clusters found for movie '{active_movie}'.")
        return "NoDataForMovie"

    # Kiểm tra cột cần thiết
    char_col = next((c for c in ["final_character_id", "cluster_id"] if c in df.columns), None)
    if not char_col or "track_centroid" not in df.columns:
        print(f"[Labeling] Missing required columns ({char_col}, track_centroid).")
        return "InvalidSchema"

    # 3. Tính Centroids cho các cụm cần gán nhãn
    print(f"[Labeling] Calculating centroids for {df[char_col].nunique()} clusters...")

    cluster_centroids_map = {}
    valid_dims_count = 0
    invalid_dims_count = 0

    for char_id, group in df.groupby(char_col):
        # Lấy list các centroid của track, convert strict sang numpy float32
        raw_vecs = group["track_centroid"].values
        valid_vecs = []
        for v in raw_vecs:
            if v is not None:
                v_arr = np.array(v, dtype=np.float32)
                # Kiểm tra dimension vector query có khớp với reference không
                if v_arr.shape[0] == ref_dim:
                    valid_vecs.append(v_arr)
                else:
                    invalid_dims_count += 1

        if not valid_vecs:
            continue

        # Tính mean vector của cụm
        mean_vec = np.mean(valid_vecs, axis=0)
        # Normalize
        norm_vec = l2_normalize(mean_vec)
        cluster_centroids_map[char_id] = norm_vec
        valid_dims_count += 1

    if invalid_dims_count > 0:
        print(
            f"[Labeling] [WARN] Ignored {invalid_dims_count} track vectors due to dimension mismatch (Expected {ref_dim}).")

    if not cluster_centroids_map:
        print("[Labeling] No valid centroids computed (possibly due to empty data or dimension mismatch).")
        return "NoCentroids"

    # 4. Search và Gán nhãn
    query_ids = list(cluster_centroids_map.keys())
    # QUAN TRỌNG: Query matrix cũng phải là float32 và C-contiguous
    query_matrix = np.ascontiguousarray(np.stack(list(cluster_centroids_map.values())), dtype=np.float32)

    # Final dimension check
    if query_matrix.shape[1] != ref_dim:
        print(
            f"[Labeling] [FATAL] Query dimension {query_matrix.shape[1]} != Index dimension {ref_dim}. FAISS will crash.")
        return "DimMismatch"

    print(f"[Labeling] Searching {len(query_ids)} clusters against {ref_count} reference faces...")

    try:
        # D: distances (similarities), I: indices
        D, I = index.search(query_matrix, k=1)
    except Exception as e:
        print(f"[Labeling] FAISS search failed: {e}")
        return "SearchFailed"

    matches_found = 0
    assigned_labels = {}  # char_id -> info

    for idx, (sim, ref_idx) in enumerate(zip(D, I)):
        sim_val = float(sim[0])
        ref_id = int(ref_idx[0])

        if sim_val >= sim_threshold:
            actor_name = id_map.get(ref_id, "Unknown")
            char_id = query_ids[idx]
            assigned_labels[char_id] = {
                "name": actor_name,
                "score": sim_val
            }
            matches_found += 1
            print(f"  -> Match: Cluster '{char_id}' == '{actor_name}' ({sim_val:.3f})")

    # 5. Cập nhật characters.json
    if not characters_json_path.exists():
        print("[Labeling] characters.json not found to update.")
        return "NoCharJson"

    try:
        # Đọc ghi atomic
        with open(characters_json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        updated_count = 0
        # Duyệt qua manifest để update
        # Lưu ý: active_movie có thể được dùng để tối ưu vòng lặp
        movies_to_scan = [active_movie] if active_movie and active_movie in manifest else manifest.keys()

        for movie_key in movies_to_scan:
            chars = manifest.get(movie_key, {})
            for char_id, char_data in chars.items():
                # char_id trong json có thể là string, trong assigned_labels cũng là string (từ cluster_id)
                if char_id in assigned_labels:
                    match_info = assigned_labels[char_id]
                    char_data["name"] = match_info["name"]
                    char_data["label_status"] = "AUTO_ASSIGNED"
                    char_data["label_confidence"] = round(match_info["score"], 4)
                    updated_count += 1

        if updated_count > 0:
            tmp_path = characters_json_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, characters_json_path)
            print(f"[Labeling] Successfully updated {updated_count} labels in characters.json")
        else:
            print("[Labeling] No labels updated (either no matches or clusters not in manifest).")

    except Exception as e:
        print(f"[Labeling] Failed to update json: {e}")
        return "UpdateFailed"

    return "Success"