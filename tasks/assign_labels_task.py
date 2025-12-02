# tasks/assign_labels_task.py
from __future__ import annotations

import os
import json
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


def _extract_embedding_from_file(app: FaceAnalysis, path: Path) -> np.ndarray | None:
    if not path.exists(): return None
    try:
        img = cv2.imread(str(path))
        if img is None: return None
        faces = app.get(img)
        if not faces: return None
        # Lấy mặt lớn nhất
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best_face.normed_embedding if hasattr(best_face, 'normed_embedding') else best_face.embedding
        if emb is None: return None
        return l2_normalize(np.array(emb, dtype=np.float32))
    except Exception as e:
        print(f"[Labeling] Error processing {path.name}: {e}")
        return None


def _build_reference_index(labeled_root: Path) -> Tuple[Any, Dict[int, str], int, int]:
    if not labeled_root.exists(): return None, {}, 0, 0
    app = _get_app()
    if app is None: return None, {}, 0, 0

    embeddings, ids = [], []
    id_to_name = {}
    current_id = 0

    actor_folders = [f for f in labeled_root.iterdir() if f.is_dir()]
    print(f"[Labeling] Found {len(actor_folders)} actor folders.")

    for folder in actor_folders:
        actor_name = folder.name
        image_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))

        count = 0
        for img_path in image_files:
            emb = _extract_embedding_from_file(app, img_path)
            if emb is not None:
                embeddings.append(emb)
                ids.append(current_id)
                id_to_name[current_id] = actor_name
                current_id += 1
                count += 1
        if count > 0:
            print(f"  + Indexed '{actor_name}': {count} images")

    if not embeddings: return None, {}, 0, 0

    emb_matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb_matrix)
    return index, id_to_name, len(embeddings), d


# --- Task Chính ---
@task(name="Assign Labels Task")
def assign_labels_task(cfg: dict | None = None) -> str:
    """
    Tự động gán nhãn dựa trên ảnh mẫu trong warehouse/labeled_faces.
    """
    if cfg is None: cfg = load_config()

    label_cfg = cfg.get("labeling", {})
    if not label_cfg.get("enable", False):
        print("[Labeling] Disabled in config.")
        return "Disabled"

    storage = cfg["storage"]
    labeled_root = Path(storage.get("labeled_faces_root", "warehouse/labeled_faces"))
    characters_json_path = Path(storage.get("characters_json", "warehouse/characters.json"))
    # Ưu tiên đọc từ file merged parquet
    clusters_path = Path(storage.get("clusters_merged_parquet", "warehouse/parquet/clusters_merged.parquet"))

    sim_threshold = float(label_cfg.get("similarity_threshold", 0.55))

    # 1. Build Index
    index, id_map, ref_count, ref_dim = _build_reference_index(labeled_root)
    if index is None:
        print("[Labeling] No reference images. Skipping.")
        return "NoRefImages"

    # 2. Load Clusters
    if not clusters_path.exists():
        print(f"[Labeling] Clusters file missing: {clusters_path}")
        return "NoClusters"

    df = pd.read_parquet(clusters_path)
    active_movie = (os.getenv("FS_ACTIVE_MOVIE") or "").strip()
    if active_movie and "movie" in df.columns:
        df = df[df["movie"] == active_movie]

    if df.empty:
        print("[Labeling] No cluster data to label.")
        return "NoData"

    char_col = next((c for c in ["final_character_id", "cluster_id"] if c in df.columns), None)

    # 3. Compute Centroids (Optimized)
    # Thay vì tính lại từ raw vector, ta gom nhóm track_centroid có sẵn
    print(f"[Labeling] Computing centroids for {df[char_col].nunique()} clusters...")

    cluster_vectors = []
    cluster_ids = []

    # Lọc bỏ các dòng có track_centroid bị null/NaN
    df_valid = df.dropna(subset=["track_centroid"])

    for c_id, group in df_valid.groupby(char_col):
        # Stack các track_centroid lại (chúng đã là list/array float)
        # Lưu ý: track_centroid trong parquet có thể được lưu dưới dạng list
        vecs = np.stack(group["track_centroid"].values)
        if vecs.size == 0: continue

        # Tính mean của cluster và normalize
        mean_vec = np.mean(vecs, axis=0)
        norm_vec = l2_normalize(mean_vec)

        cluster_vectors.append(norm_vec)
        cluster_ids.append(str(c_id))

    if not cluster_vectors:
        print("[Labeling] No valid centroids.")
        return "NoCentroids"

    query_matrix = np.ascontiguousarray(np.stack(cluster_vectors), dtype=np.float32)

    # 4. Search
    D, I = index.search(query_matrix, k=1)

    matches = {}
    for idx, (sim, ref_idx) in enumerate(zip(D, I)):
        score = float(sim[0])
        if score >= sim_threshold:
            ref_id = int(ref_idx[0])
            name = id_map.get(ref_id, "Unknown")
            c_id = cluster_ids[idx]
            matches[c_id] = {"name": name, "score": score}
            print(f"  [Match] Cluster {c_id} == {name} ({score:.3f})")

    # 5. Update JSON
    if not characters_json_path.exists():
        print("[Labeling] characters.json missing. Run character_task first.")
        return "NoJson"

    try:
        with open(characters_json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        updated = 0
        movies = [active_movie] if active_movie in manifest else manifest.keys()

        for mov in movies:
            chars = manifest.get(mov, {})
            for c_id, c_data in chars.items():
                if c_id in matches:
                    # [Logic Override] Nếu manual đã gán tên rồi thì có thể chọn bỏ qua
                    # Ở đây ta ưu tiên Auto nếu status != 'MANUAL'
                    if c_data.get("label_status") == "MANUAL":
                        continue

                    m = matches[c_id]
                    c_data["name"] = m["name"]
                    c_data["label_status"] = "AUTO_ASSIGNED"
                    c_data["label_confidence"] = round(m["score"], 4)
                    updated += 1

        if updated > 0:
            tmp = characters_json_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            os.replace(tmp, characters_json_path)
            print(f"[Labeling] Updated {updated} labels in characters.json")
        else:
            print("[Labeling] No new labels applied.")

    except Exception as e:
        print(f"[Labeling] Error updating json: {e}")
        return "Error"

    return "Success"