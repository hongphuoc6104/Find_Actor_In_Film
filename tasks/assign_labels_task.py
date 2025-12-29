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
            # Use smaller det_size for better small face detection (labeled images are often crops)
            _APP.prepare(ctx_id=ctx_id, det_size=(416, 416))
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

    sim_threshold = float(label_cfg.get("similarity_threshold", 0.55))

    # 1. Build Reference Index
    index, id_map, ref_count, ref_dim = _build_reference_index(labeled_root)
    if index is None:
        print("[Labeling] No reference images. Skipping.")
        return "NoRefImages"

    # 2. Load characters.json to get list of movies
    if not characters_json_path.exists():
        print("[Labeling] characters.json missing.")
        return "NoJson"

    try:
        with open(characters_json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[Labeling] Error loading JSON: {e}")
        return "Error"

    # 3. Process each movie separately
    total_matches = {}
    
    for movie_title in manifest.keys():
        if movie_title.startswith("_"):
            continue

        print(f"\n[Labeling] Processing movie: {movie_title}")
        
        # Load per-movie parquet file
        parquet_path = Path(f"warehouse/parquet/{movie_title}_clusters.parquet")
        if not parquet_path.exists():
            print(f"  ⚠️  Parquet not found: {parquet_path}")
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  ⚠️  Error loading parquet: {e}")
            continue

        if df.empty:
            print(f"  ⚠️  Empty parquet")
            continue

        char_col = next((c for c in ["final_character_id", "cluster_id"] if c in df.columns), None)
        if not char_col:
            print(f"  ⚠️  No character ID column")
            continue

        # Compute centroids for this movie
        cluster_vectors = []
        cluster_ids = []

        df_valid = df.dropna(subset=["track_centroid"])

        for c_id, group in df_valid.groupby(char_col):
            vecs = np.stack(group["track_centroid"].values)
            if vecs.size == 0:
                continue

            mean_vec = np.mean(vecs, axis=0)
            norm_vec = l2_normalize(mean_vec)

            cluster_vectors.append(norm_vec)
            cluster_ids.append(str(c_id))

        if not cluster_vectors:
            print(f"  ⚠️  No valid centroids")
            continue

        query_matrix = np.ascontiguousarray(np.stack(cluster_vectors), dtype=np.float32)

        # Search against labeled faces
        D, I = index.search(query_matrix, k=1)

        # Collect matches for this movie
        for idx, (sim, ref_idx) in enumerate(zip(D, I)):
            score = float(sim[0])
            if score >= sim_threshold:
                ref_id = int(ref_idx[0])
                name = id_map.get(ref_id, "Unknown")
                c_id = cluster_ids[idx]
                
                if movie_title not in total_matches:
                    total_matches[movie_title] = {}
                
                total_matches[movie_title][c_id] = {"name": name, "score": score}
                print(f"  ✅ Match: {c_id} == {name} ({score:.3f})")

    # 4. Update JSON with all matches
    if not total_matches:
        print("\n[Labeling] No matches found across all movies.")
        return "NoMatches"

    try:
        updated = 0
        for movie_title, matches in total_matches.items():
            if movie_title not in manifest:
                continue
                
            chars = manifest[movie_title]
            for c_id, match_data in matches.items():
                if c_id not in chars:
                    continue
                
                # Skip if already manually labeled
                if chars[c_id].get("label_status") == "MANUAL":
                    continue

                chars[c_id]["name"] = match_data["name"]
                chars[c_id]["label_status"] = "AUTO_ASSIGNED"
                chars[c_id]["label_confidence"] = round(match_data["score"], 4)
                updated += 1

        if updated > 0:
            tmp = characters_json_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            os.replace(tmp, characters_json_path)
            print(f"\n[Labeling] ✅ Updated {updated} labels across {len(total_matches)} movies")
        else:
            print("\n[Labeling] No new labels applied.")

    except Exception as e:
        print(f"[Labeling] Error updating json: {e}")
        return "Error"

    return "Success"