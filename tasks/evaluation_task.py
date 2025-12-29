"""
Evaluation Task - Stage 12 in Pipeline

Evaluates clustering quality using INTERNAL METRICS (unsupervised).
Optional: Also matches labeled faces if warehouse/labeled_faces exists.

Internal metrics computed:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Dunn Index

Visualizations generated:
- UMAP 2D projection
- Internal metrics bar chart
- Cluster cohesion chart
- Cluster composition matrix (if labeled faces available)
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from prefect import task

from utils.visualization import (
    compute_internal_metrics,
    plot_umap_2d,
    plot_internal_metrics_bar,
    plot_cluster_cohesion_chart,
    plot_cluster_composition_matrix,
    print_internal_metrics_summary
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@task(name="Evaluation")
def evaluation_task(cfg: dict) -> Dict[str, float]:
    """
    Evaluate clustering results using INTERNAL METRICS (unsupervised).
    
    This task:
    1. Loads clustering results from parquet
    2. Computes internal metrics (Silhouette, DB, CH, Dunn)
    3. Generates UMAP 2D visualization
    4. Generates internal metrics bar chart
    5. Generates cluster cohesion chart
    6. (Optional) If labeled_faces exists, generates composition matrix
    7. Prints summary to console
    
    Args:
        cfg: Config dictionary
        
    Returns:
        Dict with evaluation metrics
    """
    print("[Info] Starting clustering evaluation with INTERNAL metrics...")
    
    # Get paths from config
    labeled_faces_path = cfg.get('evaluation', {}).get('labeled_faces_path', 'warehouse/labeled_faces')
    output_dir_base = cfg.get('evaluation', {}).get('output_dir', 'warehouse/evaluation')
    clusters_parquet = cfg.get('storage', {}).get('clusters_merged_parquet', 'warehouse/parquet/clusters_merged.parquet')
    
    # Get movie name
    movie_name = os.getenv('FS_ACTIVE_MOVIE', 'unknown')
    
    # Create per-movie output directory
    output_dir = Path(output_dir_base) / movie_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Info] Evaluation outputs will be saved to: {output_dir}")
    
    # Load cluster data
    print("[Info] Loading cluster data from parquet...")
    import pandas as pd
    
    try:
        clusters_df = pd.read_parquet(clusters_parquet)
    except Exception as e:
        print(f"[Error] Failed to load clusters parquet: {e}")
        return {}
    
    # Get current movie's data
    movie_meta = cfg.get('storage', {}).get('metadata_json', 'Data/metadata.json')
    movie_id = None
    
    try:
        with open(movie_meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            movie_id_map = meta.get('movie_id_map', {})
            movie_id = movie_id_map.get(movie_name, None)
    except:
        pass
    
    if movie_id is None and 'movie_id' in clusters_df.columns and not clusters_df.empty:
        movie_id = clusters_df['movie_id'].iloc[0]
    
    # Filter to current movie
    if movie_id is not None and 'movie_id' in clusters_df.columns:
        clusters_df = clusters_df[clusters_df['movie_id'] == movie_id]
    
    # Check if empty after filter
    if clusters_df.empty:
        print(f"\n⚠️  Không có dữ liệu cluster cho movie '{movie_name}'.")
        print("💡 Video có thể quá ngắn hoặc min_cluster_size quá cao.")
        return {'error': 'No cluster data', 'num_samples': 0, 'num_clusters': 0}
    
    print(f"[Info] Loaded {len(clusters_df)} faces for movie '{movie_name}'")
    
    # Determine cluster column
    cluster_col = 'final_character_id' if 'final_character_id' in clusters_df.columns else 'cluster_id'
    
    # Build embeddings and labels arrays
    print("[Info] Preparing embeddings for internal metrics...")
    
    embeddings_list = []
    cluster_labels_list = []
    cluster_embeddings = {}  # For cohesion chart
    
    # Map cluster IDs to integers for sklearn
    unique_clusters = clusters_df[cluster_col].unique()
    cluster_to_int = {cid: i for i, cid in enumerate(unique_clusters)}
    
    for _, row in clusters_df.iterrows():
        emb = row.get('track_centroid')
        cid = row[cluster_col]
        
        if emb is not None:
            try:
                emb_array = np.array(emb)
                embeddings_list.append(emb_array)
                cluster_labels_list.append(cluster_to_int[cid])
                
                # For cohesion chart
                if cid not in cluster_embeddings:
                    cluster_embeddings[cid] = []
                cluster_embeddings[cid].append(emb_array)
            except:
                continue
    
    if len(embeddings_list) < 2:
        print("[Error] Not enough embeddings for evaluation!")
        return {}
    
    embeddings = np.array(embeddings_list)
    cluster_labels = np.array(cluster_labels_list)
    
    print(f"[Info] Prepared {len(embeddings)} embeddings in {len(unique_clusters)} clusters")
    
    # =================================================================
    # COMPUTE INTERNAL METRICS (KHÔNG CẦN NHÃN)
    # =================================================================
    print("[Info] Computing internal metrics...")
    
    internal_metrics = compute_internal_metrics(embeddings, cluster_labels)
    
    # Print summary to console
    print_internal_metrics_summary(internal_metrics, f"INTERNAL METRICS - {movie_name}")
    
    # =================================================================
    # GENERATE VISUALIZATIONS
    # =================================================================
    print("[Info] Generating visualizations...")
    
    # 1. UMAP 2D projection
    umap_path = output_dir / 'umap_projection.png'
    try:
        plot_umap_2d(
            embeddings,
            cluster_labels,
            str(umap_path),
            title=f"UMAP Projection - {movie_name}"
        )
    except Exception as e:
        print(f"[Warning] UMAP visualization failed: {e}")
    
    # 2. Internal metrics bar chart
    metrics_bar_path = output_dir / 'internal_metrics.png'
    plot_internal_metrics_bar(
        internal_metrics,
        str(metrics_bar_path),
        title=f"Chỉ Số Phân Cụm - {movie_name}"
    )
    
    # 3. Cluster cohesion chart
    cohesion_path = output_dir / 'cluster_cohesion.png'
    plot_cluster_cohesion_chart(
        list(clusters_df[cluster_col]),
        cluster_embeddings,
        str(cohesion_path),
        title=f"Độ Đồng Nhất Cụm - {movie_name}"
    )
    
    # =================================================================
    # OPTIONAL: LABELED FACES MATCHING (NẾU CÓ DỮ LIỆU NHÃN)
    # =================================================================
    labeled_faces_available = Path(labeled_faces_path).exists() and any(Path(labeled_faces_path).iterdir())
    
    if labeled_faces_available:
        print("\n[Info] Found labeled_faces directory. Generating composition analysis...")
        
        try:
            composition_result = _generate_composition_analysis(
                cfg, clusters_df, cluster_col, cluster_embeddings, 
                labeled_faces_path, output_dir
            )
            if composition_result:
                internal_metrics['composition_analysis'] = True
        except Exception as e:
            print(f"[Warning] Composition analysis failed: {e}")
            internal_metrics['composition_analysis'] = False
    else:
        print("[Info] No labeled_faces found. Skipping composition analysis.")
        internal_metrics['composition_analysis'] = False
    
    # =================================================================
    # SAVE RESULTS
    # =================================================================
    results_path = output_dir / 'results.json'
    
    results = {
        'movie': movie_name,
        'internal_metrics': {
            'silhouette': internal_metrics.get('silhouette'),
            'davies_bouldin': internal_metrics.get('davies_bouldin'),
            'calinski_harabasz': internal_metrics.get('calinski_harabasz'),
            'dunn_index': internal_metrics.get('dunn_index'),
            'mean_intra_cluster_distance': internal_metrics.get('mean_intra_cluster_distance')
        },
        'dataset_info': {
            'num_faces': internal_metrics.get('num_samples'),
            'num_clusters': internal_metrics.get('num_clusters')
        },
        'composition_analysis_available': internal_metrics.get('composition_analysis', False),
        'visualizations': {
            'umap_projection': str(umap_path),
            'internal_metrics_chart': str(metrics_bar_path),
            'cluster_cohesion': str(cohesion_path)
        }
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Info] ✅ Evaluation complete! Results saved to: {results_path}")
    
    return internal_metrics


def _generate_composition_analysis(
    cfg: dict,
    clusters_df,
    cluster_col: str,
    cluster_embeddings: dict,
    labeled_faces_path: str,
    output_dir: Path
) -> bool:
    """
    Phân tích thành phần cụm dựa trên dữ liệu nhãn.
    
    Chỉ chạy khi có thư mục labeled_faces.
    """
    from utils.dataset_validator import validate_labeled_faces_dataset
    from insightface.app import FaceAnalysis
    import cv2
    
    # Validate labeled faces
    validation = validate_labeled_faces_dataset(labeled_faces_path)
    if not validation['valid']:
        print(f"[Warning] Labeled faces validation failed: {validation.get('error')}")
        return False
    
    print(f"[Info] Found {validation['num_actors']} actors with {validation['total_images']} labeled images")
    
    # Initialize face model
    app = FaceAnalysis(
        name=cfg.get('embedding', {}).get('model', 'buffalo_l'),
        providers=cfg.get('embedding', {}).get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Extract embeddings from labeled faces
    labeled_embeddings = []
    labeled_actors = []
    
    for actor in validation['actors']:
        actor_name = actor['name']
        actor_folder = Path(labeled_faces_path) / actor_name
        
        for img_file in actor_folder.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue
            
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                faces = app.get(img)
                
                # Retry with padding
                if len(faces) == 0:
                    pad = 50
                    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[128, 128, 128])
                    faces = app.get(padded)
                
                if len(faces) > 0:
                    labeled_embeddings.append(faces[0].normed_embedding)
                    labeled_actors.append(actor_name)
            except:
                continue
    
    if len(labeled_embeddings) == 0:
        print("[Warning] No embeddings extracted from labeled faces")
        return False
    
    print(f"[Info] Extracted {len(labeled_embeddings)} embeddings from labeled faces")
    
    # Match cluster faces to labeled actors
    cluster_face_labels = []
    matched_actor_labels = []
    similarity_threshold = 0.33
    
    for cluster_id, face_embs in cluster_embeddings.items():
        for face_emb in face_embs:
            best_actor = None
            best_sim = -1.0
            
            for test_emb, actor_name in zip(labeled_embeddings, labeled_actors):
                sim = cosine_similarity(face_emb, test_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_actor = actor_name
            
            cluster_face_labels.append(cluster_id)
            matched_actor_labels.append(best_actor if best_sim > similarity_threshold else "Khác")
    
    # Generate composition matrix
    composition_path = output_dir / 'cluster_composition.png'
    plot_cluster_composition_matrix(
        cluster_face_labels,
        matched_actor_labels,
        cluster_embeddings,
        str(composition_path),
        title="Thành Phần Cụm"
    )
    
    return True
