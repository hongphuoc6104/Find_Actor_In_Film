#!/usr/bin/env python3
"""
Script chạy evaluation trên 5 video để tạo sơ đồ cho luận văn.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Thêm project root vào path
sys.path.insert(0, '/home/hongphuoc/Desktop/myproject')

from utils.visualization import (
    compute_internal_metrics,
    plot_umap_2d,
    plot_internal_metrics_bar,
    plot_cluster_cohesion_chart,
    print_internal_metrics_summary,
    plot_multi_video_metrics_comparison
)

# 5 VIDEO ĐỂ ĐÁNH GIÁ (đa dạng về thể loại và độ dài)
VIDEOS = [
    'CHUYENXOMTUI',    # Sitcom nhiều nhân vật
    'EMCHUA18',        # Phim ngắn
    'DENAMHON',        # Phim kinh dị, ánh sáng yếu
    'HEMCUT',          # Drama
    'TAMCAM',          # Phim cổ trang, nhiều nhân vật
]

OUTPUT_DIR = Path('/home/hongphuoc/Desktop/myproject/warehouse/evaluation')
PARQUET_DIR = Path('/home/hongphuoc/Desktop/myproject/warehouse/parquet')


def load_movie_data(movie_name: str):
    """Load embeddings và cluster labels từ parquet file."""
    parquet_file = PARQUET_DIR / f'{movie_name}_clusters.parquet'
    
    if not parquet_file.exists():
        print(f"[Error] File không tồn tại: {parquet_file}")
        return None, None, None
    
    df = pd.read_parquet(parquet_file)
    
    # Xác định cột cluster
    cluster_col = 'final_character_id' if 'final_character_id' in df.columns else 'cluster_id'
    
    # Lấy embeddings và labels
    embeddings_list = []
    cluster_labels_list = []
    cluster_embeddings = {}
    
    # Map cluster IDs to integers
    unique_clusters = df[cluster_col].unique()
    cluster_to_int = {cid: i for i, cid in enumerate(unique_clusters)}
    
    for _, row in df.iterrows():
        emb = row.get('track_centroid')
        cid = row[cluster_col]
        
        if emb is not None:
            try:
                emb_array = np.array(emb)
                embeddings_list.append(emb_array)
                cluster_labels_list.append(cluster_to_int[cid])
                
                if cid not in cluster_embeddings:
                    cluster_embeddings[cid] = []
                cluster_embeddings[cid].append(emb_array)
            except:
                continue
    
    if len(embeddings_list) < 10:
        print(f"[Warning] Không đủ embeddings cho {movie_name}")
        return None, None, None
    
    return (
        np.array(embeddings_list), 
        np.array(cluster_labels_list),
        cluster_embeddings
    )


def evaluate_single_movie(movie_name: str):
    """Chạy evaluation cho một video."""
    print(f"\n{'='*60}")
    print(f"📽️  ĐÁNH GIÁ: {movie_name}")
    print(f"{'='*60}")
    
    # Load data
    embeddings, cluster_labels, cluster_embeddings = load_movie_data(movie_name)
    
    if embeddings is None:
        return None
    
    print(f"[Info] Loaded {len(embeddings)} embeddings, {len(set(cluster_labels))} clusters")
    
    # Tạo output directory
    movie_output_dir = OUTPUT_DIR / movie_name
    movie_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute internal metrics
    metrics = compute_internal_metrics(embeddings, cluster_labels)
    print_internal_metrics_summary(metrics, f"INTERNAL METRICS - {movie_name}")
    
    # Generate visualizations
    # 1. UMAP 2D
    umap_path = movie_output_dir / 'umap_projection.png'
    try:
        plot_umap_2d(
            embeddings, cluster_labels,
            str(umap_path),
            title=f"UMAP Projection - {movie_name}"
        )
    except Exception as e:
        print(f"[Warning] UMAP failed: {e}")
    
    # 2. Internal metrics bar
    metrics_path = movie_output_dir / 'internal_metrics.png'
    plot_internal_metrics_bar(
        metrics,
        str(metrics_path),
        title=f"Chỉ Số Phân Cụm - {movie_name}"
    )
    
    # 3. Cluster cohesion
    cohesion_path = movie_output_dir / 'cluster_cohesion.png'
    plot_cluster_cohesion_chart(
        list(cluster_embeddings.keys()) * 1,  # Dummy list
        cluster_embeddings,
        str(cohesion_path),
        title=f"Độ Đồng Nhất Cụm - {movie_name}"
    )
    
    # Save results JSON
    results_path = movie_output_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'movie': movie_name,
            'internal_metrics': {
                'silhouette': metrics.get('silhouette'),
                'davies_bouldin': metrics.get('davies_bouldin'),
                'calinski_harabasz': metrics.get('calinski_harabasz'),
                'dunn_index': metrics.get('dunn_index'),
            },
            'num_faces': metrics.get('num_samples'),
            'num_clusters': metrics.get('num_clusters')
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[Info] ✅ Saved results to: {movie_output_dir}")
    
    return metrics


def main():
    print("\n" + "="*70)
    print("🎬 CHẠY ĐÁNH GIÁ PHÂN CỤM TRÊN 5 VIDEO")
    print("="*70)
    
    all_metrics = {}
    
    for movie in VIDEOS:
        metrics = evaluate_single_movie(movie)
        if metrics:
            all_metrics[movie] = metrics
    
    # Tạo biểu đồ so sánh tất cả video
    if len(all_metrics) >= 2:
        print("\n" + "="*60)
        print("📊 TẠO BIỂU ĐỒ SO SÁNH TỔNG HỢP")
        print("="*60)
        
        comparison_path = OUTPUT_DIR / 'all_videos_comparison.png'
        plot_multi_video_metrics_comparison(
            all_metrics,
            str(comparison_path),
            title="So Sánh Chỉ Số Phân Cụm Giữa Các Video"
        )
    
    # Tổng kết
    print("\n" + "="*70)
    print("📋 TỔNG KẾT")
    print("="*70)
    print(f"\n| Video | Silhouette | Davies-Bouldin | Dunn Index | Clusters |")
    print(f"|-------|------------|----------------|------------|----------|")
    
    for movie, m in all_metrics.items():
        sil = m.get('silhouette', 0)
        db = m.get('davies_bouldin', 0)
        dunn = m.get('dunn_index', 0)
        n_clusters = m.get('num_clusters', 0)
        print(f"| {movie[:15]:15} | {sil:10.4f} | {db:14.4f} | {dunn:10.4f} | {n_clusters:8} |")
    
    print(f"\n✅ Kết quả được lưu tại: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
