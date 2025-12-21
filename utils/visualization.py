"""
Visualization Module for Clustering Evaluation (Unsupervised Version)

Generates charts and reports for evaluation results using internal metrics only.
All visualizations work WITHOUT ground truth labels.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter


def compute_internal_metrics(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá nội tại cho phân cụm (KHÔNG CẦN NHÃN).
    
    Các chỉ số được tính:
    1. Silhouette Score - Đo độ gắn kết nội cụm và độ phân tách liên cụm
    2. Davies-Bouldin Index - Tỷ lệ phân tán/khoảng cách (thấp = tốt)
    3. Calinski-Harabasz Index - Tỷ lệ phương sai (cao = tốt)
    4. Dunn Index - Khoảng cách liên cụm min / đường kính cụm max (cao = tốt)
    
    Args:
        embeddings: Ma trận embeddings shape (n_samples, n_features)
        cluster_labels: Nhãn cụm cho mỗi mẫu
        
    Returns:
        Dict chứa các metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.metrics.pairwise import cosine_distances
    
    # Validate input
    if len(embeddings) < 2:
        return {
            'silhouette': 0.0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0.0,
            'dunn_index': 0.0,
            'num_samples': len(embeddings),
            'num_clusters': 0
        }
    
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return {
            'silhouette': 0.0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0.0,
            'dunn_index': 0.0,
            'num_samples': len(embeddings),
            'num_clusters': n_clusters
        }
    
    # Tính các metrics cơ bản
    try:
        sil_score = silhouette_score(embeddings, cluster_labels, metric='cosine')
    except Exception as e:
        print(f"[Warning] Silhouette calculation failed: {e}")
        sil_score = 0.0
    
    try:
        db_score = davies_bouldin_score(embeddings, cluster_labels)
    except Exception as e:
        print(f"[Warning] Davies-Bouldin calculation failed: {e}")
        db_score = float('inf')
    
    try:
        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
    except Exception as e:
        print(f"[Warning] Calinski-Harabasz calculation failed: {e}")
        ch_score = 0.0
    
    # Tính Dunn Index (thủ công vì sklearn không có)
    try:
        dunn = _compute_dunn_index(embeddings, cluster_labels)
    except Exception as e:
        print(f"[Warning] Dunn Index calculation failed: {e}")
        dunn = 0.0
    
    # Tính khoảng cách trung bình nội cụm
    try:
        intra_dist = _compute_mean_intra_cluster_distance(embeddings, cluster_labels)
    except:
        intra_dist = 0.0
    
    return {
        'silhouette': float(sil_score),
        'davies_bouldin': float(db_score),
        'calinski_harabasz': float(ch_score),
        'dunn_index': float(dunn),
        'mean_intra_cluster_distance': float(intra_dist),
        'num_samples': len(embeddings),
        'num_clusters': n_clusters
    }


def _compute_dunn_index(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Tính Dunn Index = min(inter-cluster distance) / max(intra-cluster diameter)
    
    Giá trị cao hơn tốt hơn (cụm gọn và phân tách tốt).
    """
    from sklearn.metrics.pairwise import cosine_distances
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # Tính ma trận khoảng cách
    dist_matrix = cosine_distances(embeddings)
    
    # Tính đường kính mỗi cụm (max intra-cluster distance)
    diameters = []
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) < 2:
            diameters.append(0.0)
        else:
            cluster_dists = dist_matrix[np.ix_(mask, mask)]
            diameters.append(np.max(cluster_dists))
    
    max_diameter = max(diameters) if diameters else 0.0
    
    if max_diameter == 0:
        return 0.0
    
    # Tính khoảng cách liên cụm nhỏ nhất
    min_inter_dist = float('inf')
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i >= j:
                continue
            mask_i = labels == label_i
            mask_j = labels == label_j
            inter_dists = dist_matrix[np.ix_(mask_i, mask_j)]
            min_inter_dist = min(min_inter_dist, np.min(inter_dists))
    
    return min_inter_dist / max_diameter


def _compute_mean_intra_cluster_distance(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Tính khoảng cách trung bình trong mỗi cụm."""
    from sklearn.metrics.pairwise import cosine_distances
    
    unique_labels = np.unique(labels)
    total_dist = 0.0
    total_pairs = 0
    
    for label in unique_labels:
        mask = labels == label
        cluster_emb = embeddings[mask]
        n = len(cluster_emb)
        if n < 2:
            continue
        dists = cosine_distances(cluster_emb)
        total_dist += np.sum(dists) / 2  # Chia 2 vì ma trận đối xứng
        total_pairs += n * (n - 1) / 2
    
    return total_dist / total_pairs if total_pairs > 0 else 0.0


def plot_umap_2d(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
    title: str = "UMAP Projection của Face Embeddings",
    cluster_names: Optional[Dict] = None
):
    """
    Vẽ biểu đồ UMAP 2D để trực quan hóa các embeddings.
    
    KHÔNG CẦN NHÃN - Chỉ sử dụng embeddings và cluster_id.
    
    Args:
        embeddings: Ma trận embeddings shape (n_samples, 512)
        cluster_labels: Nhãn cụm cho mỗi mẫu
        output_path: Đường dẫn lưu file PNG
        title: Tiêu đề biểu đồ
        cluster_names: Dict ánh xạ cluster_id -> tên hiển thị (tùy chọn)
    """
    try:
        import umap
    except ImportError:
        print("[Error] umap-learn not installed. Run: pip install umap-learn")
        return
    
    print(f"[Info] Computing UMAP projection for {len(embeddings)} embeddings...")
    
    # Cấu hình UMAP cho face embeddings
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    
    # Giảm chiều
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Đếm số lượng mỗi cụm để scale kích thước điểm
    unique_labels = np.unique(cluster_labels)
    cluster_sizes = {label: np.sum(cluster_labels == label) for label in unique_labels}
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color palette
    n_clusters = len(unique_labels)
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Vẽ từng cụm
    for idx, label in enumerate(unique_labels):
        mask = cluster_labels == label
        size = cluster_sizes[label]
        
        # Scale kích thước điểm lớn hơn (50-250) để dễ nhìn từ xa
        point_size = max(50, min(250, 30 + np.log1p(size) * 30))
        
        # Tên hiển thị
        if cluster_names and label in cluster_names:
            display_name = cluster_names[label]
        else:
            display_name = f"Cụm {idx}" if isinstance(label, (int, np.integer)) else str(label)[-8:]
        
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[idx]],
            s=point_size,
            label=f"{display_name} ({size})",
            alpha=0.75,
            edgecolors='black',
            linewidth=0.8
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend nếu không quá nhiều cụm
    if n_clusters <= 15:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=True,
            title="Cụm (số ảnh)"
        )
    else:
        # Hiển thị chú thích tổng quan
        ax.text(
            1.02, 0.5,
            f"Tổng: {n_clusters} cụm\n{len(embeddings)} ảnh",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='center'
        )
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] UMAP 2D projection saved to: {output_path}")


def plot_internal_metrics_bar(
    metrics: Dict[str, float],
    output_path: str,
    title: str = "Chỉ Số Đánh Giá Phân Cụm (Internal Metrics)"
):
    """
    Vẽ biểu đồ cột cho các internal metrics.
    
    KHÔNG CẦN NHÃN - Hiển thị Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn Index.
    
    Args:
        metrics: Dict từ compute_internal_metrics()
        output_path: Đường dẫn lưu file PNG
        title: Tiêu đề biểu đồ
    """
    # Chuẩn bị dữ liệu - normalize về 0-1 cho dễ so sánh
    metric_data = []
    
    # Silhouette: đã trong khoảng [-1, 1], normalize sang [0, 1]
    sil = metrics.get('silhouette', 0)
    sil_normalized = (sil + 1) / 2
    metric_data.append(('Silhouette\n(↑ tốt)', sil_normalized, sil, sil >= 0.3))
    
    # Davies-Bouldin: thấp hơn tốt hơn, nghịch đảo
    db = metrics.get('davies_bouldin', 2)
    db_normalized = 1 / (1 + db)  # Ánh xạ [0, inf) -> (0, 1]
    metric_data.append(('Davies-Bouldin\n(↓ tốt)', db_normalized, db, db < 1.5))
    
    # Calinski-Harabasz: cao hơn tốt hơn, normalize bằng log
    ch = metrics.get('calinski_harabasz', 0)
    ch_normalized = np.log1p(ch) / 10  # Chia 10 để scale về ~0-1
    ch_normalized = min(ch_normalized, 1.0)
    metric_data.append(('Calinski-Harabasz\n(↑ tốt)', ch_normalized, ch, ch > 50))
    
    # Dunn Index: cao hơn tốt hơn, đã trong khoảng hợp lý
    dunn = metrics.get('dunn_index', 0)
    dunn_normalized = min(dunn, 1.0)
    metric_data.append(('Dunn Index\n(↑ tốt)', dunn_normalized, dunn, dunn > 0.3))
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [d[0] for d in metric_data]
    values = [d[1] for d in metric_data]
    raw_values = [d[2] for d in metric_data]
    is_good = [d[3] for d in metric_data]
    
    # Màu theo chất lượng
    colors = ['#4CAF50' if good else '#FFC107' if v > 0.3 else '#F44336' 
              for v, good in zip(values, is_good)]
    
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Thêm giá trị thực trên thanh
    for bar, raw_val, norm_val in zip(bars, raw_values, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f'{raw_val:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Đường ngưỡng
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.4, label='Ngưỡng tốt')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.4, label='Ngưỡng chấp nhận')
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Giá trị (đã normalize)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Internal metrics bar chart saved to: {output_path}")


def plot_multi_video_metrics_comparison(
    video_metrics: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "So Sánh Chỉ Số Phân Cụm Giữa Các Video"
):
    """
    Vẽ biểu đồ cột so sánh metrics giữa nhiều video.
    
    Args:
        video_metrics: Dict[video_name, metrics_dict]
        output_path: Đường dẫn lưu file PNG
        title: Tiêu đề biểu đồ
    """
    video_names = list(video_metrics.keys())
    n_videos = len(video_names)
    
    if n_videos == 0:
        print("[Warning] No video metrics to plot")
        return
    
    # Metrics để so sánh
    metric_keys = ['silhouette', 'dunn_index']
    metric_labels = ['Silhouette', 'Dunn Index']
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_videos)
    width = 0.35
    
    colors = ['#2196F3', '#4CAF50']
    
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = []
        for video in video_names:
            val = video_metrics[video].get(key, 0)
            # Normalize silhouette từ [-1,1] sang [0,1]
            if key == 'silhouette':
                val = (val + 1) / 2
            values.append(val)
        
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i], alpha=0.8)
        
        # Thêm giá trị
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.2f}',
                ha='center',
                fontsize=9
            )
    
    ax.set_xlabel('Video', fontsize=11)
    ax.set_ylabel('Giá trị (normalized)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(video_names, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Multi-video metrics comparison saved to: {output_path}")


def plot_autotuning_comparison(
    metrics_with_tuning: Dict[str, float],
    metrics_without_tuning: Dict[str, float],
    video_name: str,
    output_path: str
):
    """
    Vẽ biểu đồ so sánh hiệu quả Auto-Tuning.
    
    So sánh các metrics KHI CÓ và KHÔNG CÓ auto-tuning.
    
    Args:
        metrics_with_tuning: Metrics khi bật auto-tuning
        metrics_without_tuning: Metrics khi tắt auto-tuning
        video_name: Tên video
        output_path: Đường dẫn lưu file PNG
    """
    metrics_keys = ['silhouette', 'dunn_index']
    metric_labels = ['Silhouette Score', 'Dunn Index']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_keys))
    width = 0.35
    
    # Lấy giá trị
    values_with = []
    values_without = []
    improvements = []
    
    for key in metrics_keys:
        v_with = metrics_with_tuning.get(key, 0)
        v_without = metrics_without_tuning.get(key, 0)
        
        # Normalize silhouette
        if key == 'silhouette':
            v_with = (v_with + 1) / 2
            v_without = (v_without + 1) / 2
        
        values_with.append(v_with)
        values_without.append(v_without)
        
        # Tính % cải thiện
        if v_without > 0:
            improvement = ((v_with - v_without) / v_without) * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    # Vẽ bars
    bars1 = ax.bar(x - width/2, values_without, width, label='Không Auto-Tuning', 
                   color='#FF9800', alpha=0.8)
    bars2 = ax.bar(x + width/2, values_with, width, label='Có Auto-Tuning',
                   color='#4CAF50', alpha=0.8)
    
    # Thêm giá trị và % cải thiện
    for i, (bar1, bar2, impr) in enumerate(zip(bars1, bars2, improvements)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                f'{values_without[i]:.3f}', ha='center', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                f'{values_with[i]:.3f}', ha='center', fontsize=10)
        
        # Annotation cho % cải thiện
        if impr != 0:
            color = 'green' if impr > 0 else 'red'
            ax.annotate(
                f'{impr:+.1f}%',
                xy=(x[i] + width/2, max(values_with[i], values_without[i]) + 0.08),
                fontsize=11,
                fontweight='bold',
                color=color,
                ha='center'
            )
    
    ax.set_ylabel('Giá trị (normalized)', fontsize=11)
    ax.set_title(f'Hiệu Quả Auto-Tuning - Video: {video_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Auto-tuning comparison chart saved to: {output_path}")


# =====================================================================
# CÁC HÀM GIỮ LẠI TỪ PHIÊN BẢN CŨ (cho phân tích có labeled data)
# =====================================================================

def plot_cluster_composition_matrix(
    cluster_labels: List[int],
    matched_actors: List[str],
    cluster_embeddings: dict,
    output_path: str,
    title: str = "Cluster Composition Matrix"
):
    """
    Tạo heatmap thành phần cụm showing actor distribution.
    
    CẦN DỮ LIỆU NHÃN - Chỉ chạy khi có labeled_faces.
    
    Args:
        cluster_labels: Cluster ID cho mỗi face
        matched_actors: Tên actor (hoặc "Khác") cho mỗi face
        cluster_embeddings: Không dùng, giữ để tương thích
        output_path: Đường dẫn lưu file PNG
        title: Tiêu đề biểu đồ
    """
    # Build composition matrix
    unique_clusters = sorted(set(cluster_labels))
    unique_actors = sorted(set(matched_actors))
    
    # Tạo tên cụm đơn giản
    def simplify_cluster_name(cluster_id):
        if isinstance(cluster_id, str) and '_' in cluster_id:
            parts = cluster_id.split('_')
            return parts[-1]
        return str(cluster_id)
    
    cluster_display_names = [f"Cụm_{simplify_cluster_name(c)}" for c in unique_clusters]
    
    # Đảm bảo "Khác" ở cuối
    if "Khác" in unique_actors:
        unique_actors.remove("Khác")
        unique_actors.append("Khác")
    
    # Map labels to indices
    cluster_to_idx = {label: i for i, label in enumerate(unique_clusters)}
    actor_to_idx = {label: i for i, label in enumerate(unique_actors)}
    
    # Initialize matrix
    matrix = np.zeros((len(unique_clusters), len(unique_actors)), dtype=int)
    
    # Fill matrix
    for cluster_label, actor_label in zip(cluster_labels, matched_actors):
        cluster_idx = cluster_to_idx[cluster_label]
        actor_idx = actor_to_idx[actor_label]
        matrix[cluster_idx, actor_idx] += 1
    
    # Create figure
    figsize = (max(10, len(unique_actors) * 1.2), max(6, len(unique_clusters) * 0.6))
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=unique_actors,
        yticklabels=cluster_display_names,
        cbar_kws={'label': 'Số ảnh'},
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    plt.title(f"Thành Phần Cụm ({len(cluster_labels)} ảnh)", fontsize=15, fontweight='bold')
    plt.xlabel('Diễn viên', fontsize=11)
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout(pad=1.5)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Cluster composition matrix saved to: {output_path}")


def plot_cluster_cohesion_chart(
    cluster_labels: List[int],
    cluster_embeddings: dict,
    output_path: str,
    title: str = "Cluster Cohesion Analysis"
):
    """
    Tạo bar chart hiển thị độ đồng nhất cho mỗi cụm.
    
    KHÔNG CẦN NHÃN - Đánh giá dựa trên embedding similarity.
    
    Args:
        cluster_labels: Cluster ID cho mỗi face
        cluster_embeddings: {cluster_id: [embeddings]}
        output_path: Đường dẫn lưu file PNG
        title: Tiêu đề biểu đồ
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    
    unique_clusters = sorted(set(cluster_labels))
    
    def simplify_cluster_name(cluster_id):
        if isinstance(cluster_id, str) and '_' in cluster_id:
            parts = cluster_id.split('_')
            return parts[-1]
        return str(cluster_id)
    
    cluster_display_names = [f"Cụm_{simplify_cluster_name(c)}" for c in unique_clusters]
    
    def calculate_cohesion(embeddings_list):
        if len(embeddings_list) < 2:
            return 1.0
        
        embeddings = np.array(embeddings_list)
        similarities = cos_sim(embeddings)
        n = len(embeddings)
        avg_sim = (similarities.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
        return (avg_sim + 1) / 2  # Convert to 0-1 scale
    
    cohesion_scores = []
    cluster_sizes = []
    
    for cluster_id in unique_clusters:
        if cluster_id in cluster_embeddings and len(cluster_embeddings[cluster_id]) > 0:
            cohesion = calculate_cohesion(cluster_embeddings[cluster_id])
            size = len(cluster_embeddings[cluster_id])
        else:
            cohesion = 0.0
            size = 0
        cohesion_scores.append(cohesion)
        cluster_sizes.append(size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(unique_clusters) * 0.4)))
    
    # Color by quality
    colors = []
    for score in cohesion_scores:
        if score >= 0.75:
            colors.append('#2ca02c')  # Green - Excellent
        elif score >= 0.60:
            colors.append('#8fce00')  # Light green - Good
        elif score >= 0.45:
            colors.append('#ffc107')  # Yellow - Fair
        else:
            colors.append('#ff5252')  # Red - Poor
    
    # Horizontal bar chart
    y_pos = np.arange(len(unique_clusters))
    bars = ax.barh(y_pos, cohesion_scores, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (score, size) in enumerate(zip(cohesion_scores, cluster_sizes)):
        ax.text(score + 0.02, i, f'{score:.2f}  ({size} ảnh)', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cluster_display_names, fontsize=10)
    ax.set_xlabel('Độ đồng nhất', fontsize=11)
    ax.set_ylabel('')
    ax.set_xlim(0, 1.1)
    ax.set_title('Độ Đồng Nhất Cụm', fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Xuất sắc (≥0.75)'),
        Patch(facecolor='#8fce00', label='Tốt (≥0.60)'),
        Patch(facecolor='#ffc107', label='Khá (≥0.45)'),
        Patch(facecolor='#ff5252', label='Kém (<0.45)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Quality summary
    excellent = sum(1 for s in cohesion_scores if s >= 0.75)
    good = sum(1 for s in cohesion_scores if 0.60 <= s < 0.75)
    fair = sum(1 for s in cohesion_scores if 0.45 <= s < 0.60)
    poor = sum(1 for s in cohesion_scores if s < 0.45)
    
    print(f"[Info] Cohesion chart saved to: {output_path}")
    print(f"[Info] Cohesion Quality: {excellent} Excellent, {good} Good, {fair} Fair, {poor} Poor")


def print_internal_metrics_summary(metrics: Dict[str, float], title: str = "INTERNAL CLUSTERING METRICS"):
    """
    In tóm tắt các internal metrics ra console.
    
    Args:
        metrics: Dict từ compute_internal_metrics()
        title: Tiêu đề hiển thị
    """
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    print(f"\n📈 Số liệu tổng quan:")
    print(f"   • Số mẫu: {metrics.get('num_samples', 'N/A')}")
    print(f"   • Số cụm: {metrics.get('num_clusters', 'N/A')}")
    
    print(f"\n📉 Chỉ số đánh giá nội tại:")
    
    # Silhouette
    sil = metrics.get('silhouette', 0)
    sil_emoji = "🟢" if sil > 0.5 else "🟡" if sil > 0.25 else "🔴"
    print(f"   {sil_emoji} Silhouette Score: {sil:.4f} (tốt > 0.5)")
    
    # Davies-Bouldin
    db = metrics.get('davies_bouldin', float('inf'))
    db_emoji = "🟢" if db < 1.0 else "🟡" if db < 2.0 else "🔴"
    print(f"   {db_emoji} Davies-Bouldin Index: {db:.4f} (tốt < 1.0)")
    
    # Calinski-Harabasz
    ch = metrics.get('calinski_harabasz', 0)
    ch_emoji = "🟢" if ch > 100 else "🟡" if ch > 50 else "🔴"
    print(f"   {ch_emoji} Calinski-Harabasz Index: {ch:.4f} (cao = tốt)")
    
    # Dunn Index
    dunn = metrics.get('dunn_index', 0)
    dunn_emoji = "🟢" if dunn > 0.5 else "🟡" if dunn > 0.2 else "🔴"
    print(f"   {dunn_emoji} Dunn Index: {dunn:.4f} (tốt > 0.5)")
    
    # Mean intra distance
    intra = metrics.get('mean_intra_cluster_distance', 0)
    print(f"   📏 Khoảng cách nội cụm TB: {intra:.4f} (thấp = tốt)")
    
    print(f"\n{'='*60}\n")
