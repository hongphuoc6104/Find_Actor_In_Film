"""
Visualization Module for Clustering Evaluation

Generates charts and reports for evaluation results.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter


def plot_confusion_matrix(
    true_labels: List[str],
    pred_labels: List[int],
    output_path: str,
    title: str = "Clustering Confusion Matrix"
):
    """
    Create confusion matrix heatmap.
    
    Rows = True classes (actors)
    Columns = Predicted clusters
    Cell value = number of images
    
    Args:
        true_labels: Ground truth labels (actor names)
        pred_labels: Predicted cluster IDs
        output_path: Path to save PNG
        title: Chart title
    """
    # Build confusion matrix
    unique_true = sorted(set(true_labels))
    unique_pred = sorted(set(pred_labels))
    
    # Map labels to indices
    true_to_idx = {label: i for i, label in enumerate(unique_true)}
    pred_to_idx = {label: i for i, label in enumerate(unique_pred)}
    
    # Initialize matrix
    matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=int)
    
    # Fill matrix
    for true_label, pred_label in zip(true_labels, pred_labels):
        true_idx = true_to_idx[true_label]
        pred_idx = pred_to_idx[pred_label]
        matrix[true_idx, pred_idx] += 1
    
    # Create figure
    figsize = (max(8, len(unique_pred) * 0.8), max(6, len(unique_true) * 0.6))
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=[f"C{i}" for i in unique_pred],
        yticklabels=unique_true,
        cbar_kws={'label': 'Image Count'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Cluster ID', fontsize=12)
    plt.ylabel('True Actor', fontsize=12)
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Confusion matrix saved to: {output_path}")


def plot_cluster_composition_matrix(
    cluster_labels: List[int],
    matched_actors: List[str],
    cluster_embeddings: dict,
    output_path: str,
    title: str = "Cluster Composition Matrix"
):
    """
    Create cluster composition heatmap showing actor distribution.
    
    Args:
        cluster_labels: Cluster ID for each face
        matched_actors: Actor name (or "Khác") for each face
        cluster_embeddings: Not used here, kept for compatibility
        output_path: Path to save PNG
        title: Chart title
    """
    # Build composition matrix
    unique_clusters = sorted(set(cluster_labels))
    unique_actors = sorted(set(matched_actors))
    
    # Create simplified cluster names: extract last number from cluster ID
    def simplify_cluster_name(cluster_id):
        # Extract number from cluster ID (e.g., "3_merged_1" -> "1")
        if isinstance(cluster_id, str) and '_' in cluster_id:
            parts = cluster_id.split('_')
            return parts[-1]  # Last part after underscore
        return str(cluster_id)
    
    cluster_display_names = [f"Cụm_{simplify_cluster_name(c)}" for c in unique_clusters]
    
    # Ensure "Khác" is last column
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
    
    # Plot heatmap with optimized styling
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=unique_actors,
        yticklabels=cluster_display_names,  # Use simplified names
        cbar_kws={'label': 'Số ảnh'},
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    plt.title(f"Thành Phần Cụm ({len(cluster_labels)} ảnh)", fontsize=15, fontweight='bold')
    plt.xlabel('Diễn viên', fontsize=11)
    plt.ylabel('')  # Remove Y label since yticks already say "Cụm X"
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)  # Horizontal like X-axis
    plt.tight_layout(pad=1.5)  # Reduce padding
    
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
    Create bar chart showing cohesion score for each cluster.
    Helps evaluate "Khác" clusters quality.
    
    Args:
        cluster_labels: Cluster ID for each face
        cluster_embeddings: {cluster_id: [embeddings]}
        output_path: Path to save PNG
        title: Chart title
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    unique_clusters = sorted(set(cluster_labels))
    
    # Create simplified cluster names
    def simplify_cluster_name(cluster_id):
        if isinstance(cluster_id, str) and '_' in cluster_id:
            parts = cluster_id.split('_')
            return parts[-1]
        return str(cluster_id)
    
    cluster_display_names = [f"Cụm_{simplify_cluster_name(c)}" for c in unique_clusters]
    
    # Calculate cohesion for each cluster
    def calculate_cohesion(embeddings_list):
        if len(embeddings_list) < 2:
            return 1.0
        
        embeddings = np.array(embeddings_list)
        similarities = cosine_similarity(embeddings)
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
    
    # Add value labels - bigger and bolder
    for i, (score, size) in enumerate(zip(cohesion_scores, cluster_sizes)):
        ax.text(score + 0.02, i, f'{score:.2f}  ({size} ảnh)', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cluster_display_names, fontsize=10)  # Use simplified names
    ax.set_xlabel('Độ đồng nhất', fontsize=11)
    ax.set_ylabel('')  # Remove since yticks say "Cụm_X"
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




def plot_metrics_comparison(
    metrics: Dict[str, float],
    output_path: str,
    title: str = "Clustering Evaluation Metrics"
):
    """
    Create bar chart comparing all metrics.
    
    Args:
        metrics: Dict from evaluate_clustering()
        output_path: Path to save PNG
        title: Chart title
    """
    # Select metrics to plot
    metric_names = ['Purity', 'NMI', 'ARI', 'BCubed\nPrecision', 'BCubed\nRecall', 'BCubed\nF1']
    metric_keys = ['purity', 'nmi', 'ari', 'bcubed_precision', 'bcubed_recall', 'bcubed_f1']
    values = [metrics[key] for key in metric_keys]
    
    # Color code by value
    colors = []
    for val in values:
        if val >= 0.8:
            colors.append('#4CAF50')  # Green
        elif val >= 0.6:
            colors.append('#FFC107')  # Yellow
        else:
            colors.append('#F44336')  # Red
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Formatting
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.3, label='Excellent (>0.8)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, label='Fair (>0.6)')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Info] Metrics chart saved to: {output_path}")


def generate_markdown_report(
    metrics: Dict[str, float],
    true_labels: List[str],
    pred_labels: List[int],
    output_path: str,
    additional_info: Dict = None
):
    """
    Generate detailed Markdown evaluation report.
    
    Args:
        metrics: Dict from evaluate_clustering()
        true_labels: Ground truth labels
        pred_labels: Predicted cluster IDs
        output_path: Path to save .md file
        additional_info: Optional dict with movie name, config, etc.
    """
    from datetime import datetime
    from utils.evaluation import get_metric_interpretation
    
    # Build cluster analysis
    cluster_analysis = analyze_clusters(true_labels, pred_labels)
    
    # Start report
    lines = []
    lines.append("# Clustering Evaluation Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if additional_info:
        lines.append(f"**Movie:** {additional_info.get('movie', 'N/A')}\n")
        lines.append(f"**Config Profile:** {additional_info.get('profile', 'N/A')}\n")
    
    lines.append("\n---\n")
    
    # Dataset Statistics
    lines.append("## 📁 Dataset Statistics\n")
    lines.append(f"- **Total Images:** {metrics['num_samples']}\n")
    lines.append(f"- **True Classes (Actors):** {metrics['num_true_classes']}\n")
    lines.append(f"- **Predicted Clusters:** {metrics['num_pred_clusters']}\n")
    lines.append("\n")
    
    # Metrics Summary
    lines.append("## 📊 Evaluation Metrics\n")
    lines.append("| Metric | Score | Interpretation |\n")
    lines.append("|--------|-------|----------------|\n")
    
    metric_display = [
        ('Purity', 'purity'),
        ('NMI', 'nmi'),
        ('ARI', 'ari'),
        ('BCubed Precision', 'bcubed_precision'),
        ('BCubed Recall', 'bcubed_recall'),
        ('BCubed F1', 'bcubed_f1'),
    ]
    
    for display_name, key in metric_display:
        value = metrics[key]
        interpretation, emoji = get_metric_interpretation(key, value)
        lines.append(f"| {display_name} | {value:.4f} | {emoji} {interpretation} |\n")
    
    lines.append("\n")
    
    # Detailed Metrics Explanation
    lines.append("## 📖 Metrics Explanation\n")
    lines.append("\n### Purity\n")
    lines.append(f"**Score:** {metrics['purity']:.4f}\n\n")
    lines.append("Measures whether each cluster contains mostly one actor. ")
    lines.append("Higher purity means clusters are \"pure\" (not mixing different actors).\n\n")
    
    lines.append("### NMI (Normalized Mutual Information)\n")
    lines.append(f"**Score:** {metrics['nmi']:.4f}\n\n")
    lines.append("Measures how much information the clustering shares with ground truth. ")
    lines.append("Standard metric in research papers. Not affected by number of clusters.\n\n")
    
    lines.append("### ARI (Adjusted Rand Index)\n")
    lines.append(f"**Score:** {metrics['ari']:.4f}\n\n")
    lines.append("Measures pairwise agreement, adjusted for random chance. ")
    lines.append("0 = random clustering, 1 = perfect match.\n\n")
    
    lines.append("### BCubed F1\n")
    lines.append(f"**Precision:** {metrics['bcubed_precision']:.4f} | ")
    lines.append(f"**Recall:** {metrics['bcubed_recall']:.4f} | ")
    lines.append(f"**F1:** {metrics['bcubed_f1']:.4f}\n\n")
    lines.append("- **Precision:** How pure are the clusters? (no mixing)\n")
    lines.append("- **Recall:** Are images of the same actor grouped together? (no splitting)\n")
    lines.append("- **F1:** Balance between precision and recall\n\n")
    
    # Cluster Analysis
    lines.append("## 🔍 Cluster Analysis\n")
    lines.append("| Cluster ID | Size | Dominant Actor | Purity | All Actors (count) |\n")
    lines.append("|------------|------|----------------|--------|--------------------|\n")
    
    for cluster_id in sorted(cluster_analysis.keys()):
        info = cluster_analysis[cluster_id]
        actors_str = ", ".join([f"{name} ({cnt})" for name, cnt in info['actor_counts'].most_common()])
        lines.append(f"| {cluster_id} | {info['size']} | {info['dominant_actor']} | ")
        lines.append(f"{info['purity']:.2f} | {actors_str} |\n")
    
    lines.append("\n")
    
    # Recommendations
    lines.append("## 💡 Tuning Recommendations\n")
    lines.append(generate_tuning_recommendations(metrics))
    
    # Visualizations
    lines.append("\n## 📈 Visualizations\n")
    lines.append("### Actor → Cluster Confusion Matrix\n")
    lines.append("Shows how test images are distributed across clusters:\n\n")
    lines.append("![Confusion Matrix](confusion_matrix.png)\n\n")
    lines.append("### Cluster → Actor Composition Matrix\n")
    lines.append("Shows cluster composition (including Khác/supporting actors):\n\n")
    lines.append("![Cluster Composition](cluster_composition.png)\n\n")
    lines.append("### Metrics Comparison\n\n")
    lines.append("![Metrics Comparison](metrics_chart.png)\n\n")
    
    # Footer
    lines.append("---\n")
    lines.append("*For more details on metrics, see `docs/EVALUATION_METRICS.md`*\n")
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding='utf-8')
    
    print(f"[Info] Evaluation report saved to: {output_path}")


def analyze_clusters(true_labels: List[str], pred_labels: List[int]) -> Dict:
    """
    Analyze each cluster's composition.
    
    Returns:
        Dict mapping cluster_id -> {
            'size': int,
            'dominant_actor': str,
            'purity': float,
            'actor_counts': Counter
        }
    """
    clusters = defaultdict(list)
    for true_label, pred_label in zip(true_labels, pred_labels):
        clusters[pred_label].append(true_label)
    
    analysis = {}
    for cluster_id, actor_list in clusters.items():
        actor_counts = Counter(actor_list)
        dominant_actor, dominant_count = actor_counts.most_common(1)[0]
        purity = dominant_count / len(actor_list)
        
        analysis[cluster_id] = {
            'size': len(actor_list),
            'dominant_actor': dominant_actor,
            'purity': purity,
            'actor_counts': actor_counts
        }
    
    return analysis


def generate_tuning_recommendations(metrics: Dict[str, float]) -> str:
    """
    Generate parameter tuning recommendations based on metrics.
    
    Args:
        metrics: Evaluation metrics dict
        
    Returns:
        Markdown text with recommendations
    """
    recommendations = []
    
    # Analyze patterns
    purity = metrics['purity']
    recall = metrics['bcubed_recall']
    precision = metrics['bcubed_precision']
    nmi = metrics['nmi']
    
    # Overall assessment
    if nmi >= 0.8 and purity >= 0.85:
        recommendations.append("✅ **Excellent clustering quality!** Current parameters are well-tuned.\n")
        return "".join(recommendations)
    
    # High purity but low recall -> Over-segmentation
    if purity > 0.8 and recall < 0.6:
        recommendations.append("### ⚠️ Over-Segmentation Detected\n")
        recommendations.append("Clusters are pure but actors are split into multiple clusters.\n\n")
        recommendations.append("**Suggested fixes:**\n")
        recommendations.append("- **Increase** `clustering.distance_threshold` (e.g., 0.85 → 0.90)\n")
        recommendations.append("- **Increase** `post_merge.distance_threshold` (e.g., 0.60 → 0.70)\n")
        recommendations.append("- **Decrease** `merge.within_movie_threshold` (e.g., 0.55 → 0.50)\n\n")
    
    # High recall but low precision -> Under-segmentation (merging different actors)
    if recall > 0.8 and precision < 0.6:
        recommendations.append("### ⚠️ Over-Merging Detected\n")
        recommendations.append("Different actors are being merged into the same cluster.\n\n")
        recommendations.append("**Suggested fixes:**\n")
        recommendations.append("- **Decrease** `clustering.distance_threshold` (e.g., 0.90 → 0.85)\n")
        recommendations.append("- **Decrease** `post_merge.distance_threshold` (e.g., 0.70 → 0.60)\n")
        recommendations.append("- **Increase** `merge.within_movie_threshold` (e.g., 0.50 → 0.55)\n\n")
    
    # Both low -> General quality issues
    if precision < 0.7 and recall < 0.7:
        recommendations.append("### ⚠️ General Quality Issues\n")
        recommendations.append("Both precision and recall are low. Consider:\n\n")
        recommendations.append("**Suggested fixes:**\n")
        recommendations.append("- Check if `quality_filters.min_score_hard_cutoff` is too low (allowing poor embeddings)\n")
        recommendations.append("- Review video quality (lighting, clarity) and apply appropriate auto-tuning rules\n")
        recommendations.append("- Ensure test dataset images are representative of video frames\n\n")
    
    # Low NMI specifically
    if nmi < 0.6:
        recommendations.append("### 📉 Low NMI Score\n")
        recommendations.append("Clustering structure doesn't match ground truth well.\n\n")
        recommendations.append("**Suggested fixes:**\n")
        recommendations.append("- Review all 3 merge stages (clustering, merge, post_merge) thresholds\n")
        recommendations.append("- Check if `filter_clusters.min_size` is removing important clusters\n\n")
    
    if not recommendations:
        recommendations.append("### 🔧 Moderate Quality\n")
        recommendations.append("Results are reasonable but can be improved. ")
        recommendations.append("Fine-tune thresholds based on precision/recall balance.\n")
    
    return "".join(recommendations)
