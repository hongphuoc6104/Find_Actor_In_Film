"""
Clustering Evaluation Metrics Module

This module implements 4 evaluation metrics for face clustering:
1. Purity - measures cluster homogeneity
2. NMI (Normalized Mutual Information) - measures information sharing
3. ARI (Adjusted Rand Index) - measures pairwise agreement
4. BCubed F1 - balanced precision and recall for clustering

All metrics require ground truth labels to compare against predicted clusters.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def compute_purity(true_labels: List[str], pred_labels: List[int]) -> float:
    """
    Compute Purity score.
    
    Purity = (1/N) × Σᵢ max_j |cluster_i ∩ class_j|
    
    Args:
        true_labels: Ground truth labels (e.g., actor names)
        pred_labels: Predicted cluster IDs
        
    Returns:
        Purity score in range [0, 1]. Higher is better.
        1.0 = perfect (each cluster contains only one class)
        0.0 = worst (random assignment)
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have same length")
    
    if len(true_labels) == 0:
        return 0.0
    
    # Group by predicted cluster
    clusters = defaultdict(list)
    for true_label, pred_label in zip(true_labels, pred_labels):
        clusters[pred_label].append(true_label)
    
    # For each cluster, count the most frequent true class
    correct_count = 0
    for cluster_id, true_classes in clusters.items():
        most_common_class_count = Counter(true_classes).most_common(1)[0][1]
        correct_count += most_common_class_count
    
    purity = correct_count / len(true_labels)
    return purity


def compute_nmi(true_labels: List[str], pred_labels: List[int]) -> float:
    """
    Compute Normalized Mutual Information (NMI).
    
    Uses sklearn's implementation with average_method='arithmetic'.
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted cluster IDs
        
    Returns:
        NMI score in range [0, 1]. Higher is better.
        1.0 = perfect clustering
        0.0 = random clustering
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have same length")
    
    if len(true_labels) == 0:
        return 0.0
    
    return normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')


def compute_ari(true_labels: List[str], pred_labels: List[int]) -> float:
    """
    Compute Adjusted Rand Index (ARI).
    
    Uses sklearn's implementation. Adjusts for random chance.
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted cluster IDs
        
    Returns:
        ARI score in range [-1, 1]. Higher is better.
        1.0 = perfect clustering
        0.0 = random clustering
        <0 = worse than random
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have same length")
    
    if len(true_labels) == 0:
        return 0.0
    
    return adjusted_rand_score(true_labels, pred_labels)


def compute_bcubed_f1(true_labels: List[str], pred_labels: List[int]) -> Dict[str, float]:
    """
    Compute BCubed Precision, Recall, and F1 score.
    
    BCubed metrics evaluate each item individually:
    - Precision: For each item, what fraction of items in its cluster share its true class?
    - Recall: For each item, what fraction of items with its true class are in its cluster?
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted cluster IDs
        
    Returns:
        Dict with keys: 'precision', 'recall', 'f1'
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have same length")
    
    if len(true_labels) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    n = len(true_labels)
    
    # Build lookup structures
    # cluster_members[cluster_id] = list of indices
    cluster_members = defaultdict(list)
    # class_members[true_label] = list of indices
    class_members = defaultdict(list)
    
    for idx, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
        cluster_members[pred_label].append(idx)
        class_members[true_label].append(idx)
    
    total_precision = 0.0
    total_recall = 0.0
    
    for idx in range(n):
        true_label = true_labels[idx]
        pred_label = pred_labels[idx]
        
        # Get all items in the same cluster
        same_cluster = set(cluster_members[pred_label])
        # Get all items with the same true class
        same_class = set(class_members[true_label])
        
        # Precision: |same_cluster ∩ same_class| / |same_cluster|
        precision_i = len(same_cluster & same_class) / len(same_cluster)
        total_precision += precision_i
        
        # Recall: |same_cluster ∩ same_class| / |same_class|
        recall_i = len(same_cluster & same_class) / len(same_class)
        total_recall += recall_i
    
    # Average over all items
    avg_precision = total_precision / n
    avg_recall = total_recall / n
    
    # F1 = harmonic mean
    if avg_precision + avg_recall > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1 = 0.0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1
    }


def evaluate_clustering(true_labels: List[str], pred_labels: List[int]) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.
    
    Args:
        true_labels: Ground truth labels (e.g., actor names)
        pred_labels: Predicted cluster IDs
        
    Returns:
        Dict with all metrics:
        {
            'purity': float,
            'nmi': float,
            'ari': float,
            'bcubed_precision': float,
            'bcubed_recall': float,
            'bcubed_f1': float,
            'num_samples': int,
            'num_true_classes': int,
            'num_pred_clusters': int
        }
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have same length")
    
    # Compute all metrics
    purity = compute_purity(true_labels, pred_labels)
    nmi = compute_nmi(true_labels, pred_labels)
    ari = compute_ari(true_labels, pred_labels)
    bcubed = compute_bcubed_f1(true_labels, pred_labels)
    
    # Compute statistics
    num_samples = len(true_labels)
    num_true_classes = len(set(true_labels)) if num_samples > 0 else 0
    num_pred_clusters = len(set(pred_labels)) if num_samples > 0 else 0
    
    return {
        'purity': purity,
        'nmi': nmi,
        'ari': ari,
        'bcubed_precision': bcubed['precision'],
        'bcubed_recall': bcubed['recall'],
        'bcubed_f1': bcubed['f1'],
        'num_samples': num_samples,
        'num_true_classes': num_true_classes,
        'num_pred_clusters': num_pred_clusters
    }


def get_metric_interpretation(metric_name: str, value: float) -> Tuple[str, str]:
    """
    Get interpretation and color code for a metric value.
    
    Args:
        metric_name: Name of metric
        value: Metric value
        
    Returns:
        Tuple of (interpretation_text, emoji_color)
        - interpretation: "Excellent", "Good", "Fair", "Poor"
        - emoji_color: "🟢", "🟡", "🔴"
    """
    # For ARI, handle negative values
    if metric_name == 'ari':
        if value >= 0.8:
            return "Excellent", "🟢"
        elif value >= 0.6:
            return "Good", "🟡"
        elif value >= 0.3:
            return "Fair", "🟡"
        elif value >= 0:
            return "Poor", "🔴"
        else:
            return "Worse than random", "🔴"
    
    # For other metrics (0-1 range)
    if value >= 0.85:
        return "Excellent", "🟢"
    elif value >= 0.70:
        return "Good", "🟢"
    elif value >= 0.60:
        return "Fair", "🟡"
    else:
        return "Poor", "🔴"


def print_metrics_summary(metrics: Dict[str, float], title: str = "CLUSTERING EVALUATION RESULTS"):
    """
    Print metrics with color-coded interpretation to console.
    
    Args:
        metrics: Dict from evaluate_clustering()
        title: Title to display
    """
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    
    # Dataset info
    print(f"\n📁 Dataset Statistics:")
    print(f"   • Samples: {metrics['num_samples']}")
    print(f"   • True Classes (Actors): {metrics['num_true_classes']}")
    print(f"   • Predicted Clusters: {metrics['num_pred_clusters']}")
    
    # Metrics
    print(f"\n📈 Evaluation Metrics:")
    
    metric_display = [
        ('Purity', 'purity'),
        ('NMI (Normalized Mutual Information)', 'nmi'),
        ('ARI (Adjusted Rand Index)', 'ari'),
        ('BCubed Precision', 'bcubed_precision'),
        ('BCubed Recall', 'bcubed_recall'),
        ('BCubed F1', 'bcubed_f1'),
    ]
    
    for display_name, key in metric_display:
        value = metrics[key]
        interpretation, color = get_metric_interpretation(key, value)
        print(f"   {color} {display_name:35s}: {value:6.4f}  ({interpretation})")
    
    print("\n" + "=" * 60)
    print("💡 Tip: Check warehouse/evaluation/report.md for detailed analysis")
    print("=" * 60 + "\n")
