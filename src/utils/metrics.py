"""
Metrics for evaluating book recommendation model.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_tag_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute metrics for multi-label tag prediction.
    
    Args:
        predictions: Predicted probabilities [batch_size, num_tags]
        targets: Ground truth binary labels [batch_size, num_tags]
        threshold: Threshold for converting probabilities to binary predictions
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Compute metrics
    metrics = {}
    
    # Micro-averaged metrics (treats each tag occurrence as a separate prediction)
    metrics['micro_precision'] = precision_score(targets.flatten(), binary_preds.flatten(), zero_division=0)
    metrics['micro_recall'] = recall_score(targets.flatten(), binary_preds.flatten(), zero_division=0)
    metrics['micro_f1'] = f1_score(targets.flatten(), binary_preds.flatten(), zero_division=0)
    
    # Macro-averaged metrics (averages across tags)
    try:
        metrics['macro_precision'] = precision_score(targets, binary_preds, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(targets, binary_preds, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(targets, binary_preds, average='macro', zero_division=0)
    except:
        metrics['macro_precision'] = 0.0
        metrics['macro_recall'] = 0.0
        metrics['macro_f1'] = 0.0
    
    # Sample-wise metrics (exact match and subset accuracy)
    metrics['exact_match'] = accuracy_score(targets, binary_preds)
    
    # Hamming loss (fraction of incorrect labels)
    metrics['hamming_loss'] = np.mean(targets != binary_preds)
    
    # Coverage error (how far we need to go in rank to cover all true labels)
    try:
        coverage_errors = []
        for i in range(len(targets)):
            if targets[i].sum() > 0:  # Skip samples with no true labels
                # Sort predictions in descending order
                sorted_indices = np.argsort(predictions[i])[::-1]
                # Find the position of the last true label
                true_label_positions = np.where(targets[i][sorted_indices] == 1)[0]
                if len(true_label_positions) > 0:
                    coverage_errors.append(true_label_positions[-1] + 1)
        
        if coverage_errors:
            metrics['coverage_error'] = np.mean(coverage_errors)
        else:
            metrics['coverage_error'] = 0.0
    except:
        metrics['coverage_error'] = 0.0
    
    # Average precision score
    try:
        ap_scores = []
        for i in range(targets.shape[1]):  # For each tag
            if targets[:, i].sum() > 0:  # Skip tags with no positive examples
                ap_score = average_precision_score(targets[:, i], predictions[:, i])
                ap_scores.append(ap_score)
        
        if ap_scores:
            metrics['mean_average_precision'] = np.mean(ap_scores)
        else:
            metrics['mean_average_precision'] = 0.0
    except:
        metrics['mean_average_precision'] = 0.0
    
    return metrics


def compute_embedding_metrics(embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                            labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for embedding similarity.
    
    Args:
        embeddings1: First set of embeddings [batch_size, embedding_dim]
        embeddings2: Second set of embeddings [batch_size, embedding_dim]
        labels: Similarity labels [batch_size] (1 for similar, 0 for dissimilar)
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(embeddings1, torch.Tensor):
        embeddings1 = embeddings1.cpu().numpy()
    if isinstance(embeddings2, torch.Tensor):
        embeddings2 = embeddings2.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Compute cosine similarities
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    metrics = {}
    
    # AUC-ROC for similarity prediction
    try:
        metrics['similarity_auc'] = roc_auc_score(labels, similarities)
    except:
        metrics['similarity_auc'] = 0.5
    
    # Average precision for similarity prediction
    try:
        metrics['similarity_ap'] = average_precision_score(labels, similarities)
    except:
        metrics['similarity_ap'] = 0.0
    
    # Correlation between similarities and labels
    try:
        correlation = np.corrcoef(similarities, labels)[0, 1]
        metrics['similarity_correlation'] = correlation if not np.isnan(correlation) else 0.0
    except:
        metrics['similarity_correlation'] = 0.0
    
    return metrics


def compute_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings."""
    # Normalize embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Avoid division by zero
    norm1 = np.where(norm1 == 0, 1, norm1)
    norm2 = np.where(norm2 == 0, 1, norm2)
    
    embeddings1_norm = embeddings1 / norm1
    embeddings2_norm = embeddings2 / norm2
    
    # Compute cosine similarity
    similarities = np.sum(embeddings1_norm * embeddings2_norm, axis=1)
    
    return similarities


def compute_recommendation_metrics(recommendations: List[List[str]], 
                                 ground_truth: List[List[str]], 
                                 k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute recommendation quality metrics.
    
    Args:
        recommendations: List of recommendation lists for each query
        ground_truth: List of relevant items for each query
        k_values: Values of k for computing precision@k and recall@k
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        
        for recs, truth in zip(recommendations, ground_truth):
            # Truncate recommendations to top-k
            recs_k = recs[:k]
            
            # Compute precision@k
            if len(recs_k) > 0:
                relevant_in_k = len(set(recs_k) & set(truth))
                precision_k = relevant_in_k / len(recs_k)
                precisions.append(precision_k)
            
            # Compute recall@k
            if len(truth) > 0:
                relevant_in_k = len(set(recs_k) & set(truth))
                recall_k = relevant_in_k / len(truth)
                recalls.append(recall_k)
        
        # Average across all queries
        if precisions:
            metrics[f'precision@{k}'] = np.mean(precisions)
        if recalls:
            metrics[f'recall@{k}'] = np.mean(recalls)
        
        # F1@k
        if f'precision@{k}' in metrics and f'recall@{k}' in metrics:
            p = metrics[f'precision@{k}']
            r = metrics[f'recall@{k}']
            if p + r > 0:
                metrics[f'f1@{k}'] = 2 * p * r / (p + r)
            else:
                metrics[f'f1@{k}'] = 0.0
    
    return metrics


def compute_diversity_metrics(recommendations: List[List[str]]) -> Dict[str, float]:
    """
    Compute diversity metrics for recommendations.
    
    Args:
        recommendations: List of recommendation lists
    
    Returns:
        Dictionary of diversity metrics
    """
    metrics = {}
    
    # Intra-list diversity (average pairwise distance within each recommendation list)
    intra_diversities = []
    for recs in recommendations:
        if len(recs) > 1:
            unique_items = list(set(recs))
            diversity = len(unique_items) / len(recs)  # Simplified diversity measure
            intra_diversities.append(diversity)
    
    if intra_diversities:
        metrics['intra_list_diversity'] = np.mean(intra_diversities)
    else:
        metrics['intra_list_diversity'] = 0.0
    
    # Coverage (fraction of unique items in all recommendations)
    all_recommended = set()
    total_recommendations = 0
    
    for recs in recommendations:
        all_recommended.update(recs)
        total_recommendations += len(recs)
    
    if total_recommendations > 0:
        metrics['coverage'] = len(all_recommended) / total_recommendations
    else:
        metrics['coverage'] = 0.0
    
    # Gini coefficient (measure of inequality in item popularity)
    item_counts = {}
    for recs in recommendations:
        for item in recs:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    if item_counts:
        counts = list(item_counts.values())
        counts.sort()
        n = len(counts)
        
        if n > 1:
            gini_numerator = sum((2 * i + 1) * count for i, count in enumerate(counts))
            gini = gini_numerator / (n * sum(counts)) - (n + 1) / n
            metrics['gini_coefficient'] = gini
        else:
            metrics['gini_coefficient'] = 0.0
    else:
        metrics['gini_coefficient'] = 0.0
    
    return metrics


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   metric_types: List[str] = ['tag']) -> Dict[str, float]:
    """
    Compute all relevant metrics based on prediction type.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        metric_types: Types of metrics to compute
    
    Returns:
        Combined dictionary of all metrics
    """
    all_metrics = {}
    
    if 'tag' in metric_types:
        tag_metrics = compute_tag_metrics(predictions, targets)
        all_metrics.update(tag_metrics)
    
    return all_metrics


class MetricsTracker:
    """Track metrics across training/validation."""
    
    def __init__(self):
        self.metrics_history = {}
    
    def update(self, metrics: Dict[str, float], step: int) -> None:
        """Update metrics for a given step."""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append((step, value))
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Tuple[int, float]:
        """Get best value for a metric."""
        if metric_name not in self.metrics_history:
            return 0, 0.0
        
        values = self.metrics_history[metric_name]
        if mode == 'max':
            best_step, best_value = max(values, key=lambda x: x[1])
        else:
            best_step, best_value = min(values, key=lambda x: x[1])
        
        return best_step, best_value
    
    def get_latest(self, metric_name: str) -> Tuple[int, float]:
        """Get latest value for a metric."""
        if metric_name not in self.metrics_history:
            return 0, 0.0
        
        return self.metrics_history[metric_name][-1]
    
    def get_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """Get full history for a metric."""
        return self.metrics_history.get(metric_name, [])