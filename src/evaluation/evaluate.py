"""Evaluation utilities for recommendation system.

This module provides functions to compute various evaluation metrics
for recommendation systems, including RMSE and Precision@K.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict


def compute_rmse(predictions, targets):
    """Compute Root Mean Squared Error (RMSE).
    
    Args:
        predictions: Array or tensor of predicted ratings
        targets: Array or tensor of actual ratings
        
    Returns:
        rmse: Root Mean Squared Error value
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().detach().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().detach().numpy()
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    return rmse


def compute_mae(predictions, targets):
    """Compute Mean Absolute Error (MAE).
    
    Args:
        predictions: Array or tensor of predicted ratings
        targets: Array or tensor of actual ratings
        
    Returns:
        mae: Mean Absolute Error value
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().detach().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().detach().numpy()
    
    mae = mean_absolute_error(targets, predictions)
    return mae


def compute_precision_at_k(recommendations, ground_truth, k=10):
    """Compute Precision@K for recommendations.
    
    Precision@K measures the proportion of recommended items in the top-K
    that are relevant (present in the ground truth).
    
    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        ground_truth: Dict mapping user_id to list of relevant item_ids
        k: Number of top recommendations to consider (default: 10)
        
    Returns:
        precision: Average Precision@K across all users
    """
    precisions = []
    
    for user_id, rec_items in recommendations.items():
        if user_id not in ground_truth:
            continue
        
        # Get top-K recommendations
        top_k_recs = rec_items[:k]
        relevant_items = set(ground_truth[user_id])
        
        # Count hits in top-K
        hits = len(set(top_k_recs) & relevant_items)
        precision = hits / k if k > 0 else 0
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def compute_recall_at_k(recommendations, ground_truth, k=10):
    """Compute Recall@K for recommendations.
    
    Recall@K measures the proportion of relevant items that are successfully
    recommended in the top-K.
    
    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        ground_truth: Dict mapping user_id to list of relevant item_ids
        k: Number of top recommendations to consider (default: 10)
        
    Returns:
        recall: Average Recall@K across all users
    """
    recalls = []
    
    for user_id, rec_items in recommendations.items():
        if user_id not in ground_truth:
            continue
        
        # Get top-K recommendations
        top_k_recs = rec_items[:k]
        relevant_items = set(ground_truth[user_id])
        
        if len(relevant_items) == 0:
            continue
        
        # Count hits in top-K
        hits = len(set(top_k_recs) & relevant_items)
        recall = hits / len(relevant_items)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def compute_f1_at_k(recommendations, ground_truth, k=10):
    """Compute F1@K for recommendations.
    
    F1@K is the harmonic mean of Precision@K and Recall@K.
    
    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        ground_truth: Dict mapping user_id to list of relevant item_ids
        k: Number of top recommendations to consider (default: 10)
        
    Returns:
        f1: Average F1@K across all users
    """
    precision = compute_precision_at_k(recommendations, ground_truth, k)
    recall = compute_recall_at_k(recommendations, ground_truth, k)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_ndcg_at_k(recommendations, ground_truth, relevance_scores=None, k=10):
    """Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG@K measures the ranking quality of recommendations, giving more
    weight to relevant items appearing earlier in the recommendation list.
    
    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        ground_truth: Dict mapping user_id to list of relevant item_ids
        relevance_scores: Optional dict mapping (user_id, item_id) to relevance scores
        k: Number of top recommendations to consider (default: 10)
        
    Returns:
        ndcg: Average NDCG@K across all users
    """
    ndcg_scores = []
    
    for user_id, rec_items in recommendations.items():
        if user_id not in ground_truth:
            continue
        
        # Get top-K recommendations
        top_k_recs = rec_items[:k]
        relevant_items = set(ground_truth[user_id])
        
        # Compute DCG
        dcg = 0.0
        for i, item_id in enumerate(top_k_recs):
            if item_id in relevant_items:
                # Use relevance score if provided, otherwise binary relevance
                rel = relevance_scores.get((user_id, item_id), 1.0) if relevance_scores else 1.0
                dcg += rel / np.log2(i + 2)  # i+2 because index starts at 0
        
        # Compute IDCG (ideal DCG)
        if relevance_scores:
            sorted_relevance = sorted(
                [relevance_scores.get((user_id, item), 0.0) for item in relevant_items],
                reverse=True
            )[:k]
        else:
            sorted_relevance = [1.0] * min(len(relevant_items), k)
        
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevance))
        
        # Compute NDCG
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_hit_rate_at_k(recommendations, ground_truth, k=10):
    """Compute Hit Rate@K.
    
    Hit Rate@K measures the proportion of users for whom at least one
    relevant item appears in the top-K recommendations.
    
    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        ground_truth: Dict mapping user_id to list of relevant item_ids
        k: Number of top recommendations to consider (default: 10)
        
    Returns:
        hit_rate: Hit Rate@K
    """
    hits = 0
    total = 0
    
    for user_id, rec_items in recommendations.items():
        if user_id not in ground_truth:
            continue
        
        total += 1
        top_k_recs = set(rec_items[:k])
        relevant_items = set(ground_truth[user_id])
        
        # Check if there's any overlap
        if len(top_k_recs & relevant_items) > 0:
            hits += 1
    
    return hits / total if total > 0 else 0.0


def evaluate_model(model, test_data, k=10, metrics=['rmse', 'precision', 'recall']):
    """Evaluate a recommendation model on test data.
    
    Args:
        model: Trained recommendation model with predict() method
        test_data: DataFrame with columns ['user_id', 'item_id', 'rating']
        k: Number of top recommendations for ranking metrics (default: 10)
        metrics: List of metrics to compute (default: ['rmse', 'precision', 'recall'])
        
    Returns:
        results: Dictionary mapping metric names to values
    """
    results = {}
    
    # Get predictions and ground truth
    user_ids = test_data['user_id'].values
    item_ids = test_data['item_id'].values
    actual_ratings = test_data['rating'].values
    
    # For rating prediction metrics
    if 'rmse' in metrics or 'mae' in metrics:
        # Get predictions from model
        model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor(user_ids, dtype=torch.long)
            item_tensor = torch.tensor(item_ids, dtype=torch.long)
            predicted_ratings = model(user_tensor, item_tensor)
        
        if 'rmse' in metrics:
            results['rmse'] = compute_rmse(predicted_ratings, actual_ratings)
        if 'mae' in metrics:
            results['mae'] = compute_mae(predicted_ratings, actual_ratings)
    
    # For ranking metrics
    ranking_metrics = {'precision', 'recall', 'f1', 'ndcg', 'hit_rate'}
    if any(m in metrics for m in ranking_metrics):
        # Generate recommendations for each unique user
        unique_users = test_data['user_id'].unique()
        recommendations = {}
        ground_truth = defaultdict(list)
        
        # Build ground truth
        for _, row in test_data.iterrows():
            if row['rating'] >= 4.0:  # Consider ratings >= 4 as relevant
                ground_truth[row['user_id']].append(row['item_id'])
        
        # Generate recommendations
        for user_id in unique_users:
            rec_items = model.predict(user_id, n_items=k)
            recommendations[user_id] = [item_id for item_id, _ in rec_items]
        
        if 'precision' in metrics:
            results['precision@k'] = compute_precision_at_k(recommendations, ground_truth, k)
        if 'recall' in metrics:
            results['recall@k'] = compute_recall_at_k(recommendations, ground_truth, k)
        if 'f1' in metrics:
            results['f1@k'] = compute_f1_at_k(recommendations, ground_truth, k)
        if 'ndcg' in metrics:
            results['ndcg@k'] = compute_ndcg_at_k(recommendations, ground_truth, k=k)
        if 'hit_rate' in metrics:
            results['hit_rate@k'] = compute_hit_rate_at_k(recommendations, ground_truth, k)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully!")
    print("Available metrics: RMSE, MAE, Precision@K, Recall@K, F1@K, NDCG@K, Hit Rate@K")
    print("Example usage:")
    print("  results = evaluate_model(model, test_df, k=10, metrics=['rmse', 'precision', 'recall'])")
