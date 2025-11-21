"""Evaluation metrics for recommendation systems."""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict


def precision_at_k(
    recommended_items: List[str],
    relevant_items: Set[str],
    k: int,
) -> float:
    """Calculate Precision@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        Precision@K score.
    """
    if k == 0:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    if not top_k_recommendations:
        return 0.0
    
    relevant_recommended = sum(1 for item in top_k_recommendations if item in relevant_items)
    return relevant_recommended / len(top_k_recommendations)


def recall_at_k(
    recommended_items: List[str],
    relevant_items: Set[str],
    k: int,
) -> float:
    """Calculate Recall@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        Recall@K score.
    """
    if not relevant_items:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    if not top_k_recommendations:
        return 0.0
    
    relevant_recommended = sum(1 for item in top_k_recommendations if item in relevant_items)
    return relevant_recommended / len(relevant_items)


def average_precision_at_k(
    recommended_items: List[str],
    relevant_items: Set[str],
    k: int,
) -> float:
    """Calculate Average Precision@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        Average Precision@K score.
    """
    if not relevant_items:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    if not top_k_recommendations:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, item in enumerate(top_k_recommendations):
        if item in relevant_items:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant_items) if relevant_items else 0.0


def mean_average_precision_at_k(
    user_recommendations: Dict[str, List[str]],
    user_relevant_items: Dict[str, Set[str]],
    k: int,
) -> float:
    """Calculate Mean Average Precision@K.
    
    Args:
        user_recommendations: Dictionary mapping user IDs to recommended item lists.
        user_relevant_items: Dictionary mapping user IDs to relevant item sets.
        k: Number of top recommendations to consider.
        
    Returns:
        Mean Average Precision@K score.
    """
    if not user_recommendations:
        return 0.0
    
    ap_scores = []
    for user_id in user_recommendations:
        if user_id in user_relevant_items:
            ap = average_precision_at_k(
                user_recommendations[user_id],
                user_relevant_items[user_id],
                k,
            )
            ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0


def ndcg_at_k(
    recommended_items: List[str],
    relevant_items: Set[str],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        NDCG@K score.
    """
    if k == 0:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    if not top_k_recommendations:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k_recommendations):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    recommended_items: List[str],
    relevant_items: Set[str],
    k: int,
) -> float:
    """Calculate Hit Rate@K.
    
    Args:
        recommended_items: List of recommended item IDs.
        relevant_items: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.
        
    Returns:
        Hit Rate@K score (1 if any relevant item is in top-k, 0 otherwise).
    """
    if not relevant_items:
        return 0.0
    
    top_k_recommendations = recommended_items[:k]
    return 1.0 if any(item in relevant_items for item in top_k_recommendations) else 0.0


def coverage_at_k(
    user_recommendations: Dict[str, List[str]],
    k: int,
) -> float:
    """Calculate Coverage@K (fraction of items that can be recommended).
    
    Args:
        user_recommendations: Dictionary mapping user IDs to recommended item lists.
        k: Number of top recommendations to consider.
        
    Returns:
        Coverage@K score.
    """
    if not user_recommendations:
        return 0.0
    
    all_items = set()
    recommended_items = set()
    
    for recommendations in user_recommendations.values():
        top_k = recommendations[:k]
        recommended_items.update(top_k)
    
    # Get all unique items from recommendations
    for recommendations in user_recommendations.values():
        all_items.update(recommendations)
    
    return len(recommended_items) / len(all_items) if all_items else 0.0


def diversity_at_k(
    user_recommendations: Dict[str, List[str]],
    item_similarity_matrix: Optional[np.ndarray] = None,
    item_ids: Optional[List[str]] = None,
    k: int = 10,
) -> float:
    """Calculate Diversity@K (average pairwise dissimilarity of recommendations).
    
    Args:
        user_recommendations: Dictionary mapping user IDs to recommended item lists.
        item_similarity_matrix: Item similarity matrix (optional).
        item_ids: List of item IDs corresponding to similarity matrix (optional).
        k: Number of top recommendations to consider.
        
    Returns:
        Diversity@K score.
    """
    if not user_recommendations:
        return 0.0
    
    diversity_scores = []
    
    for recommendations in user_recommendations.values():
        top_k = recommendations[:k]
        if len(top_k) < 2:
            continue
        
        if item_similarity_matrix is not None and item_ids is not None:
            # Use provided similarity matrix
            diversity = 0.0
            pairs = 0
            
            for i, item1 in enumerate(top_k):
                for j, item2 in enumerate(top_k[i+1:], i+1):
                    if item1 in item_ids and item2 in item_ids:
                        idx1 = item_ids.index(item1)
                        idx2 = item_ids.index(item2)
                        similarity = item_similarity_matrix[idx1, idx2]
                        diversity += 1 - similarity
                        pairs += 1
            
            if pairs > 0:
                diversity_scores.append(diversity / pairs)
        else:
            # Simple diversity: assume all items are dissimilar
            diversity_scores.append(1.0)
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def evaluate_model(
    model,
    test_data: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    user_column: str = "user_id",
    item_column: str = "item_id",
) -> Dict[str, Dict[str, float]]:
    """Evaluate a recommendation model.
    
    Args:
        model: Recommendation model with recommend() method.
        test_data: Test dataset with user-item interactions.
        k_values: List of k values to evaluate.
        user_column: Name of user column in test data.
        item_column: Name of item column in test data.
        
    Returns:
        Dictionary with evaluation results.
    """
    # Group test data by user
    user_relevant_items = {}
    for user_id, group in test_data.groupby(user_column):
        user_relevant_items[user_id] = set(group[item_column].tolist())
    
    # Get recommendations for all users
    user_recommendations = {}
    for user_id in user_relevant_items.keys():
        try:
            recommendations = model.recommend(user_id, top_k=max(k_values))
            user_recommendations[user_id] = recommendations
        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {e}")
            user_recommendations[user_id] = []
    
    # Calculate metrics
    results = {}
    
    for k in k_values:
        results[f"k_{k}"] = {
            "precision": 0.0,
            "recall": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
            "hit_rate": 0.0,
        }
        
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_rate_scores = []
        
        for user_id in user_relevant_items:
            if user_id in user_recommendations:
                recommendations = user_recommendations[user_id]
                relevant_items = user_relevant_items[user_id]
                
                precision_scores.append(precision_at_k(recommendations, relevant_items, k))
                recall_scores.append(recall_at_k(recommendations, relevant_items, k))
                ndcg_scores.append(ndcg_at_k(recommendations, relevant_items, k))
                hit_rate_scores.append(hit_rate_at_k(recommendations, relevant_items, k))
        
        results[f"k_{k}"]["precision"] = np.mean(precision_scores) if precision_scores else 0.0
        results[f"k_{k}"]["recall"] = np.mean(recall_scores) if recall_scores else 0.0
        results[f"k_{k}"]["ndcg"] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        results[f"k_{k}"]["hit_rate"] = np.mean(hit_rate_scores) if hit_rate_scores else 0.0
        
        # Calculate MAP
        results[f"k_{k}"]["map"] = mean_average_precision_at_k(
            user_recommendations,
            user_relevant_items,
            k,
        )
    
    # Calculate coverage
    coverage_scores = {}
    for k in k_values:
        coverage_scores[f"k_{k}"] = coverage_at_k(user_recommendations, k)
    
    results["coverage"] = coverage_scores
    
    return results
