"""Baseline recommendation models for comparison."""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class PopularityRecommender:
    """Popularity-based recommender system."""
    
    def __init__(self):
        """Initialize popularity recommender."""
        self.item_popularity = None
        self.item_ids = None
    
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit the popularity model.
        
        Args:
            interactions_df: DataFrame with user-item interactions.
        """
        # Calculate item popularity (number of interactions)
        self.item_popularity = interactions_df.groupby("item_id").size().sort_values(ascending=False)
        self.item_ids = self.item_popularity.index.tolist()
    
    def recommend(self, user_id: str, top_k: int = 10) -> List[str]:
        """Recommend items based on popularity.
        
        Args:
            user_id: User ID to recommend for.
            top_k: Number of recommendations.
            
        Returns:
            List of recommended item IDs.
        """
        if self.item_popularity is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        return self.item_ids[:top_k]
    
    def get_item_scores(self, user_id: str) -> Dict[str, float]:
        """Get popularity scores for all items.
        
        Args:
            user_id: User ID.
            
        Returns:
            Dictionary mapping item IDs to popularity scores.
        """
        if self.item_popularity is None:
            raise ValueError("Model must be fitted before getting scores")
        
        # Normalize popularity scores
        max_popularity = self.item_popularity.max()
        scores = {}
        
        for item_id in self.item_ids:
            scores[item_id] = self.item_popularity[item_id] / max_popularity
        
        return scores


class UserKNNRecommender:
    """User-based collaborative filtering using k-nearest neighbors."""
    
    def __init__(self, k: int = 50, metric: str = "cosine"):
        """Initialize user-kNN recommender.
        
        Args:
            k: Number of nearest neighbors.
            metric: Distance metric for kNN.
        """
        self.k = k
        self.metric = metric
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.knn_model = None
    
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit the user-kNN model.
        
        Args:
            interactions_df: DataFrame with user-item interactions.
        """
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # Fit kNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.k, len(self.user_ids)),
            metric=self.metric,
        )
        self.knn_model.fit(self.user_item_matrix.values)
    
    def recommend(self, user_id: str, top_k: int = 10) -> List[str]:
        """Recommend items for a user.
        
        Args:
            user_id: User ID to recommend for.
            top_k: Number of recommendations.
            
        Returns:
            List of recommended item IDs.
        """
        if self.knn_model is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_ids:
            # Return popular items for new users
            popularity = self.user_item_matrix.sum().sort_values(ascending=False)
            return popularity.head(top_k).index.tolist()
        
        # Get user vector
        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(user_vector)
        
        # Get recommendations from similar users
        similar_users = [self.user_ids[idx] for idx in indices[0][1:]]  # Exclude self
        
        # Calculate weighted scores
        scores = {}
        for similar_user in similar_users:
            similar_user_idx = self.user_ids.index(similar_user)
            similar_user_vector = self.user_item_matrix.iloc[similar_user_idx]
            
            for item_id in self.item_ids:
                if item_id not in scores:
                    scores[item_id] = 0
                scores[item_id] += similar_user_vector[item_id]
        
        # Sort by scores and return top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:top_k]]
    
    def get_item_scores(self, user_id: str) -> Dict[str, float]:
        """Get recommendation scores for all items.
        
        Args:
            user_id: User ID.
            
        Returns:
            Dictionary mapping item IDs to recommendation scores.
        """
        if self.knn_model is None:
            raise ValueError("Model must be fitted before getting scores")
        
        if user_id not in self.user_ids:
            # Return popularity scores for new users
            popularity = self.user_item_matrix.sum()
            max_popularity = popularity.max()
            return {item_id: popularity[item_id] / max_popularity for item_id in self.item_ids}
        
        # Get user vector
        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(user_vector)
        
        # Get scores from similar users
        similar_users = [self.user_ids[idx] for idx in indices[0][1:]]  # Exclude self
        
        scores = {}
        for similar_user in similar_users:
            similar_user_idx = self.user_ids.index(similar_user)
            similar_user_vector = self.user_item_matrix.iloc[similar_user_idx]
            
            for item_id in self.item_ids:
                if item_id not in scores:
                    scores[item_id] = 0
                scores[item_id] += similar_user_vector[item_id]
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1
        return {item_id: score / max_score for item_id, score in scores.items()}


class ItemKNNRecommender:
    """Item-based collaborative filtering using k-nearest neighbors."""
    
    def __init__(self, k: int = 50, metric: str = "cosine"):
        """Initialize item-kNN recommender.
        
        Args:
            k: Number of nearest neighbors.
            metric: Distance metric for kNN.
        """
        self.k = k
        self.metric = metric
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.knn_model = None
    
    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Fit the item-kNN model.
        
        Args:
            interactions_df: DataFrame with user-item interactions.
        """
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # Fit kNN model on items
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.k, len(self.item_ids)),
            metric=self.metric,
        )
        self.knn_model.fit(self.user_item_matrix.T.values)  # Transpose for item-based
    
    def recommend(self, user_id: str, top_k: int = 10) -> List[str]:
        """Recommend items for a user.
        
        Args:
            user_id: User ID to recommend for.
            top_k: Number of recommendations.
            
        Returns:
            List of recommended item IDs.
        """
        if self.knn_model is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_ids:
            # Return popular items for new users
            popularity = self.user_item_matrix.sum().sort_values(ascending=False)
            return popularity.head(top_k).index.tolist()
        
        # Get user's interacted items
        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx]
        interacted_items = user_vector[user_vector > 0].index.tolist()
        
        if not interacted_items:
            # Return popular items for users with no interactions
            popularity = self.user_item_matrix.sum().sort_values(ascending=False)
            return popularity.head(top_k).index.tolist()
        
        # Calculate scores based on similar items
        scores = {}
        for item_id in self.item_ids:
            if item_id in interacted_items:
                continue  # Skip already interacted items
            
            score = 0
            for interacted_item in interacted_items:
                # Find similar items to the interacted item
                interacted_item_idx = self.item_ids.index(interacted_item)
                interacted_item_vector = self.user_item_matrix.T.iloc[interacted_item_idx].values.reshape(1, -1)
                
                distances, indices = self.knn_model.kneighbors(interacted_item_vector)
                
                # Check if current item is among similar items
                similar_items = [self.item_ids[idx] for idx in indices[0]]
                if item_id in similar_items:
                    # Get similarity score
                    item_idx = similar_items.index(item_id)
                    similarity = 1 - distances[0][item_idx]  # Convert distance to similarity
                    score += similarity * user_vector[interacted_item]
            
            scores[item_id] = score
        
        # Sort by scores and return top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:top_k]]
    
    def get_item_scores(self, user_id: str) -> Dict[str, float]:
        """Get recommendation scores for all items.
        
        Args:
            user_id: User ID.
            
        Returns:
            Dictionary mapping item IDs to recommendation scores.
        """
        if self.knn_model is None:
            raise ValueError("Model must be fitted before getting scores")
        
        if user_id not in self.user_ids:
            # Return popularity scores for new users
            popularity = self.user_item_matrix.sum()
            max_popularity = popularity.max()
            return {item_id: popularity[item_id] / max_popularity for item_id in self.item_ids}
        
        # Get user's interacted items
        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_item_matrix.iloc[user_idx]
        interacted_items = user_vector[user_vector > 0].index.tolist()
        
        scores = {}
        for item_id in self.item_ids:
            score = 0
            for interacted_item in interacted_items:
                # Find similar items to the interacted item
                interacted_item_idx = self.item_ids.index(interacted_item)
                interacted_item_vector = self.user_item_matrix.T.iloc[interacted_item_idx].values.reshape(1, -1)
                
                distances, indices = self.knn_model.kneighbors(interacted_item_vector)
                
                # Check if current item is among similar items
                similar_items = [self.item_ids[idx] for idx in indices[0]]
                if item_id in similar_items:
                    # Get similarity score
                    item_idx = similar_items.index(item_id)
                    similarity = 1 - distances[0][item_idx]  # Convert distance to similarity
                    score += similarity * user_vector[interacted_item]
            
            scores[item_id] = score
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1
        return {item_id: score / max_score for item_id, score in scores.items()}
