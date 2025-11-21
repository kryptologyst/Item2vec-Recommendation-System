"""Unit tests for Item2vec recommendation system."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

from src.models.item2vec import Item2Vec, Item2VecDataset
from src.models.baselines import PopularityRecommender, UserKNNRecommender, ItemKNNRecommender
from src.data.data_utils import generate_synthetic_data, create_item_sequences, encode_items, create_negative_samples
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k, hit_rate_at_k
)
from src.utils.seed import set_seed


class TestItem2Vec:
    """Test cases for Item2vec model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = Item2Vec(vocab_size=100, embedding_dim=32)
        
        assert model.vocab_size == 100
        assert model.embedding_dim == 32
        assert model.item_embeddings.weight.shape == (100, 32)
        assert model.context_embeddings.weight.shape == (100, 32)
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = Item2Vec(vocab_size=100, embedding_dim=32)
        
        item_ids = torch.tensor([0, 1, 2])
        context_ids = torch.tensor([1, 2, 0])
        
        scores = model.forward(item_ids, context_ids)
        
        assert scores.shape == (3,)
        assert torch.all(torch.isfinite(scores))
    
    def test_get_similar_items(self):
        """Test getting similar items."""
        model = Item2Vec(vocab_size=10, embedding_dim=16)
        
        similar_items = model.get_similar_items(item_id=0, top_k=5)
        
        assert len(similar_items) == 5
        assert all(isinstance(item_id, int) for item_id, _ in similar_items)
        assert all(isinstance(score, float) for _, score in similar_items)
        assert all(0 <= item_id < 10 for item_id, _ in similar_items)
    
    def test_dataset(self):
        """Test Item2VecDataset."""
        samples = [(0, 1, 1), (1, 2, 0), (2, 0, 1)]
        dataset = Item2VecDataset(samples)
        
        assert len(dataset) == 3
        
        item, context, label = dataset[0]
        assert item.item() == 0
        assert context.item() == 1
        assert label.item() == 1.0


class TestBaselineModels:
    """Test cases for baseline models."""
    
    def setup_method(self):
        """Set up test data."""
        self.interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user2", "user2", "user3"],
            "item_id": ["item1", "item2", "item1", "item3", "item2"],
            "rating": [5, 4, 3, 5, 4],
            "timestamp": [1, 2, 3, 4, 5]
        })
    
    def test_popularity_recommender(self):
        """Test popularity recommender."""
        model = PopularityRecommender()
        model.fit(self.interactions_df)
        
        recommendations = model.recommend("user1", top_k=2)
        assert len(recommendations) == 2
        assert all(isinstance(item, str) for item in recommendations)
        
        scores = model.get_item_scores("user1")
        assert isinstance(scores, dict)
        assert all(isinstance(score, float) for score in scores.values())
    
    def test_user_knn_recommender(self):
        """Test user-kNN recommender."""
        model = UserKNNRecommender(k=2)
        model.fit(self.interactions_df)
        
        recommendations = model.recommend("user1", top_k=2)
        assert len(recommendations) <= 2
        assert all(isinstance(item, str) for item in recommendations)
    
    def test_item_knn_recommender(self):
        """Test item-kNN recommender."""
        model = ItemKNNRecommender(k=2)
        model.fit(self.interactions_df)
        
        recommendations = model.recommend("user1", top_k=2)
        assert len(recommendations) <= 2
        assert all(isinstance(item, str) for item in recommendations)


class TestDataUtils:
    """Test cases for data utilities."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        interactions_df, items_df, users_df = generate_synthetic_data(
            n_users=10,
            n_items=20,
            n_interactions=50,
            random_seed=42
        )
        
        assert len(interactions_df) == 50
        assert len(items_df) == 20
        assert len(users_df) == 10
        
        assert "user_id" in interactions_df.columns
        assert "item_id" in interactions_df.columns
        assert "rating" in interactions_df.columns
        assert "timestamp" in interactions_df.columns
        
        assert "item_id" in items_df.columns
        assert "title" in items_df.columns
        assert "category" in items_df.columns
        
        assert "user_id" in users_df.columns
        assert "age_group" in users_df.columns
    
    def test_create_item_sequences(self):
        """Test sequence creation."""
        interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user1", "user2", "user2"],
            "item_id": ["item1", "item2", "item3", "item1", "item2"],
            "timestamp": [1, 2, 3, 1, 2]
        })
        
        sequences = create_item_sequences(interactions_df, window_size=2)
        
        assert len(sequences) > 0
        assert all(isinstance(seq, list) for seq in sequences)
        assert all(len(seq) == 2 for seq in sequences)
    
    def test_encode_items(self):
        """Test item encoding."""
        items = ["item1", "item2", "item3"]
        encoder, encoded_items = encode_items(items)
        
        assert len(encoded_items) == 3
        assert all(isinstance(item, int) for item in encoded_items)
        assert set(encoded_items) == {0, 1, 2}
        
        # Test inverse transform
        decoded_items = encoder.inverse_transform(encoded_items)
        assert list(decoded_items) == items
    
    def test_create_negative_samples(self):
        """Test negative sample creation."""
        positive_pairs = [(0, 1), (1, 2), (2, 0)]
        all_items = [0, 1, 2]
        
        samples = create_negative_samples(positive_pairs, all_items, num_negative=2)
        
        assert len(samples) == len(positive_pairs) * 3  # 1 positive + 2 negative per pair
        assert all(len(sample) == 3 for sample in samples)  # (item, context, label)
        assert all(sample[2] in [0, 1] for sample in samples)  # label is 0 or 1


class TestEvaluationMetrics:
    """Test cases for evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommended_items = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = {"item1", "item3", "item5"}
        
        precision = precision_at_k(recommended_items, relevant_items, k=5)
        assert precision == 0.6  # 3 relevant out of 5 recommended
        
        precision = precision_at_k(recommended_items, relevant_items, k=3)
        assert precision == 2/3  # 2 relevant out of 3 recommended
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommended_items = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = {"item1", "item3", "item5", "item6"}
        
        recall = recall_at_k(recommended_items, relevant_items, k=5)
        assert recall == 0.75  # 3 relevant found out of 4 total relevant
        
        recall = recall_at_k(recommended_items, relevant_items, k=3)
        assert recall == 0.5  # 2 relevant found out of 4 total relevant
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommended_items = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = {"item1", "item3", "item5"}
        
        ndcg = ndcg_at_k(recommended_items, relevant_items, k=5)
        assert 0 <= ndcg <= 1
        
        # Perfect ranking should give NDCG = 1
        perfect_recommendations = ["item1", "item3", "item5", "item2", "item4"]
        perfect_ndcg = ndcg_at_k(perfect_recommendations, relevant_items, k=5)
        assert perfect_ndcg == 1.0
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        recommended_items = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = {"item1", "item6"}
        
        hit_rate = hit_rate_at_k(recommended_items, relevant_items, k=5)
        assert hit_rate == 1.0  # At least one relevant item in top-5
        
        hit_rate = hit_rate_at_k(recommended_items, relevant_items, k=1)
        assert hit_rate == 1.0  # Relevant item is first
        
        hit_rate = hit_rate_at_k(["item2", "item3"], relevant_items, k=2)
        assert hit_rate == 0.0  # No relevant items in top-2


class TestSeedUtils:
    """Test cases for seed utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy
        np_val1 = np.random.random()
        set_seed(42)
        np_val2 = np.random.random()
        assert np_val1 == np_val2
        
        # Test torch
        torch_val1 = torch.rand(1)
        set_seed(42)
        torch_val2 = torch.rand(1)
        assert torch.allclose(torch_val1, torch_val2)


if __name__ == "__main__":
    pytest.main([__file__])
