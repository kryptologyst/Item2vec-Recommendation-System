"""Data generation and loading utilities."""

import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_data(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 10000,
    rating_range: Tuple[int, int] = (1, 5),
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic user-item interaction data.
    
    Args:
        n_users: Number of users to generate.
        n_items: Number of items to generate.
        n_interactions: Number of interactions to generate.
        rating_range: Range of rating values (min, max).
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (interactions_df, items_df, users_df).
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Generate user-item interactions with popularity bias
    interactions = []
    
    # Create popularity distribution (some items are more popular)
    item_popularity = np.random.power(2, n_items)
    item_popularity = item_popularity / item_popularity.sum()
    
    # Generate interactions
    for _ in range(n_interactions):
        user_id = f"user_{random.randint(1, n_users)}"
        item_id = np.random.choice(n_items, p=item_popularity)
        item_id = f"item_{item_id + 1}"
        
        # Generate rating with some noise
        base_rating = random.uniform(rating_range[0], rating_range[1])
        rating = max(rating_range[0], min(rating_range[1], int(base_rating + random.gauss(0, 0.5))))
        
        timestamp = random.randint(1000000000, 2000000000)  # Unix timestamp range
        
        interactions.append({
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "timestamp": timestamp,
        })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Generate item metadata
    items_data = []
    categories = ["electronics", "books", "clothing", "home", "sports", "beauty", "food"]
    
    for i in range(n_items):
        item_id = f"item_{i + 1}"
        category = random.choice(categories)
        price = random.uniform(10, 1000)
        
        items_data.append({
            "item_id": item_id,
            "title": f"Item {i + 1}",
            "category": category,
            "price": round(price, 2),
            "description": f"This is a {category} item with ID {item_id}",
        })
    
    items_df = pd.DataFrame(items_data)
    
    # Generate user metadata
    users_data = []
    age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    
    for i in range(n_users):
        user_id = f"user_{i + 1}"
        age_group = random.choice(age_groups)
        
        users_data.append({
            "user_id": user_id,
            "age_group": age_group,
            "location": f"City_{random.randint(1, 50)}",
        })
    
    users_df = pd.DataFrame(users_data)
    
    return interactions_df, items_df, users_df


def create_item_sequences(
    interactions_df: pd.DataFrame,
    window_size: int = 5,
    min_sequence_length: int = 2,
) -> List[List[str]]:
    """Create item sequences from user interactions for Item2vec training.
    
    Args:
        interactions_df: DataFrame with user-item interactions.
        window_size: Size of the context window.
        min_sequence_length: Minimum length of sequences to include.
        
    Returns:
        List of item sequences.
    """
    sequences = []
    
    # Group by user and sort by timestamp
    for user_id, user_interactions in interactions_df.groupby("user_id"):
        user_items = user_interactions.sort_values("timestamp")["item_id"].tolist()
        
        if len(user_items) >= min_sequence_length:
            # Create sequences using sliding window
            for i in range(len(user_items) - window_size + 1):
                sequence = user_items[i:i + window_size]
                sequences.append(sequence)
    
    return sequences


def encode_items(items: List[str]) -> Tuple[LabelEncoder, List[int]]:
    """Encode item IDs to integer indices.
    
    Args:
        items: List of item IDs.
        
    Returns:
        Tuple of (label_encoder, encoded_items).
    """
    encoder = LabelEncoder()
    encoded_items = encoder.fit_transform(items)
    return encoder, encoded_items.tolist()


def create_negative_samples(
    positive_pairs: List[Tuple[int, int]],
    all_items: List[int],
    num_negative: int = 5,
    random_seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """Create negative samples for training.
    
    Args:
        positive_pairs: List of (item, context) positive pairs.
        all_items: List of all item indices.
        num_negative: Number of negative samples per positive pair.
        random_seed: Random seed for reproducibility.
        
    Returns:
        List of (item, context, label) tuples where label is 1 for positive, 0 for negative.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    samples = []
    
    for item, context in positive_pairs:
        # Add positive sample
        samples.append((item, context, 1))
        
        # Add negative samples
        for _ in range(num_negative):
            # Sample negative context (different from positive context)
            negative_context = random.choice(all_items)
            while negative_context == context:
                negative_context = random.choice(all_items)
            
            samples.append((item, negative_context, 0))
    
    return samples
