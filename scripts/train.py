"""Main training script for Item2vec recommendation system."""

import os
import yaml
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

from src.data.data_utils import (
    generate_synthetic_data,
    create_item_sequences,
    encode_items,
    create_negative_samples,
)
from src.models.item2vec import Item2Vec
from src.models.baselines import (
    PopularityRecommender,
    UserKNNRecommender,
    ItemKNNRecommender,
)
from src.evaluation.metrics import evaluate_model
from src.utils.seed import set_seed, get_device


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare training data.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (interactions_df, items_df, users_df).
    """
    print("Generating synthetic data...")
    
    # Generate synthetic data
    interactions_df, items_df, users_df = generate_synthetic_data(
        n_users=config["data"].get("n_users", 1000),
        n_items=config["data"].get("n_items", 500),
        n_interactions=config["data"].get("n_interactions", 10000),
        random_seed=config["data"]["random_seed"],
    )
    
    # Save data
    os.makedirs("data", exist_ok=True)
    interactions_df.to_csv("data/interactions.csv", index=False)
    items_df.to_csv("data/items.csv", index=False)
    users_df.to_csv("data/users.csv", index=False)
    
    print(f"Generated {len(interactions_df)} interactions for {len(users_df)} users and {len(items_df)} items")
    
    return interactions_df, items_df, users_df


def split_data(
    interactions_df: pd.DataFrame,
    config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test sets.
    
    Args:
        interactions_df: Interactions DataFrame.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    print("Splitting data into train/validation/test sets...")
    
    # Sort by timestamp for temporal split
    interactions_df = interactions_df.sort_values("timestamp")
    
    # Split by users to ensure each user appears in only one set
    users = interactions_df["user_id"].unique()
    
    train_users, temp_users = train_test_split(
        users,
        test_size=config["data"]["val_split"] + config["data"]["test_split"],
        random_state=config["data"]["random_seed"],
    )
    
    val_users, test_users = train_test_split(
        temp_users,
        test_size=config["data"]["test_split"] / (config["data"]["val_split"] + config["data"]["test_split"]),
        random_state=config["data"]["random_seed"],
    )
    
    train_df = interactions_df[interactions_df["user_id"].isin(train_users)]
    val_df = interactions_df[interactions_df["user_id"].isin(val_users)]
    test_df = interactions_df[interactions_df["user_id"].isin(test_users)]
    
    print(f"Train: {len(train_df)} interactions, {len(train_users)} users")
    print(f"Validation: {len(val_df)} interactions, {len(val_users)} users")
    print(f"Test: {len(test_df)} interactions, {len(test_users)} users")
    
    return train_df, val_df, test_df


def train_item2vec(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict,
) -> Tuple[Item2Vec, LabelEncoder]:
    """Train Item2vec model.
    
    Args:
        train_df: Training data.
        val_df: Validation data.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (trained_model, item_encoder).
    """
    print("Training Item2vec model...")
    
    # Create item sequences
    train_sequences = create_item_sequences(
        train_df,
        window_size=config["model"]["window_size"],
    )
    
    val_sequences = create_item_sequences(
        val_df,
        window_size=config["model"]["window_size"],
    )
    
    print(f"Created {len(train_sequences)} training sequences and {len(val_sequences)} validation sequences")
    
    # Encode items
    all_items = list(set([item for seq in train_sequences for item in seq]))
    item_encoder, encoded_items = encode_items(all_items)
    
    print(f"Encoded {len(all_items)} unique items")
    
    # Create positive pairs from sequences
    train_positive_pairs = []
    for sequence in train_sequences:
        encoded_seq = [item_encoder.transform([item])[0] for item in sequence]
        for i in range(len(encoded_seq) - 1):
            train_positive_pairs.append((encoded_seq[i], encoded_seq[i + 1]))
    
    val_positive_pairs = []
    for sequence in val_sequences:
        encoded_seq = [item_encoder.transform([item])[0] for item in sequence]
        for i in range(len(encoded_seq) - 1):
            val_positive_pairs.append((encoded_seq[i], encoded_seq[i + 1]))
    
    # Create negative samples
    train_samples = create_negative_samples(
        train_positive_pairs,
        encoded_items,
        num_negative=config["model"]["negative_samples"],
        random_seed=config["data"]["random_seed"],
    )
    
    val_samples = create_negative_samples(
        val_positive_pairs,
        encoded_items,
        num_negative=config["model"]["negative_samples"],
        random_seed=config["data"]["random_seed"],
    )
    
    print(f"Created {len(train_samples)} training samples and {len(val_samples)} validation samples")
    
    # Initialize model
    device = get_device(config["training"]["device"])
    model = Item2Vec(
        vocab_size=len(all_items),
        embedding_dim=config["model"]["embedding_dim"],
        negative_samples=config["model"]["negative_samples"],
        device=device,
    )
    
    # Train model
    history = model.train_model(
        train_samples=train_samples,
        val_samples=val_samples,
        num_epochs=config["model"]["num_epochs"],
        batch_size=config["model"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        verbose=True,
    )
    
    print("Item2vec training completed!")
    
    return model, item_encoder


def train_baselines(
    train_df: pd.DataFrame,
    config: Dict,
) -> Dict[str, any]:
    """Train baseline models.
    
    Args:
        train_df: Training data.
        config: Configuration dictionary.
        
    Returns:
        Dictionary of trained baseline models.
    """
    print("Training baseline models...")
    
    models = {}
    
    # Popularity recommender
    print("Training popularity recommender...")
    models["popularity"] = PopularityRecommender()
    models["popularity"].fit(train_df)
    
    # User-kNN recommender
    print("Training user-kNN recommender...")
    models["user_knn"] = UserKNNRecommender(k=50)
    models["user_knn"].fit(train_df)
    
    # Item-kNN recommender
    print("Training item-kNN recommender...")
    models["item_knn"] = ItemKNNRecommender(k=50)
    models["item_knn"].fit(train_df)
    
    print("Baseline training completed!")
    
    return models


def evaluate_models(
    models: Dict[str, any],
    test_df: pd.DataFrame,
    config: Dict,
) -> Dict[str, Dict]:
    """Evaluate all models.
    
    Args:
        models: Dictionary of trained models.
        test_df: Test data.
        config: Configuration dictionary.
        
    Returns:
        Dictionary with evaluation results.
    """
    print("Evaluating models...")
    
    k_values = config["evaluation"]["k_values"]
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        results[model_name] = evaluate_model(
            model=model,
            test_data=test_df,
            k_values=k_values,
        )
    
    return results


def print_results(results: Dict[str, Dict]) -> None:
    """Print evaluation results in a formatted table.
    
    Args:
        results: Evaluation results dictionary.
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Get all k values
    k_values = []
    for model_results in results.values():
        for key in model_results.keys():
            if key.startswith("k_"):
                k_values.append(int(key.split("_")[1]))
    k_values = sorted(set(k_values))
    
    # Print header
    header = f"{'Model':<15}"
    for k in k_values:
        header += f"{'P@' + str(k):<8}{'R@' + str(k):<8}{'MAP@' + str(k):<8}{'NDCG@' + str(k):<8}{'HR@' + str(k):<8}"
    print(header)
    print("-" * len(header))
    
    # Print results for each model
    for model_name, model_results in results.items():
        row = f"{model_name:<15}"
        for k in k_values:
            k_key = f"k_{k}"
            if k_key in model_results:
                metrics = model_results[k_key]
                row += f"{metrics['precision']:<8.3f}{metrics['recall']:<8.3f}{metrics['map']:<8.3f}{metrics['ndcg']:<8.3f}{metrics['hit_rate']:<8.3f}"
            else:
                row += f"{'N/A':<8}{'N/A':<8}{'N/A':<8}{'N/A':<8}{'N/A':<8}"
        print(row)
    
    print("="*80)


def main():
    """Main training pipeline."""
    # Load configuration
    config = load_config()
    
    # Set random seeds
    set_seed(config["data"]["random_seed"])
    
    # Prepare data
    interactions_df, items_df, users_df = prepare_data(config)
    
    # Split data
    train_df, val_df, test_df = split_data(interactions_df, config)
    
    # Train Item2vec model
    item2vec_model, item_encoder = train_item2vec(train_df, val_df, config)
    
    # Train baseline models
    baseline_models = train_baselines(train_df, config)
    
    # Combine all models
    all_models = {"item2vec": item2vec_model, **baseline_models}
    
    # Evaluate models
    results = evaluate_models(all_models, test_df, config)
    
    # Print results
    print_results(results)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/evaluation_results.csv")
    
    print("\nTraining completed! Results saved to results/evaluation_results.csv")


if __name__ == "__main__":
    main()
