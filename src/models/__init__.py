"""Item2vec model implementation using PyTorch."""

from typing import List, Tuple, Optional, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class Item2VecDataset(Dataset):
    """Dataset class for Item2vec training."""
    
    def __init__(self, samples: List[Tuple[int, int, int]]):
        """Initialize dataset.
        
        Args:
            samples: List of (item, context, label) tuples.
        """
        self.samples = samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (item_tensor, context_tensor, label_tensor).
        """
        item, context, label = self.samples[idx]
        return (
            torch.tensor(item, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


class Item2Vec(nn.Module):
    """Item2vec model implementation using skip-gram architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        negative_samples: int = 5,
        device: Optional[torch.device] = None,
    ):
        """Initialize Item2vec model.
        
        Args:
            vocab_size: Number of unique items.
            embedding_dim: Dimension of item embeddings.
            negative_samples: Number of negative samples per positive sample.
            device: Device to run the model on.
        """
        super(Item2Vec, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.device = device or torch.device("cpu")
        
        # Item embeddings (target)
        self.item_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context embeddings (context)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Move to device
        self.to(self.device)
    
    def _init_embeddings(self) -> None:
        """Initialize embeddings with uniform distribution."""
        init_range = 0.5 / self.embedding_dim
        self.item_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, item_ids: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            item_ids: Tensor of item indices.
            context_ids: Tensor of context item indices.
            
        Returns:
            Tensor of similarity scores.
        """
        # Get embeddings
        item_emb = self.item_embeddings(item_ids)  # [batch_size, embedding_dim]
        context_emb = self.context_embeddings(context_ids)  # [batch_size, embedding_dim]
        
        # Compute similarity scores (dot product)
        scores = torch.sum(item_emb * context_emb, dim=1)  # [batch_size]
        
        return scores
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get all item embeddings.
        
        Returns:
            Tensor of all item embeddings.
        """
        return self.item_embeddings.weight.data
    
    def get_similar_items(
        self,
        item_id: int,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> List[Tuple[int, float]]:
        """Get most similar items to a given item.
        
        Args:
            item_id: ID of the item to find similar items for.
            top_k: Number of similar items to return.
            exclude_self: Whether to exclude the item itself from results.
            
        Returns:
            List of (item_id, similarity_score) tuples.
        """
        with torch.no_grad():
            # Get embedding for the target item
            target_emb = self.item_embeddings(torch.tensor([item_id], device=self.device))
            
            # Get all item embeddings
            all_embeddings = self.item_embeddings.weight.data
            
            # Compute cosine similarities
            similarities = F.cosine_similarity(
                target_emb.unsqueeze(0),
                all_embeddings.unsqueeze(0),
                dim=2,
            ).squeeze(0)
            
            # Get top-k similar items
            if exclude_self:
                similarities[item_id] = -1.0  # Exclude self
            
            top_indices = torch.topk(similarities, top_k).indices
            top_scores = similarities[top_indices]
            
            return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]
    
    def train_model(
        self,
        train_samples: List[Tuple[int, int, int]],
        val_samples: Optional[List[Tuple[int, int, int]]] = None,
        num_epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the Item2vec model.
        
        Args:
            train_samples: Training samples.
            val_samples: Validation samples (optional).
            num_epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            verbose: Whether to print training progress.
            
        Returns:
            Dictionary with training history.
        """
        # Create datasets and data loaders
        train_dataset = Item2VecDataset(train_samples)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = None
        if val_samples:
            val_dataset = Item2VecDataset(val_samples)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            if verbose:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_loader
            
            for batch in pbar:
                item_ids, context_ids, labels = batch
                item_ids = item_ids.to(self.device)
                context_ids = context_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                scores = self.forward(item_ids, context_ids)
                
                # Compute loss
                loss = criterion(scores, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if verbose:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_loader:
                self.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        item_ids, context_ids, labels = batch
                        item_ids = item_ids.to(self.device)
                        context_ids = context_ids.to(self.device)
                        labels = labels.to(self.device)
                        
                        scores = self.forward(item_ids, context_ids)
                        loss = criterion(scores, labels)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history["val_loss"].append(avg_val_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        return history
