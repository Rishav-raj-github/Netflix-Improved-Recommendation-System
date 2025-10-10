"""Base model classes and sample Matrix Factorization model implementation.

This module provides the foundation for building recommendation models,
including a base class and a sample Matrix Factorization implementation.
"""

import torch
import torch.nn as nn
import numpy as np


class BaseRecommender(nn.Module):
    """Base class for all recommendation models.
    
    All custom models should inherit from this class and implement
    the forward() and predict() methods.
    """
    
    def __init__(self):
        super(BaseRecommender, self).__init__()
    
    def forward(self, user_ids, item_ids):
        """Forward pass for training.
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            predictions: Predicted ratings
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def predict(self, user_id, n_items=10):
        """Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID for recommendations
            n_items: Number of items to recommend
            
        Returns:
            recommendations: List of recommended item IDs
        """
        raise NotImplementedError("Subclasses must implement predict()")


class MatrixFactorization(BaseRecommender):
    """Matrix Factorization model for collaborative filtering.
    
    This model learns latent representations for users and items
    and predicts ratings as the dot product of their embeddings.
    
    Args:
        n_users: Number of users in the dataset
        n_items: Number of items in the dataset
        embedding_dim: Dimension of latent embeddings (default: 50)
    """
    
    def __init__(self, n_users, n_items, embedding_dim=50):
        super(MatrixFactorization, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with small random values."""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        """Forward pass - compute predicted ratings.
        
        Args:
            user_ids: Tensor of user IDs (batch_size,)
            item_ids: Tensor of item IDs (batch_size,)
            
        Returns:
            predictions: Predicted ratings (batch_size,)
        """
        # Get embeddings
        user_embed = self.user_embeddings(user_ids)  # (batch_size, embedding_dim)
        item_embed = self.item_embeddings(item_ids)  # (batch_size, embedding_dim)
        
        # Get biases
        user_b = self.user_bias(user_ids).squeeze()  # (batch_size,)
        item_b = self.item_bias(item_ids).squeeze()  # (batch_size,)
        
        # Compute prediction: dot product + biases
        dot_product = (user_embed * item_embed).sum(dim=1)
        predictions = dot_product + user_b + item_b + self.global_bias
        
        return predictions
    
    def predict(self, user_id, n_items=10):
        """Generate top-N item recommendations for a user.
        
        Args:
            user_id: Integer user ID
            n_items: Number of items to recommend
            
        Returns:
            recommendations: List of (item_id, score) tuples
        """
        self.eval()
        with torch.no_grad():
            # Get user embedding
            user_tensor = torch.tensor([user_id])
            user_embed = self.user_embeddings(user_tensor)  # (1, embedding_dim)
            user_b = self.user_bias(user_tensor)
            
            # Compute scores for all items
            all_item_embeds = self.item_embeddings.weight  # (n_items, embedding_dim)
            all_item_bias = self.item_bias.weight  # (n_items, 1)
            
            scores = torch.matmul(user_embed, all_item_embeds.t()).squeeze()
            scores = scores + user_b.squeeze() + all_item_bias.squeeze() + self.global_bias
            
            # Get top-N items
            top_scores, top_indices = torch.topk(scores, n_items)
            
            recommendations = [
                (idx.item(), score.item()) 
                for idx, score in zip(top_indices, top_scores)
            ]
            
        return recommendations
    
    def save_model(self, path):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path))
