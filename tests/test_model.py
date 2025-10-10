"""Unit tests for recommendation model.

This module provides pytest tests for the Matrix Factorization model.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model import MatrixFactorization, BaseRecommender


class TestBaseRecommender:
    """Tests for BaseRecommender base class."""
    
    def test_base_class_exists(self):
        """Test that base class can be instantiated."""
        assert BaseRecommender is not None
    
    def test_base_class_is_abstract(self):
        """Test that base class methods are not implemented."""
        recommender = BaseRecommender()
        
        with pytest.raises(NotImplementedError):
            recommender.forward(None, None)
        
        with pytest.raises(NotImplementedError):
            recommender.predict(0, 10)


class TestMatrixFactorization:
    """Tests for MatrixFactorization model."""
    
    @pytest.fixture
    def model_params(self):
        """Fixture providing model parameters."""
        return {
            'n_users': 100,
            'n_items': 50,
            'embedding_dim': 10
        }
    
    @pytest.fixture
    def model(self, model_params):
        """Fixture providing initialized model."""
        return MatrixFactorization(**model_params)
    
    def test_model_initialization(self, model, model_params):
        """Test that model initializes correctly."""
        assert model.n_users == model_params['n_users']
        assert model.n_items == model_params['n_items']
        assert model.embedding_dim == model_params['embedding_dim']
    
    def test_model_has_embeddings(self, model):
        """Test that model has user and item embeddings."""
        assert hasattr(model, 'user_embeddings')
        assert hasattr(model, 'item_embeddings')
        assert hasattr(model, 'user_bias')
        assert hasattr(model, 'item_bias')
        assert hasattr(model, 'global_bias')
    
    def test_embedding_dimensions(self, model, model_params):
        """Test embedding dimensions are correct."""
        user_embed_shape = model.user_embeddings.weight.shape
        item_embed_shape = model.item_embeddings.weight.shape
        
        assert user_embed_shape == (model_params['n_users'], model_params['embedding_dim'])
        assert item_embed_shape == (model_params['n_items'], model_params['embedding_dim'])
    
    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        batch_size = 32
        user_ids = torch.randint(0, model.n_users, (batch_size,))
        item_ids = torch.randint(0, model.n_items, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        
        assert predictions.shape == (batch_size,)
        assert not torch.isnan(predictions).any()
    
    def test_forward_pass_single_sample(self, model):
        """Test forward pass with single sample."""
        user_id = torch.tensor([0])
        item_id = torch.tensor([0])
        
        prediction = model(user_id, item_id)
        
        assert prediction.shape == (1,)
        assert not torch.isnan(prediction).any()
    
    def test_predict_method(self, model):
        """Test predict method returns recommendations."""
        user_id = 0
        n_items = 10
        
        recommendations = model.predict(user_id, n_items)
        
        assert len(recommendations) == n_items
        assert all(isinstance(item, tuple) for item in recommendations)
        assert all(len(item) == 2 for item in recommendations)
    
    def test_predict_returns_sorted_scores(self, model):
        """Test that predict returns items sorted by score."""
        user_id = 0
        n_items = 10
        
        recommendations = model.predict(user_id, n_items)
        scores = [score for _, score in recommendations]
        
        # Check scores are in descending order
        assert scores == sorted(scores, reverse=True)
    
    def test_model_parameter_count(self, model, model_params):
        """Test model has expected number of parameters."""
        n_params = sum(p.numel() for p in model.parameters())
        
        # Expected parameters:
        # user_embeddings: n_users * embedding_dim
        # item_embeddings: n_items * embedding_dim
        # user_bias: n_users
        # item_bias: n_items
        # global_bias: 1
        expected = (
            model_params['n_users'] * model_params['embedding_dim'] +
            model_params['n_items'] * model_params['embedding_dim'] +
            model_params['n_users'] +
            model_params['n_items'] +
            1
        )
        
        assert n_params == expected
    
    def test_model_training_mode(self, model):
        """Test model can switch between train and eval modes."""
        model.train()
        assert model.training is True
        
        model.eval()
        assert model.training is False
    
    def test_save_and_load_model(self, model, tmp_path):
        """Test model can be saved and loaded."""
        # Save model
        model_path = tmp_path / "test_model.pth"
        model.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model into new instance
        new_model = MatrixFactorization(
            n_users=model.n_users,
            n_items=model.n_items,
            embedding_dim=model.embedding_dim
        )
        new_model.load_model(str(model_path))
        
        # Check parameters are the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
