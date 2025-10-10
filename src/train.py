"""Main training script for the recommendation system.

This script handles the complete training pipeline: loading data, building model,
training, evaluation, and saving the trained model.
"""

import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

# Import custom modules
from models.model import MatrixFactorization
from preprocessing.preprocess import (
    load_ratings_data, clean_ratings_data, encode_users_items,
    split_train_test, filter_by_min_interactions
)
from evaluation.evaluate import evaluate_model


class RatingsDataset(Dataset):
    """PyTorch Dataset for ratings data."""
    
    def __init__(self, df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Initialize dataset from DataFrame.
        
        Args:
            df: DataFrame with ratings data
            user_col: Name of user ID column
            item_col: Name of item ID column
            rating_col: Name of rating column
        """
        self.users = torch.tensor(df[user_col].values, dtype=torch.long)
        self.items = torch.tensor(df[item_col].values, dtype=torch.long)
        self.ratings = torch.tensor(df[rating_col].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch.
    
    Args:
        model: The recommendation model
        dataloader: DataLoader with training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for users, items, ratings in dataloader:
        # Move data to device
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(users, items)
        
        # Compute loss
        loss = criterion(predictions, ratings)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model on validation data.
    
    Args:
        model: The recommendation model
        dataloader: DataLoader with validation data
        criterion: Loss function
        device: Device to validate on (cpu or cuda)
        
    Returns:
        avg_loss: Average loss for validation data
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for users, items, ratings in dataloader:
            # Move data to device
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)
            
            # Forward pass
            predictions = model(users, items)
            
            # Compute loss
            loss = criterion(predictions, ratings)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model(config):
    """Main training function.
    
    Args:
        config: Dictionary with configuration parameters
    """
    print("=" * 60)
    print("Netflix Recommendation System - Training")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n[1/6] Loading data...")
    data_path = config['data_path']
    df = load_ratings_data(data_path)
    
    print("\n[2/6] Cleaning and preprocessing data...")
    df_clean = clean_ratings_data(
        df, 
        min_rating=config.get('min_rating'),
        max_rating=config.get('max_rating')
    )
    
    # Filter by minimum interactions if specified
    if config.get('min_user_interactions') or config.get('min_item_interactions'):
        df_clean = filter_by_min_interactions(
            df_clean,
            min_user_interactions=config.get('min_user_interactions', 5),
            min_item_interactions=config.get('min_item_interactions', 5)
        )
    
    # Encode users and items
    df_encoded, user_encoder, item_encoder = encode_users_items(df_clean)
    
    # Split into train and test
    train_df, test_df = split_train_test(
        df_encoded, 
        test_size=config['test_size'],
        random_state=config['random_seed']
    )
    
    # Get number of unique users and items
    n_users = df_encoded['user_id'].max() + 1
    n_items = df_encoded['item_id'].max() + 1
    print(f"Number of users: {n_users}")
    print(f"Number of items: {n_items}")
    
    # Create datasets and dataloaders
    print("\n[3/6] Creating dataloaders...")
    train_dataset = RatingsDataset(train_df)
    test_dataset = RatingsDataset(test_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    # Create model
    print("\n[4/6] Building model...")
    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config['embedding_dim']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set up training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("\n[5/6] Training model...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, test_loader, criterion, device)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_dir = Path(config['model_save_path']).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(config['model_save_path'])
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 5):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for evaluation
    print("\n[6/6] Evaluating model...")
    model.load_model(config['model_save_path'])
    
    # Evaluate
    results = evaluate_model(
        model, 
        test_df,
        k=config.get('top_k', 10),
        metrics=['rmse', 'mae', 'precision', 'recall', 'ndcg']
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric:15s}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {config['model_save_path']}")
    print("=" * 60)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Netflix recommendation model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train model
    train_model(config)
