"""Data preprocessing utilities for recommendation system.

This module provides functions for reading data (CSV), encoding users/items,
and performing simple data cleaning operations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


def load_ratings_data(file_path, sep=',', names=None):
    """Load ratings data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        sep: Delimiter to use (default: ',')
        names: List of column names (default: None)
        
    Returns:
        df: Pandas DataFrame with ratings data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read CSV file
    if names:
        df = pd.read_csv(file_path, sep=sep, names=names)
    else:
        df = pd.read_csv(file_path, sep=sep)
    
    print(f"Loaded {len(df)} ratings from {file_path}")
    return df


def clean_ratings_data(df, user_col='user_id', item_col='item_id', 
                       rating_col='rating', min_rating=None, max_rating=None):
    """Clean ratings data by removing invalid entries.
    
    Args:
        df: DataFrame with ratings data
        user_col: Name of user ID column
        item_col: Name of item ID column
        rating_col: Name of rating column
        min_rating: Minimum valid rating (default: None)
        max_rating: Maximum valid rating (default: None)
        
    Returns:
        df_clean: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove rows with missing values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=[user_col, item_col, rating_col])
    print(f"Removed {initial_count - len(df_clean)} rows with missing values")
    
    # Filter by rating range if specified
    if min_rating is not None:
        df_clean = df_clean[df_clean[rating_col] >= min_rating]
    if max_rating is not None:
        df_clean = df_clean[df_clean[rating_col] <= max_rating]
    
    # Remove duplicate entries (keep first occurrence)
    df_clean = df_clean.drop_duplicates(subset=[user_col, item_col], keep='first')
    
    print(f"Final dataset size: {len(df_clean)} ratings")
    return df_clean


def encode_users_items(df, user_col='user_id', item_col='item_id'):
    """Encode user and item IDs to sequential integers.
    
    Args:
        df: DataFrame with ratings data
        user_col: Name of user ID column
        item_col: Name of item ID column
        
    Returns:
        df_encoded: DataFrame with encoded IDs
        user_encoder: Fitted LabelEncoder for users
        item_encoder: Fitted LabelEncoder for items
    """
    df_encoded = df.copy()
    
    # Create and fit encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # Encode user and item IDs
    df_encoded[user_col] = user_encoder.fit_transform(df[user_col])
    df_encoded[item_col] = item_encoder.fit_transform(df[item_col])
    
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    print(f"Encoded {n_users} unique users and {n_items} unique items")
    
    return df_encoded, user_encoder, item_encoder


def split_train_test(df, test_size=0.2, random_state=42):
    """Split data into train and test sets.
    
    Args:
        df: DataFrame with ratings data
        test_size: Fraction of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        train_df: Training DataFrame
        test_df: Test DataFrame
    """
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split into train and test
    split_idx = int(len(df_shuffled) * (1 - test_size))
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]
    
    print(f"Train set: {len(train_df)} ratings")
    print(f"Test set: {len(test_df)} ratings")
    
    return train_df, test_df


def filter_by_min_interactions(df, user_col='user_id', item_col='item_id',
                               min_user_interactions=5, min_item_interactions=5):
    """Filter users and items by minimum number of interactions.
    
    This helps remove cold-start problems by ensuring all users and items
    have a minimum number of ratings.
    
    Args:
        df: DataFrame with ratings data
        user_col: Name of user ID column
        item_col: Name of item ID column
        min_user_interactions: Minimum ratings per user (default: 5)
        min_item_interactions: Minimum ratings per item (default: 5)
        
    Returns:
        df_filtered: Filtered DataFrame
    """
    df_filtered = df.copy()
    
    # Iteratively filter until no more users/items are removed
    prev_size = len(df_filtered)
    while True:
        # Filter users
        user_counts = df_filtered[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df_filtered = df_filtered[df_filtered[user_col].isin(valid_users)]
        
        # Filter items
        item_counts = df_filtered[item_col].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df_filtered = df_filtered[df_filtered[item_col].isin(valid_items)]
        
        # Check if size changed
        if len(df_filtered) == prev_size:
            break
        prev_size = len(df_filtered)
    
    print(f"Filtered dataset size: {len(df_filtered)} ratings")
    print(f"Unique users: {df_filtered[user_col].nunique()}")
    print(f"Unique items: {df_filtered[item_col].nunique()}")
    
    return df_filtered


def create_user_item_matrix(df, user_col='user_id', item_col='item_id', 
                            rating_col='rating', fill_value=0):
    """Create user-item interaction matrix.
    
    Args:
        df: DataFrame with ratings data
        user_col: Name of user ID column
        item_col: Name of item ID column
        rating_col: Name of rating column
        fill_value: Value to fill for missing entries (default: 0)
        
    Returns:
        matrix: User-item matrix as numpy array
        user_ids: List of user IDs
        item_ids: List of item IDs
    """
    # Create pivot table
    matrix_df = df.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
        fill_value=fill_value
    )
    
    # Convert to numpy array
    matrix = matrix_df.values
    user_ids = matrix_df.index.tolist()
    item_ids = matrix_df.columns.tolist()
    
    print(f"Created matrix of shape: {matrix.shape}")
    print(f"Sparsity: {100 * (1 - np.count_nonzero(matrix) / matrix.size):.2f}%")
    
    return matrix, user_ids, item_ids


def save_processed_data(df, output_path):
    """Save processed data to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module loaded successfully!")
    print("Example usage:")
    print("  df = load_ratings_data('data/raw/ratings.csv')")
    print("  df_clean = clean_ratings_data(df)")
    print("  df_encoded, user_enc, item_enc = encode_users_items(df_clean)")
    print("  train_df, test_df = split_train_test(df_encoded)")
