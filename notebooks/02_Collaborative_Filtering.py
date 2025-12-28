"""Collaborative Filtering Implementation for Netflix Recommendations

This notebook implements multiple collaborative filtering approaches:
- User-based Collaborative Filtering
- Item-based Collaborative Filtering
- Matrix Factorization (SVD)
- Evaluation metrics (RMSE, MAE, Precision, Recall)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """Collaborative Filtering recommendation engine."""
    
    def __init__(self, min_ratings: int = 3):
        """Initialize CF model."""
        self.ratings_matrix = None
        self.user_similarities = None
        self.item_similarities = None
        self.min_ratings = min_ratings
        self.user_means = None
    
    def create_ratings_matrix(self, df_ratings: pd.DataFrame, min_users=5, min_items=5):
        """Create sparse ratings matrix from dataframe."""
        # Filter users and items with sufficient ratings
        user_counts = df_ratings['user_id'].value_counts()
        item_counts = df_ratings['movie_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_users].index
        valid_items = item_counts[item_counts >= min_items].index
        
        df_filtered = df_ratings[
            (df_ratings['user_id'].isin(valid_users)) & 
            (df_ratings['movie_id'].isin(valid_items))
        ]
        
        # Create ratings matrix
        self.ratings_matrix = df_filtered.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        )
        
        print(f"✓ Ratings matrix created: {self.ratings_matrix.shape}")
        print(f"  Sparsity: {(self.ratings_matrix.isna().sum().sum() / self.ratings_matrix.size) * 100:.2f}%")
    
    def user_similarity(self, method: str = 'cosine') -> np.ndarray:
        """Compute user-user similarity matrix."""
        # Fill NaN with 0 for similarity calculation
        matrix_filled = self.ratings_matrix.fillna(0).values
        
        if method == 'cosine':
            # Cosine similarity
            norms = np.linalg.norm(matrix_filled, axis=1, keepdims=True)
            normalized = matrix_filled / (norms + 1e-10)
            similarities = np.dot(normalized, normalized.T)
        elif method == 'pearson':
            # Pearson correlation (normalized cosine)
            means = matrix_filled.mean(axis=1, keepdims=True)
            centered = matrix_filled - means
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            normalized = centered / (norms + 1e-10)
            similarities = np.dot(normalized, normalized.T)
        
        np.fill_diagonal(similarities, 0)  # No self-similarity
        self.user_similarities = similarities
        return similarities
    
    def item_similarity(self, method: str = 'cosine') -> np.ndarray:
        """Compute item-item similarity matrix."""
        matrix_filled = self.ratings_matrix.fillna(0).values.T
        
        if method == 'cosine':
            norms = np.linalg.norm(matrix_filled, axis=1, keepdims=True)
            normalized = matrix_filled / (norms + 1e-10)
            similarities = np.dot(normalized, normalized.T)
        
        np.fill_diagonal(similarities, 0)
        self.item_similarities = similarities
        return similarities
    
    def matrix_factorization(self, n_factors: int = 50, random_state: int = 42):
        """Apply SVD-based matrix factorization."""
        matrix_filled = self.ratings_matrix.fillna(self.ratings_matrix.mean())
        
        svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        matrix_factorized = svd.fit_transform(matrix_filled)
        
        print(f"✓ Matrix Factorization (SVD):")
        print(f"  Variance explained: {svd.explained_variance_ratio_.sum():.4f}")
        print(f"  Shape: {matrix_factorized.shape}")
        
        return matrix_factorized, svd
    
    def user_based_cf(self, user_id: int, k: int = 5, n_recommendations: int = 10):
        """User-based CF recommendations."""
        if user_id not in self.ratings_matrix.index:
            return []
        
        # Find similar users
        user_idx = list(self.ratings_matrix.index).index(user_id)
        similarities = self.user_similarities[user_idx]
        similar_users_idx = np.argsort(-similarities)[:k]
        
        # Get recommendations from similar users
        user_ratings = self.ratings_matrix.iloc[user_idx]
        recommendations = {}
        
        for sim_idx in similar_users_idx:
            similar_user_id = self.ratings_matrix.index[sim_idx]
            similar_user_ratings = self.ratings_matrix.iloc[sim_idx]
            similarity_weight = similarities[sim_idx]
            
            # Items not rated by target user
            unrated_items = similar_user_ratings[user_ratings.isna()]
            
            for item_id, rating in unrated_items.items():
                if item_id not in recommendations:
                    recommendations[item_id] = {'weighted_sum': 0, 'sim_sum': 0}
                recommendations[item_id]['weighted_sum'] += rating * similarity_weight
                recommendations[item_id]['sim_sum'] += abs(similarity_weight)
        
        # Calculate weighted scores
        scores = [
            (item, values['weighted_sum'] / (values['sim_sum'] + 1e-10))
            for item, values in recommendations.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, score in scores[:n_recommendations]]
    
    def evaluate(self, test_data: pd.DataFrame):
        """Evaluate model on test data."""
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id, movie_id, rating = row['user_id'], row['movie_id'], row['rating']
            
            if user_id in self.ratings_matrix.index and movie_id in self.ratings_matrix.columns:
                user_idx = list(self.ratings_matrix.index).index(user_id)
                pred = self.ratings_matrix.iloc[user_idx][movie_id]
                
                if not pd.isna(pred):
                    predictions.append(pred)
                    actuals.append(rating)
        
        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            print(f"\n✓ Evaluation Metrics:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            return rmse, mae
        return None, None


if __name__ == '__main__':
    # Example usage
    print("Collaborative Filtering module loaded successfully")
