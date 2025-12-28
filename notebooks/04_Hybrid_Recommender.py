"""
04_Hybrid_Recommender.py

Hybrid Recommendation System combining multiple approaches

This notebook demonstrates hybrid recommendation systems that combine
collaborative filtering, content-based filtering, and other techniques.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HybridRecommenderNotebook:
    """
    Hybrid Recommendation System combining multiple filtering approaches.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                 content_scores: np.ndarray = None, collab_scores: np.ndarray = None):
        """
        Initialize the hybrid recommender.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe
        movies_df : pd.DataFrame
            Movies dataframe
        content_scores : np.ndarray, optional
            Content-based filtering scores
        collab_scores : np.ndarray, optional
            Collaborative filtering scores
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.content_scores = content_scores
        self.collab_scores = collab_scores
        self.hybrid_scores = None
        self.weights = {'collab': 0.6, 'content': 0.4}  # Default weights
        
    def set_weights(self, collab_weight: float = 0.6, content_weight: float = 0.4):
        """
        Set the weights for hybrid combination.
        
        Parameters:
        -----------
        collab_weight : float
            Weight for collaborative filtering
        content_weight : float
            Weight for content-based filtering
        """
        total = collab_weight + content_weight
        self.weights = {
            'collab': collab_weight / total,
            'content': content_weight / total
        }
        print(f"Weights set - Collaborative: {self.weights['collab']:.2f}, Content: {self.weights['content']:.2f}")
    
    def combine_scores(self) -> np.ndarray:
        """
        Combine collaborative and content-based scores using weighted average.
        
        Returns:
        --------
        np.ndarray
            Combined hybrid scores
        """
        if self.collab_scores is None or self.content_scores is None:
            raise ValueError("Both collab_scores and content_scores must be provided")
        
        # Normalize scores
        scaler = MinMaxScaler()
        collab_norm = scaler.fit_transform(self.collab_scores)
        content_norm = scaler.fit_transform(self.content_scores)
        
        # Weighted combination
        self.hybrid_scores = (self.weights['collab'] * collab_norm + 
                             self.weights['content'] * content_norm)
        
        print(f"Hybrid scores shape: {self.hybrid_scores.shape}")
        print(f"Hybrid scores range: [{self.hybrid_scores.min():.4f}, {self.hybrid_scores.max():.4f}]")
        
        return self.hybrid_scores
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations for a user.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        List[Tuple[int, float]]
            List of (movie_id, score) tuples
        """
        if self.hybrid_scores is None:
            self.combine_scores()
        
        # Get user's already rated movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        rated_movies = set(user_ratings['movieId'].values)
        
        # Get unrated movies with scores
        scores_dict = {}
        for idx in range(len(self.movies_df)):
            if idx not in rated_movies:
                scores_dict[idx] = self.hybrid_scores[idx] if isinstance(self.hybrid_scores[idx], (int, float)) else self.hybrid_scores[idx, user_id % self.hybrid_scores.shape[1]]
        
        # Sort and return top N
        sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n_recommendations]
    
    def rank_by_hybrid_score(self, n_top: int = 10) -> List[Tuple[int, float]]:
        """
        Rank all movies by hybrid score.
        
        Parameters:
        -----------
        n_top : int
            Number of top movies to return
            
        Returns:
        --------
        List[Tuple[int, float]]
            Top ranked movies with scores
        """
        if self.hybrid_scores is None:
            self.combine_scores()
        
        if len(self.hybrid_scores.shape) == 1:
            scores = self.hybrid_scores
        else:
            scores = self.hybrid_scores.mean(axis=1)
        
        top_indices = np.argsort(scores)[::-1][:n_top]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def get_recommendation_explanation(self, user_id: int, movie_id: int) -> Dict:
        """
        Get explanation for why a movie was recommended.
        
        Parameters:
        -----------
        user_id : int
            User ID
        movie_id : int
            Movie ID
            
        Returns:
        --------
        Dict
            Explanation containing collab and content scores
        """
        explanation = {
            'user_id': user_id,
            'movie_id': movie_id,
            'collaborative_score': float(self.collab_scores[movie_id]) if self.collab_scores is not None else None,
            'content_score': float(self.content_scores[movie_id]) if self.content_scores is not None else None,
            'collab_weight': self.weights['collab'],
            'content_weight': self.weights['content']
        }
        return explanation
    
    def evaluate_diversity(self) -> Dict[str, float]:
        """
        Evaluate diversity of recommendations.
        
        Returns:
        --------
        Dict[str, float]
            Diversity metrics
        """
        print("Evaluating recommendation diversity...")
        metrics = {
            'coverage': min(1.0, len(self.movies_df) / self.movies_df.shape[0]),
            'diversity_score': float(np.std(self.hybrid_scores)) if self.hybrid_scores is not None else 0,
        }
        return metrics

def main():
    """
    Example usage of Hybrid Recommender System.
    """
    print("="*60)
    print("Hybrid Recommendation System")
    print("="*60)
    
    # Dummy data
    ratings_data = {
        'userId': [1, 1, 1, 2, 2, 3, 3, 3],
        'movieId': [0, 1, 2, 0, 3, 1, 3, 4],
        'rating': [5, 4, 3, 5, 4, 5, 3, 4]
    }
    
    movies_data = {
        'movieId': [0, 1, 2, 3, 4],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
    }
    
    ratings_df = pd.DataFrame(ratings_data)
    movies_df = pd.DataFrame(movies_data)
    
    # Create dummy scores
    collab_scores = np.array([0.8, 0.7, 0.6, 0.9, 0.5])
    content_scores = np.array([0.7, 0.8, 0.5, 0.6, 0.9])
    
    # Initialize hybrid recommender
    hybrid = HybridRecommenderNotebook(ratings_df, movies_df, content_scores, collab_scores)
    hybrid.set_weights(collab_weight=0.6, content_weight=0.4)
    
    print("\nCombining scores...")
    hybrid_scores = hybrid.combine_scores()
    
    print("\nTop 5 movies by hybrid score:")
    top_movies = hybrid.rank_by_hybrid_score(n_top=5)
    for movie_id, score in top_movies:
        print(f"  Movie ID: {movie_id}, Score: {score:.4f}")
    
    print("\nDiversity Metrics:")
    metrics = hybrid.evaluate_diversity()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nHybrid Recommender Notebook Complete!")

if __name__ == "__main__":
    main()
