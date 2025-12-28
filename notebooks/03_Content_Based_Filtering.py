"""
03_Content_Based_Filtering.py

Content-Based Filtering for Netflix Recommendation System

This notebook demonstrates content-based filtering approaches for recommendation systems.
Content-based filtering recommends items similar to those the user has previously liked.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ContentBasedFilteringNotebook:
    """
    Content-Based Filtering implementation for Netflix data.
    Uses item features (genre, cast, director, etc.) to find similar items.
    """
    
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Initialize the content-based filtering system.
        
        Parameters:
        -----------
        movies_df : pd.DataFrame
            Movies dataframe with features like genre, director, cast
        ratings_df : pd.DataFrame
            Ratings dataframe with user-movie ratings
        """
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.similarity_matrix = None
        self.tfidf_matrix = None
        
    def compute_tfidf_features(self, content_column: str = 'genres') -> np.ndarray:
        """
        Compute TF-IDF features for movie content.
        
        Parameters:
        -----------
        content_column : str
            Column name containing text content (default: 'genres')
            
        Returns:
        --------
        np.ndarray
            TF-IDF feature matrix
        """
        print(f"Computing TF-IDF features for {content_column}...")
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.tfidf_matrix = vectorizer.fit_transform(self.movies_df[content_column].astype(str))
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix
    
    def compute_similarity_matrix(self, feature_matrix=None) -> np.ndarray:
        """
        Compute cosine similarity matrix between movies.
        
        Parameters:
        -----------
        feature_matrix : array-like, optional
            Feature matrix for similarity computation
            
        Returns:
        --------
        np.ndarray
            Similarity matrix
        """
        if feature_matrix is None:
            feature_matrix = self.tfidf_matrix
        
        print("Computing similarity matrix using cosine similarity...")
        self.similarity_matrix = cosine_similarity(feature_matrix)
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def get_similar_movies(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get top N similar movies for a given movie.
        
        Parameters:
        -----------
        movie_id : int
            Movie ID
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        List[Tuple[int, float]]
            List of (movie_id, similarity_score) tuples
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call compute_similarity_matrix() first.")
        
        similarities = self.similarity_matrix[movie_id]
        top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]  # Exclude itself
        
        recommendations = [(idx, similarities[idx]) for idx in top_indices]
        return recommendations
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Recommend movies for a user based on their rated movies.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        List[Dict]
            List of recommendation dictionaries with movie info and scores
        """
        # Get movies rated by user
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if user_ratings.empty:
            return []
        
        rated_movies = user_ratings['movieId'].values
        recommendations = {}
        
        # Get similar movies for each rated movie
        for movie_id in rated_movies:
            if movie_id < len(self.similarity_matrix):
                similar = self.get_similar_movies(movie_id, n_recommendations=10)
                for sim_movie_id, sim_score in similar:
                    if sim_movie_id not in recommendations and sim_movie_id not in rated_movies:
                        recommendations[sim_movie_id] = sim_score
        
        # Sort by score and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
    
    def evaluate_hybrid_features(self) -> Dict[str, float]:
        """
        Evaluate content-based filtering with multiple features.
        
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        print("Evaluating content-based filtering performance...")
        metrics = {
            'sparsity': 1 - (self.ratings_df.shape[0] / (self.ratings_df['userId'].max() * self.movies_df.shape[0])),
            'avg_rating': self.ratings_df['rating'].mean(),
            'coverage': len(self.movies_df) / self.movies_df.shape[0],
        }
        return metrics

def main():
    """
    Example usage of Content-Based Filtering.
    """
    print("="*60)
    print("Content-Based Filtering for Netflix Recommendation System")
    print("="*60)
    
    # Dummy data for demonstration
    movies_data = {
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genres': ['Action|Adventure', 'Action|Thriller', 'Comedy|Romance', 'Drama', 'Action|Adventure|Sci-Fi']
    }
    
    ratings_data = {
        'userId': [1, 1, 1, 2, 2, 3, 3, 3],
        'movieId': [1, 2, 3, 1, 4, 2, 4, 5],
        'rating': [5, 4, 3, 5, 4, 5, 3, 4]
    }
    
    movies_df = pd.DataFrame(movies_data)
    ratings_df = pd.DataFrame(ratings_data)
    
    # Initialize and run content-based filtering
    cbf = ContentBasedFilteringNotebook(movies_df, ratings_df)
    cbf.compute_tfidf_features('genres')
    cbf.compute_similarity_matrix()
    
    print("\nSimilar movies to Movie A (ID=0):")
    similar = cbf.get_similar_movies(0, n_recommendations=3)
    for movie_id, score in similar:
        print(f"  Movie ID: {movie_id}, Similarity: {score:.4f}")
    
    print("\nRecommendations for User 1:")
    user_recs = cbf.recommend_for_user(1, n_recommendations=3)
    for movie_id, score in user_recs:
        print(f"  Movie ID: {movie_id}, Score: {score:.4f}")
    
    print("\nPerformance Metrics:")
    metrics = cbf.evaluate_hybrid_features()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nContent-Based Filtering Notebook Complete!")

if __name__ == "__main__":
    main()
