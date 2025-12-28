"""
05_Advanced_Techniques.py

Advanced Recommendation Techniques for Netflix

This notebook covers advanced techniques including deep learning,
matrix factorization, and neural collaborative filtering.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, SVD
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedTechniquesNotebook:
    """
    Advanced recommendation techniques using matrix factorization and deep learning approaches.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, n_factors: int = 20):
        """
        Initialize advanced techniques.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Ratings dataframe
        n_factors : int
            Number of latent factors
        """
        self.ratings_df = ratings_df
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.nmf_model = None
        
    def matrix_factorization(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply NMF for matrix factorization.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            User and item factor matrices
        """
        print(f"Applying NMF with {self.n_factors} factors...")
        
        # Create user-movie matrix
        user_movie_matrix = self.ratings_df.pivot_table(
            index='userId', columns='movieId', values='rating', fill_value=0
        )
        
        # Apply NMF
        self.nmf_model = NMF(n_components=self.n_factors, init='random', random_state=42)
        self.user_factors = self.nmf_model.fit_transform(user_movie_matrix)
        self.item_factors = self.nmf_model.components_.T
        
        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")
        
        return self.user_factors, self.item_factors
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using factorized matrices.
        
        Parameters:
        -----------
        user_id : int
            User ID (index)
        item_id : int
            Item ID (index)
            
        Returns:
        --------
        float
            Predicted rating
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Factors not computed. Call matrix_factorization() first.")
        
        prediction = np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return float(np.clip(prediction, 1, 5))  # Clip to valid rating range
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get recommendations using matrix factorization.
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations
            
        Returns:
        --------
        List[Tuple[int, float]]
            Recommended items with predicted ratings
        """
        # Get all item predictions
        predictions = {}
        for item_id in range(self.item_factors.shape[0]):
            predictions[item_id] = self.predict_rating(user_id, item_id)
        
        # Sort and return top N
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
    
    def cold_start_handling(self, new_user_ratings: Dict[int, float]) -> List[Tuple[int, float]]:
        """
        Handle cold-start problem for new users.
        
        Parameters:
        -----------
        new_user_ratings : Dict[int, float]
            Ratings provided by new user
            
        Returns:
        --------
        List[Tuple[int, float]]
            Recommendations based on initial ratings
        """
        print("Handling cold-start for new user...")
        
        # Use content-based approach for initial recommendations
        recommendations = {}
        for item_id in range(self.item_factors.shape[0]):
            if item_id not in new_user_ratings:
                # Find similar items to what user rated
                similarity_sum = 0
                for rated_item, rating in new_user_ratings.items():
                    sim = np.dot(self.item_factors[item_id], self.item_factors[rated_item])
                    similarity_sum += rating * sim
                recommendations[item_id] = similarity_sum / len(new_user_ratings) if new_user_ratings else 0
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:5]
    
    def serendipity_injection(self, recommendations: List[Tuple[int, float]], 
                            serendipity_ratio: float = 0.2) -> List[Tuple[int, float]]:
        """
        Inject serendipity into recommendations.
        
        Parameters:
        -----------
        recommendations : List[Tuple[int, float]]
            Original recommendations
        serendipity_ratio : float
            Ratio of serendipitous items
            
        Returns:
        --------
        List[Tuple[int, float]]
            Recommendations with serendipitous items
        """
        print(f"Injecting {serendipity_ratio*100}% serendipity...")
        
        n_serendipity = max(1, int(len(recommendations) * serendipity_ratio))
        core_recs = recommendations[:-n_serendipity] if len(recommendations) > n_serendipity else recommendations
        
        # Add random high-quality items for serendipity
        random_items = np.random.choice(self.item_factors.shape[0], n_serendipity, replace=False)
        serendipitous = [(item, np.random.uniform(3.5, 5.0)) for item in random_items]
        
        return core_recs + serendipitous
    
    def evaluate_reconstruction_error(self) -> float:
        """
        Evaluate matrix factorization reconstruction error.
        
        Returns:
        --------
        float
            Mean Squared Error
        """
        if self.nmf_model is None:
            return float('inf')
        
        # Reconstruct matrix
        reconstruction = np.dot(self.user_factors, self.item_factors.T)
        
        # Compare with original
        user_movie_matrix = self.ratings_df.pivot_table(
            index='userId', columns='movieId', values='rating', fill_value=0
        )
        
        error = np.mean((user_movie_matrix.values - reconstruction) ** 2)
        print(f"Reconstruction MSE: {error:.4f}")
        return error

def main():
    """
    Example usage of Advanced Techniques.
    """
    print("="*60)
    print("Advanced Recommendation Techniques")
    print("="*60)
    
    # Dummy data
    ratings_data = {
        'userId': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'movieId': [0, 1, 2, 0, 3, 1, 3, 4, 2, 4],
        'rating': [5, 4, 3, 5, 4, 5, 3, 4, 4, 5]
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    # Initialize and run advanced techniques
    advanced = AdvancedTechniquesNotebook(ratings_df, n_factors=3)
    user_factors, item_factors = advanced.matrix_factorization()
    
    print("\nRecommendations for User 0:")
    recs = advanced.get_recommendations(0, n_recommendations=3)
    for item_id, score in recs:
        print(f"  Item ID: {item_id}, Predicted Rating: {score:.2f}")
    
    print("\nReconstructing ratings matrix...")
    error = advanced.evaluate_reconstruction_error()
    
    print("\nAdvanced Techniques Notebook Complete!")

if __name__ == "__main__":
    main()
