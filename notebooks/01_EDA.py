"""Exploratory Data Analysis (EDA) for Netflix Recommendation System

This notebook performs comprehensive exploratory data analysis on the Netflix dataset:
- Data loading and basic statistics
- Missing values analysis
- Distribution of ratings and reviews
- User and movie interactions
- Temporal patterns
- Statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class NetflixEDA:
    """Exploratory Data Analysis for Netflix dataset."""
    
    def __init__(self, data_path: str = 'data/'):
        """Initialize EDA with data path."""
        self.data_path = Path(data_path)
        self.df_ratings = None
        self.df_movies = None
        self.df_users = None
    
    def load_data(self):
        """Load Netflix dataset files."""
        try:
            self.df_ratings = pd.read_csv(self.data_path / 'ratings.csv')
            self.df_movies = pd.read_csv(self.data_path / 'movies.csv')
            self.df_users = pd.read_csv(self.data_path / 'users.csv')
            print("✓ Data loaded successfully")
            return True
        except FileNotFoundError:
            print("⚠ Data files not found. Using sample data.")
            self._create_sample_data()
            return False
    
    def _create_sample_data(self):
        """Create sample data for demonstration."""
        np.random.seed(42)
        
        # Sample ratings
        n_ratings = 100000
        self.df_ratings = pd.DataFrame({
            'user_id': np.random.randint(1, 10001, n_ratings),
            'movie_id': np.random.randint(1, 5001, n_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.05, 0.1, 0.2, 0.3, 0.35]),
            'timestamp': pd.date_range('2015-01-01', periods=n_ratings, freq='H')
        })
        
        # Sample movies
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
        self.df_movies = pd.DataFrame({
            'movie_id': range(1, 5001),
            'title': [f'Movie_{i}' for i in range(1, 5001)],
            'genre': np.random.choice(genres, 5000),
            'release_year': np.random.randint(1990, 2024, 5000),
            'imdb_rating': np.random.uniform(3.0, 9.0, 5000)
        })
        
        # Sample users
        self.df_users = pd.DataFrame({
            'user_id': range(1, 10001),
            'age': np.random.randint(13, 80, 10000),
            'country': np.random.choice(['USA', 'UK', 'CA', 'AU', 'IN'], 10000)
        })
    
    def basic_statistics(self):
        """Display basic dataset statistics."""
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        print(f"\nRatings:")
        print(f"  - Total ratings: {len(self.df_ratings):,}")
        print(f"  - Unique users: {self.df_ratings['user_id'].nunique():,}")
        print(f"  - Unique movies: {self.df_ratings['movie_id'].nunique():,}")
        print(f"  - Sparsity: {(1 - len(self.df_ratings) / (self.df_ratings['user_id'].nunique() * self.df_ratings['movie_id'].nunique())) * 100:.2f}%")
        
        print(f"\nRating Distribution:")
        rating_dist = self.df_ratings['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            pct = (count / len(self.df_ratings)) * 100
            print(f"  - {int(rating)} stars: {count:,} ({pct:.1f}%)")
        
        print(f"\nAverage Rating: {self.df_ratings['rating'].mean():.2f}")
        print(f"Median Rating: {self.df_ratings['rating'].median():.2f}")
        print(f"Std Dev Rating: {self.df_ratings['rating'].std():.2f}")
    
    def analyze_user_behavior(self):
        """Analyze user behavior patterns."""
        print("\n" + "="*60)
        print("USER BEHAVIOR ANALYSIS")
        print("="*60)
        
        # Ratings per user
        ratings_per_user = self.df_ratings.groupby('user_id').size()
        print(f"\nRatings per User:")
        print(f"  - Average: {ratings_per_user.mean():.2f}")
        print(f"  - Median: {ratings_per_user.median():.2f}")
        print(f"  - Min: {ratings_per_user.min()}")
        print(f"  - Max: {ratings_per_user.max()}")
        print(f"  - 90th percentile: {ratings_per_user.quantile(0.9):.0f}")
        
        # User segments
        print(f"\nUser Segments (by activity):")
        active_users = (ratings_per_user >= ratings_per_user.quantile(0.75)).sum()
        moderate_users = ((ratings_per_user >= ratings_per_user.quantile(0.25)) & (ratings_per_user < ratings_per_user.quantile(0.75))).sum()
        inactive_users = (ratings_per_user < ratings_per_user.quantile(0.25)).sum()
        
        print(f"  - Active (top 25%): {active_users:,}")
        print(f"  - Moderate (25-75%): {moderate_users:,}")
        print(f"  - Inactive (bottom 25%): {inactive_users:,}")
    
    def analyze_movie_popularity(self):
        """Analyze movie popularity patterns."""
        print("\n" + "="*60)
        print("MOVIE POPULARITY ANALYSIS")
        print("="*60)
        
        movie_stats = self.df_ratings.groupby('movie_id').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'num_ratings', 'avg_rating', 'rating_std']
        
        print(f"\nMovie Coverage:")
        print(f"  - Movies with 1+ ratings: {(movie_stats['num_ratings'] >= 1).sum():,}")
        print(f"  - Movies with 10+ ratings: {(movie_stats['num_ratings'] >= 10).sum():,}")
        print(f"  - Movies with 100+ ratings: {(movie_stats['num_ratings'] >= 100).sum():,}")
        
        print(f"\nAverage Ratings per Movie:")
        print(f"  - Mean: {movie_stats['avg_rating'].mean():.2f}")
        print(f"  - Median: {movie_stats['avg_rating'].median():.2f}")
        print(f"  - Top rated: {movie_stats['avg_rating'].max():.2f}")
        print(f"  - Lowest rated: {movie_stats['avg_rating'].min():.2f}")
    
    def run_full_analysis(self):
        """Run complete EDA."""
        self.load_data()
        self.basic_statistics()
        self.analyze_user_behavior()
        self.analyze_movie_popularity()
        print("\n" + "="*60)
        print("✓ EDA COMPLETE")
        print("="*60)


if __name__ == '__main__':
    # Run EDA
    eda = NetflixEDA()
    eda.run_full_analysis()
