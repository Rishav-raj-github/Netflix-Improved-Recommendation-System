"""
Netflix Recommendation Service API

Production-grade microservice for real-time personalized recommendations.
Features: Collaborative filtering, content-based filtering, hybrid approach.

Author: Rishav Raj
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import redis
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(title="Netflix Recommendation Service", version="1.0.0")
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class User(BaseModel):
    user_id: int
    watch_history: List[int]
    ratings: Dict[int, float]
    metadata: Optional[Dict] = None

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    score: float
    reason: str
    genre: str

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10
    algorithm: str = 'hybrid'  # 'collaborative', 'content', 'hybrid'
    filters: Optional[Dict] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    generated_at: datetime
    algorithm_used: str
    execution_time_ms: float

class RecommendationEngine:
    """
    Core recommendation engine.
    """
    def __init__(self):
        self.user_cache = {}
        self.model_version = '1.0'
    
    def get_collaborative_filtering_recommendations(
        self,
        user_id: int,
        num_recommendations: int
    ) -> List[tuple]:
        """
        Collaborative filtering recommendations using user similarity.
        """
        # Fetch user embedding from cache or compute
        user_embedding = self._get_user_embedding(user_id)
        
        # Find similar users
        similar_users = self._find_similar_users(user_embedding, k=10)
        
        # Get movies watched by similar users
        recommendations = []
        for similar_user, similarity_score in similar_users:
            watched_movies = self._get_watched_movies(similar_user)
            for movie, rating in watched_movies:
                if movie not in self._get_watched_movies(user_id):
                    recommendations.append((
                        movie,
                        rating * similarity_score
                    ))
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
    
    def get_content_based_recommendations(
        self,
        user_id: int,
        num_recommendations: int
    ) -> List[tuple]:
        """
        Content-based filtering using movie attributes.
        """
        # Get user's favorite genres and attributes
        user_profile = self._build_user_profile(user_id)
        
        # Find similar movies
        similar_movies = self._find_similar_movies(
            user_profile,
            exclude_watched=True,
            user_id=user_id
        )
        
        return similar_movies[:num_recommendations]
    
    def get_hybrid_recommendations(
        self,
        user_id: int,
        num_recommendations: int,
        cf_weight: float = 0.6,
        cb_weight: float = 0.4
    ) -> List[tuple]:
        """
        Hybrid approach combining collaborative and content-based.
        """
        cf_recs = self.get_collaborative_filtering_recommendations(
            user_id,
            num_recommendations * 2
        )
        cb_recs = self.get_content_based_recommendations(
            user_id,
            num_recommendations * 2
        )
        
        # Merge and weight scores
        merged = {}
        for movie, score in cf_recs:
            merged[movie] = merged.get(movie, 0) + cf_weight * score
        
        for movie, score in cb_recs:
            merged[movie] = merged.get(movie, 0) + cb_weight * score
        
        # Sort by combined score
        hybrid_recs = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return hybrid_recs[:num_recommendations]
    
    def _get_user_embedding(self, user_id: int) -> np.ndarray:
        """Retrieve user embedding."""
        cache_key = f"user_embedding:{user_id}"
        cached = redis_client.get(cache_key)
        if cached:
            return np.array([float(x) for x in cached.split(',')])
        # Compute if not in cache
        return np.random.randn(128)  # Placeholder
    
    def _find_similar_users(self, embedding: np.ndarray, k: int):
        """Find k most similar users."""
        return [(1, 0.9), (2, 0.85), (3, 0.80)]  # Placeholder
    
    def _get_watched_movies(self, user_id: int):
        """Get user's watch history."""
        return [(1, 4.5), (2, 4.0)]  # Placeholder
    
    def _build_user_profile(self, user_id: int):
        """Build user preference profile."""
        return {'action': 0.8, 'drama': 0.6, 'comedy': 0.7}
    
    def _find_similar_movies(self, profile: Dict, exclude_watched: bool, user_id: int):
        """Find movies similar to user preferences."""
        return [(10, 0.85), (11, 0.80), (12, 0.75)]  # Placeholder

recommendation_engine = RecommendationEngine()

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user.
    """
    start_time = datetime.now()
    
    try:
        if request.algorithm == 'collaborative':
            recs = recommendation_engine.get_collaborative_filtering_recommendations(
                request.user_id,
                request.num_recommendations
            )
        elif request.algorithm == 'content':
            recs = recommendation_engine.get_content_based_recommendations(
                request.user_id,
                request.num_recommendations
            )
        else:  # hybrid
            recs = recommendation_engine.get_hybrid_recommendations(
                request.user_id,
                request.num_recommendations
            )
        
        recommendations = [
            MovieRecommendation(
                movie_id=movie_id,
                title=f"Movie {movie_id}",
                score=float(score),
                reason=f"Similar to your preferences",
                genre="Drama"
            )
            for movie_id, score in recs
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            generated_at=datetime.now(),
            algorithm_used=request.algorithm,
            execution_time_ms=execution_time
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(
    user_id: int,
    movie_id: int,
    rating: float,
    background_tasks: BackgroundTasks
):
    """
    Record user feedback for model training.
    """
    background_tasks.add_task(store_feedback, user_id, movie_id, rating)
    return {"status": "feedback_received"}

async def store_feedback(user_id: int, movie_id: int, rating: float):
    """
    Store feedback asynchronously.
    """
    redis_client.hincrby(f"user:{user_id}:ratings", movie_id, rating)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
