"""FastAPI application for Netflix recommendation system.

This module provides a REST API with endpoints for getting recommendations
and interacting with the trained recommendation model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.model import MatrixFactorization

# Initialize FastAPI app
app = FastAPI(
    title="Netflix Recommendation API",
    description="API for getting personalized movie/show recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable (loaded at startup)
model = None
model_config = {
    'model_path': 'models/saved/matrix_factorization.pth',
    'n_users': 1000,  # Update these based on your data
    'n_items': 500,
    'embedding_dim': 50
}


class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    user_id: int = Field(..., description="User ID to get recommendations for", ge=0)
    n_recommendations: int = Field(10, description="Number of recommendations to return", ge=1, le=100)


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: int
    recommendations: List[dict]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global model
    
    try:
        print("Loading recommendation model...")
        model = MatrixFactorization(
            n_users=model_config['n_users'],
            n_items=model_config['n_items'],
            embedding_dim=model_config['embedding_dim']
        )
        
        # Load model weights if they exist
        model_path = Path(model_config['model_path'])
        if model_path.exists():
            model.load_model(str(model_path))
            model.eval()
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using untrained model.")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Netflix Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommendations": "/recommend",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user.
    
    Args:
        request: RecommendationRequest with user_id and n_recommendations
        
    Returns:
        RecommendationResponse with list of recommended items
        
    Raises:
        HTTPException: If model is not loaded or invalid user_id
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate user_id
    if request.user_id >= model_config['n_users']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_id. Must be less than {model_config['n_users']}"
        )
    
    try:
        # Get recommendations from model
        recommendations = model.predict(
            user_id=request.user_id,
            n_items=request.n_recommendations
        )
        
        # Format response
        formatted_recs = [
            {
                "item_id": int(item_id),
                "score": float(score),
                "rank": idx + 1
            }
            for idx, (item_id, score) in enumerate(recommendations)
        ]
        
        return {
            "user_id": request.user_id,
            "recommendations": formatted_recs
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.get("/user/{user_id}/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_user_recommendations(user_id: int, n: int = 10):
    """Get recommendations for a user (GET endpoint alternative).
    
    Args:
        user_id: User ID to get recommendations for
        n: Number of recommendations to return (default: 10)
        
    Returns:
        RecommendationResponse with list of recommended items
    """
    request = RecommendationRequest(user_id=user_id, n_recommendations=n)
    return await get_recommendations(request)


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "Matrix Factorization",
        "n_users": model_config['n_users'],
        "n_items": model_config['n_items'],
        "embedding_dim": model_config['embedding_dim'],
        "parameters": sum(p.numel() for p in model.parameters())
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Netflix Recommendation API...")
    print("API documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
