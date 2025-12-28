"""
07_Visualization_Analysis.py

Visualization and Analysis of Recommendation Results

This notebook demonstrates visualization techniques for analyzing
recommendation system performance and user behavior patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VisualizationAnalysisNotebook:
    """
    Visualization and analysis tools for recommendation systems.
    """
    
    def __init__(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Initialize visualization tools.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Predicted ratings
        ground_truth : np.ndarray
            Ground truth ratings
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.residuals = ground_truth - predictions
        
    def analyze_residuals(self) -> Dict[str, float]:
        """
        Analyze prediction residuals.
        
        Returns:
        --------
        Dict[str, float]
            Residual statistics
        """
        print("Analyzing residuals...")
        stats = {
            'mean_residual': float(np.mean(self.residuals)),
            'std_residual': float(np.std(self.residuals)),
            'max_abs_residual': float(np.max(np.abs(self.residuals))),
            'median_residual': float(np.median(self.residuals)),
        }
        print(f"  Mean Residual: {stats['mean_residual']:.4f}")
        print(f"  Std Residual: {stats['std_residual']:.4f}")
        print(f"  Max Abs Residual: {stats['max_abs_residual']:.4f}")
        return stats
    
    def analyze_prediction_distribution(self) -> Dict[str, float]:
        """
        Analyze distribution of predictions.
        
        Returns:
        --------
        Dict[str, float]
            Distribution statistics
        """
        print("Analyzing prediction distribution...")
        stats = {
            'pred_mean': float(np.mean(self.predictions)),
            'pred_std': float(np.std(self.predictions)),
            'pred_min': float(np.min(self.predictions)),
            'pred_max': float(np.max(self.predictions)),
            'pred_quartile_25': float(np.percentile(self.predictions, 25)),
            'pred_quartile_75': float(np.percentile(self.predictions, 75)),
        }
        print(f"  Prediction Mean: {stats['pred_mean']:.4f}")
        print(f"  Prediction Std: {stats['pred_std']:.4f}")
        print(f"  Prediction Range: [{stats['pred_min']:.4f}, {stats['pred_max']:.4f}]")
        return stats
    
    def analyze_by_rating_group(self, ground_truth: np.ndarray = None) -> Dict[float, Dict]:
        """
        Analyze predictions grouped by rating level.
        
        Parameters:
        -----------
        ground_truth : np.ndarray, optional
            Ground truth ratings for grouping
            
        Returns:
        --------
        Dict[float, Dict]
            Analysis by rating group
        """
        if ground_truth is None:
            ground_truth = self.ground_truth
            
        print("Analyzing by rating groups...")
        rating_groups = {}
        
        for rating in sorted(np.unique(ground_truth)):
            mask = ground_truth == rating
            group_predictions = self.predictions[mask]
            group_residuals = self.residuals[mask]
            
            rating_groups[rating] = {
                'count': int(np.sum(mask)),
                'pred_mean': float(np.mean(group_predictions)),
                'pred_std': float(np.std(group_predictions)),
                'residual_mean': float(np.mean(group_residuals)),
                'residual_std': float(np.std(group_residuals)),
            }
            print(f"  Rating {rating}: {rating_groups[rating]['count']} samples, Pred Mean: {rating_groups[rating]['pred_mean']:.4f}")
        
        return rating_groups
    
    def predict_quality_assessment(self) -> Dict[str, float]:
        """
        Assess overall prediction quality.
        
        Returns:
        --------
        Dict[str, float]
            Quality metrics
        """
        print("Assessing prediction quality...")
        
        mse = np.mean(self.residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.residuals))
        
        # Calibration: predictions should align with ground truth distribution
        calibration_error = np.abs(np.mean(self.predictions) - np.mean(self.ground_truth))
        
        # Sharpness: variance of predictions
        sharpness = np.std(self.predictions)
        
        quality_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'calibration_error': float(calibration_error),
            'sharpness': float(sharpness),
        }
        
        print(f"  MSE: {quality_metrics['mse']:.4f}")
        print(f"  RMSE: {quality_metrics['rmse']:.4f}")
        print(f"  Calibration Error: {quality_metrics['calibration_error']:.4f}")
        
        return quality_metrics
    
    def user_satisfaction_analysis(self) -> Dict[str, float]:
        """
        Analyze potential user satisfaction based on predictions.
        
        Returns:
        --------
        Dict[str, float]
            Satisfaction metrics
        """
        print("Analyzing user satisfaction...")
        
        # Users are satisfied when predictions are close to ground truth
        prediction_accuracy = 1 - (np.mean(np.abs(self.residuals)) / 5)  # Normalize by max rating
        prediction_accuracy = np.clip(prediction_accuracy, 0, 1)
        
        # Overprediction can disappoint, underprediction can delight
        overprediction_ratio = np.sum(self.predictions > self.ground_truth) / len(self.predictions)
        
        satisfaction_metrics = {
            'prediction_accuracy': float(prediction_accuracy),
            'overprediction_ratio': float(overprediction_ratio),
            'disappointment_risk': float(overprediction_ratio),  # High overprediction = high disappointment risk
        }
        
        print(f"  Prediction Accuracy: {satisfaction_metrics['prediction_accuracy']:.4f}")
        print(f"  Overprediction Ratio: {satisfaction_metrics['overprediction_ratio']:.4f}")
        
        return satisfaction_metrics
    
    def generate_visualization_data(self) -> Dict[str, np.ndarray]:
        """
        Generate data suitable for visualization.
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Data for various plots
        """
        visualization_data = {
            'predictions': self.predictions,
            'ground_truth': self.ground_truth,
            'residuals': self.residuals,
            'prediction_indices': np.arange(len(self.predictions)),
        }
        return visualization_data

def main():
    """
    Example usage of Visualization and Analysis.
    """
    print("="*60)
    print("Visualization and Analysis of Recommendation Results")
    print("="*60)
    
    # Generate dummy data
    np.random.seed(42)
    ground_truth = np.array([5, 4, 3, 5, 4, 5, 3, 4, 4, 5])
    predictions = ground_truth + np.random.normal(0, 0.5, len(ground_truth))
    predictions = np.clip(predictions, 1, 5)
    
    # Initialize analyzer
    analyzer = VisualizationAnalysisNotebook(predictions, ground_truth)
    
    print("\nAnalyzing Residuals:")
    residual_stats = analyzer.analyze_residuals()
    
    print("\nAnalyzing Prediction Distribution:")
    dist_stats = analyzer.analyze_prediction_distribution()
    
    print("\nAnalyzing by Rating Groups:")
    group_analysis = analyzer.analyze_by_rating_group()
    
    print("\nAssessing Prediction Quality:")
    quality = analyzer.predict_quality_assessment()
    
    print("\nAnalyzing User Satisfaction:")
    satisfaction = analyzer.user_satisfaction_analysis()
    
    print("\nVisualization and Analysis Notebook Complete!")

if __name__ == "__main__":
    main()
