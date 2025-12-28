"""
06_Evaluation_Metrics.py

Evaluation Metrics for Recommendation Systems

Comprehensive evaluation of recommendation system performance using
various metrics including RMSE, MAE, precision, recall, and ranking metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EvaluationMetricsNotebook:
    """
    Comprehensive evaluation metrics for recommendation systems.
    """
    
    def __init__(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Initialize evaluation metrics.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Predicted ratings
        ground_truth : np.ndarray
            Ground truth ratings
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.metrics = {}
        
    def calculate_rmse(self) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Returns:
        --------
        float
            RMSE value
        """
        rmse = np.sqrt(mean_squared_error(self.ground_truth, self.predictions))
        self.metrics['rmse'] = rmse
        print(f"RMSE: {rmse:.4f}")
        return rmse
    
    def calculate_mae(self) -> float:
        """
        Calculate Mean Absolute Error.
        
        Returns:
        --------
        float
            MAE value
        """
        mae = mean_absolute_error(self.ground_truth, self.predictions)
        self.metrics['mae'] = mae
        print(f"MAE: {mae:.4f}")
        return mae
    
    def calculate_precision_at_k(self, k: int = 5) -> float:
        """
        Calculate Precision@K.
        
        Parameters:
        -----------
        k : int
            Number of top items
            
        Returns:
        --------
        float
            Precision@K value
        """
        top_k_indices = np.argsort(self.predictions)[::-1][:k]
        top_k_truth = self.ground_truth[top_k_indices]
        precision = np.mean(top_k_truth > 3)  # Consider rating > 3 as positive
        self.metrics[f'precision@{k}'] = precision
        print(f"Precision@{k}: {precision:.4f}")
        return precision
    
    def calculate_recall_at_k(self, k: int = 5) -> float:
        """
        Calculate Recall@K.
        
        Parameters:
        -----------
        k : int
            Number of top items
            
        Returns:
        --------
        float
            Recall@K value
        """
        positive_indices = np.where(self.ground_truth > 3)[0]
        top_k_indices = np.argsort(self.predictions)[::-1][:k]
        recalled = len(set(top_k_indices) & set(positive_indices))
        recall = recalled / max(len(positive_indices), 1)
        self.metrics[f'recall@{k}'] = recall
        print(f"Recall@{k}: {recall:.4f}")
        return recall
    
    def calculate_ndcg(self, k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Parameters:
        -----------
        k : int
            Number of top items
            
        Returns:
        --------
        float
            NDCG value
        """
        top_k_indices = np.argsort(self.predictions)[::-1][:k]
        dcg = 0
        for i, idx in enumerate(top_k_indices):
            dcg += (2 ** self.ground_truth[idx] - 1) / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_indices = np.argsort(self.ground_truth)[::-1][:k]
        idcg = 0
        for i, idx in enumerate(ideal_indices):
            idcg += (2 ** self.ground_truth[idx] - 1) / np.log2(i + 2)
        
        ndcg = dcg / max(idcg, 1)
        self.metrics[f'ndcg@{k}'] = ndcg
        print(f"NDCG@{k}: {ndcg:.4f}")
        return ndcg
    
    def calculate_mrr(self) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Returns:
        --------
        float
            MRR value
        """
        relevant_indices = np.where(self.ground_truth > 3)[0]
        sorted_indices = np.argsort(self.predictions)[::-1]
        
        for rank, idx in enumerate(sorted_indices):
            if idx in relevant_indices:
                mrr = 1 / (rank + 1)
                self.metrics['mrr'] = mrr
                print(f"MRR: {mrr:.4f}")
                return mrr
        
        self.metrics['mrr'] = 0.0
        print("MRR: 0.0000")
        return 0.0
    
    def calculate_coverage(self, total_items: int) -> float:
        """
        Calculate catalog coverage.
        
        Parameters:
        -----------
        total_items : int
            Total number of items in catalog
            
        Returns:
        --------
        float
            Coverage percentage
        """
        recommended_items = len(np.where(self.predictions > 0)[0])
        coverage = recommended_items / max(total_items, 1)
        self.metrics['coverage'] = coverage
        print(f"Coverage: {coverage:.4f}")
        return coverage
    
    def get_all_metrics(self, k: int = 5, total_items: int = 100) -> Dict[str, float]:
        """
        Calculate all metrics at once.
        
        Parameters:
        -----------
        k : int
            Number of top items for ranking metrics
        total_items : int
            Total items for coverage calculation
            
        Returns:
        --------
        Dict[str, float]
            All calculated metrics
        """
        print("Calculating all evaluation metrics...")
        self.calculate_rmse()
        self.calculate_mae()
        self.calculate_precision_at_k(k)
        self.calculate_recall_at_k(k)
        self.calculate_ndcg(k)
        self.calculate_mrr()
        self.calculate_coverage(total_items)
        
        return self.metrics

def main():
    """
    Example usage of Evaluation Metrics.
    """
    print("="*60)
    print("Evaluation Metrics for Recommendation Systems")
    print("="*60)
    
    # Generate dummy predictions and ground truth
    np.random.seed(42)
    ground_truth = np.array([5, 4, 3, 5, 4, 5, 3, 4, 4, 5])
    predictions = ground_truth + np.random.normal(0, 0.5, len(ground_truth))
    predictions = np.clip(predictions, 1, 5)
    
    # Initialize evaluator
    evaluator = EvaluationMetricsNotebook(predictions, ground_truth)
    
    # Calculate metrics
    print("\nCalculating Metrics...")
    all_metrics = evaluator.get_all_metrics(k=5, total_items=100)
    
    print("\nAll Metrics Summary:")
    for metric_name, value in all_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("\nEvaluation Metrics Notebook Complete!")

if __name__ == "__main__":
    main()
