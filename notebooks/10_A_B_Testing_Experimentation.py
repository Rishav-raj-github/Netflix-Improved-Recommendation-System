"""
10_A_B_Testing_Experimentation.py

A/B Testing and statistical experimentation for Netflix recommendations.
Demonstrates hypothesis testing, power analysis, and experiment design.
"""

import numpy as np
from scipy import stats

class ABTestingNotebook:
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def conduct_ttest(self, control_metrics, variant_metrics):
        t_stat, p_value = stats.ttest_ind(control_metrics, variant_metrics)
        return {'t_statistic': t_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    
    def calculate_effect_size(self, control_mean, variant_mean, pooled_std):
        cohens_d = (variant_mean - control_mean) / pooled_std
        return cohens_d
    
    def sample_size_calculation(self, alpha=0.05, power=0.80):
        # Based on standard effect size
        return int(((1.96 + 1.282) ** 2 * 2) / (0.2 ** 2))
    
    def analyze_experiment(self):
        print("\n" + "="*80)
        print("A/B TESTING & EXPERIMENTATION")
        print("="*80)
        
        control = np.array([4.2, 4.1, 4.3, 4.0, 4.2] * 200)
        variant = np.array([4.35, 4.25, 4.40, 4.15, 4.30] * 200)
        
        result = self.conduct_ttest(control, variant)
        print(f"\nt-test Results:")
        print(f"  t-statistic: {result['t_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f}")
        print(f"  Statistically significant: {result['significant']}")
        
        effect_size = self.calculate_effect_size(control.mean(), variant.mean(), np.std(control))
        print(f"\nEffect Size (Cohen's d): {effect_size:.4f}")
        
        sample_size = self.sample_size_calculation()
        print(f"Required sample size: {sample_size}")

if __name__ == '__main__':
    notebook = ABTestingNotebook()
    notebook.analyze_experiment()
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Statistical significance requires adequate sample size")
    print("2. Effect size matters as much as statistical significance")
    print("3. Multiple testing correction necessary for multiple experiments")
    print("="*80 + "\n")
