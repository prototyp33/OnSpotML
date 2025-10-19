"""
Unit Tests for Advanced Class Imbalance Handler

This script contains unit tests for the AdvancedClassImbalanceHandler class
to ensure its methods for handling class imbalance work correctly. It uses
synthetic data to validate the functionality of each component.
"""

import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.modeling.advanced_class_imbalance_handler import AdvancedClassImbalanceHandler

# Create dummy data for testing
def create_imbalanced_data(rows=1000, n_features=10):
    """Creates an imbalanced dataset for testing purposes."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(rows, n_features), columns=[f'feature_{i}' for i in range(n_features)])
    # Imbalanced classes similar to the problem description
    y = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6], 
        size=rows, 
        p=[0.25, 0.20, 0.20, 0.05, 0.05, 0.1, 0.15]
    )
    return X, y

class TestAdvancedClassImbalanceHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class with dummy data and handler instance."""
        cls.X, cls.y = create_imbalanced_data()
        cls.handler = AdvancedClassImbalanceHandler(random_state=42)
    
    def test_01_analyze_class_distribution(self):
        """Test the analysis of class distribution."""
        analysis = self.handler.analyze_class_distribution(self.y)
        self.assertIn('class_counts', analysis)
        self.assertIn('imbalance_ratio', analysis)
        self.assertIn('problematic_classes', analysis)
        self.assertEqual(analysis['problematic_classes'], [3, 4])
        self.assertTrue(analysis['imbalance_ratio'] > 1)

    def test_02_create_adaptive_sampling_strategy(self):
        """Test the creation of an adaptive sampling strategy."""
        strategy = self.handler.create_adaptive_sampling_strategy(self.y)
        self.assertIsInstance(strategy, dict)
        # Check that problematic classes are targeted
        self.assertIn(3, strategy)
        self.assertIn(4, strategy)
        self.assertTrue(all(isinstance(v, int) for v in strategy.values()))

    def test_03_apply_advanced_smote_variants(self):
        """Test the application of various SMOTE variants."""
        resampled_data = self.handler.apply_advanced_smote_variants(self.X, self.y)
        self.assertIsInstance(resampled_data, dict)
        # Check if at least one strategy succeeded
        self.assertTrue(len(resampled_data) > 0)
        
        # Check a specific strategy
        if 'adasyn' in resampled_data:
            X_res, y_res = resampled_data['adasyn']
            # The output can be a DataFrame or numpy array
            self.assertTrue(isinstance(X_res, (np.ndarray, pd.DataFrame)))
            self.assertIsInstance(y_res, np.ndarray)
            self.assertTrue(len(X_res) > len(self.X))
            self.assertTrue(len(y_res) > len(self.y))

    def test_04_calculate_dynamic_class_weights(self):
        """Test the calculation of dynamic class weights."""
        weights = self.handler.calculate_dynamic_class_weights(self.y, focus_classes=[3, 4])
        self.assertIsInstance(weights, dict)
        # Check that focus classes have higher weights
        self.assertTrue(weights[3] > weights[0])
        self.assertTrue(weights[4] > weights[0])
        
    def test_05_create_cost_sensitive_models(self):
        """Test the creation of cost-sensitive models."""
        class_weights = self.handler.calculate_dynamic_class_weights(self.y)
        models = self.handler.create_cost_sensitive_models(class_weights)
        self.assertIsInstance(models, dict)
        self.assertIn('lgb_cost_sensitive', models)
        self.assertIn('balanced_rf', models)
        self.assertTrue(hasattr(models['lgb_cost_sensitive'], 'fit'))
        
    def test_06_evaluate_class_specific_performance(self):
        """Test the class-specific performance evaluation."""
        y_true = np.array([0, 1, 2, 3, 4, 3, 4])
        y_pred = np.array([0, 1, 2, 0, 1, 3, 4])
        evaluation = self.handler.evaluate_class_specific_performance(y_true, y_pred)
        
        self.assertIn('accuracy', evaluation)
        self.assertIn('class_3_metrics', evaluation)
        self.assertIn('class_4_metrics', evaluation)
        self.assertIn('problematic_classes_f1', evaluation)
        self.assertAlmostEqual(evaluation['class_3_metrics']['f1'], 0.66666, places=4)
        self.assertAlmostEqual(evaluation['class_4_metrics']['f1'], 0.66666, places=4)

    def test_07_run_comprehensive_evaluation(self):
        """Test the comprehensive evaluation pipeline."""
        # Use a smaller dataset for this test to speed it up
        X_small, y_small = create_imbalanced_data(rows=200, n_features=5)
        
        results = self.handler.run_comprehensive_evaluation(X_small, y_small, n_splits=2)
        
        self.assertIn('strategy_results', results)
        self.assertIn('best_strategy', results)
        self.assertIn('best_performance', results)
        self.assertIsNotNone(results['best_strategy'])
        self.assertTrue(len(results['strategy_results']) > 0)

    def test_08_generate_improvement_report(self):
        """Test the generation of the improvement report."""
        # First, run evaluation to populate results
        X_small, y_small = create_imbalanced_data(rows=200, n_features=5)
        self.handler.run_comprehensive_evaluation(X_small, y_small, n_splits=2)
        
        report = self.handler.generate_improvement_report()
        self.assertIsInstance(report, str)
        self.assertIn("CLASS IMBALANCE IMPROVEMENT REPORT", report)
        self.assertIn("BEST STRATEGY", report)
        self.assertIn("RECOMMENDATIONS", report)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 