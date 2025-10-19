"""
Training Script with Advanced Class Imbalance Handling

This script integrates the Advanced Class Imbalance Handler with your existing
Barcelona parking prediction model to specifically address the poor performance
of classes 3 & 4 (currently 31-33% precision).

Usage:
    python src/modeling/train_with_advanced_imbalance_handling.py
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.impute import SimpleImputer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your existing modules
from src.modeling.advanced_class_imbalance_handler import AdvancedClassImbalanceHandler
from src.modeling.train_main_model import (
    load_data, select_features, create_time_series_split
)
# Import the target variable creation script
from src.modeling.target_variable import create_target_variable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImbalanceAwareTrainer:
    """
    Enhanced trainer that combines your existing pipeline with advanced imbalance handling.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.imbalance_handler = AdvancedClassImbalanceHandler(random_state=42)
        self.results = {}
        self.best_model = None
        self.best_strategy = None
        
    def load_and_prepare_data(self, sample_size: int = 500000) -> tuple:
        """
        Load and prepare data with sampling for faster experimentation.
        """
        logger.info(f"Loading data with sample size: {sample_size:,}")
        
        # Load data using your existing function
        data_path = 'data/processed/features/features_master_table_historical.parquet'
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df):,} rows")
        
        # Sample data for faster processing during development
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled down to {len(df):,} rows")
        
        # Select features using your existing logic
        features = select_features(df)
        logger.info(f"Selected {len(features)} features")
        
        X = df[features].copy()

        # --- Imputation Step ---
        if X.isnull().sum().sum() > 0:
            logger.warning(f"Found {X.isnull().sum().sum()} NaN values in features. Imputing with median.")
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=features)
            logger.info("Imputation complete.")
        # --- End Imputation Step ---

        # --- Correct Target Variable Handling ---
        # The 'actual_state' column appears to be the target variable already.
        possible_target_cols = ['actual_state', 'prediction_code', 'occupancy_level', 'target']
        target_col = next((col for col in possible_target_cols if col in df.columns), None)

        if not target_col:
            raise ValueError(f"Could not find a valid target column in {df.columns.tolist()}")
        
        logger.info(f"Using '{target_col}' as the target variable.")
        y = df[target_col].astype(int).copy()
        # --- End Target Variable Handling ---
        
        logger.info(f"Target column: {target_col}")
        logger.info(f"Class distribution: {y.value_counts().sort_index().to_dict()}")
        
        return X, y, features, target_col
    
    def run_imbalance_analysis(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """
        Run comprehensive class imbalance analysis and find best strategy.
        """
        logger.info("🎯 Starting comprehensive class imbalance analysis...")
        
        # Run the comprehensive evaluation
        results = self.imbalance_handler.run_comprehensive_evaluation(
            X, y, n_splits=3  # Reduced for faster processing
        )
        
        self.results = results
        self.best_strategy = results.get('best_strategy')
        
        # Generate and save report
        report = self.imbalance_handler.generate_improvement_report()
        
        # Save report to file
        report_dir = Path("reports/imbalance_analysis")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"class_imbalance_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {report_path}")
        
        # Create visualizations
        try:
            self.imbalance_handler.plot_improvement_analysis(
                save_path=str(report_dir / f"imbalance_analysis_{timestamp}.png")
            )
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
        
        # Print key results
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        return results
    
    def train_best_model(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        """
        Train model using the best strategy identified by imbalance analysis.
        """
        if not self.best_strategy:
            logger.error("No best strategy found. Run imbalance analysis first.")
            return None, None
        
        logger.info(f"🏆 Training final model with best strategy: {self.best_strategy}")
        
        # Extract strategy components
        strategy_parts = self.best_strategy.split('_')
        sampling_method = strategy_parts[0]
        model_type = '_'.join(strategy_parts[1:])
        
        # Get the best resampling technique
        resampled_datasets = self.imbalance_handler.apply_advanced_smote_variants(X, y)
        
        if sampling_method not in resampled_datasets:
            logger.warning(f"Sampling method {sampling_method} not available. Using conservative_smote.")
            sampling_method = 'conservative_smote'
        
        X_resampled, y_resampled = resampled_datasets[sampling_method]
        
        # Get class weights
        class_weights = self.imbalance_handler.calculate_dynamic_class_weights(y)
        
        # Create and train the best model
        models = self.imbalance_handler.create_cost_sensitive_models(class_weights)
        
        if model_type not in models:
            logger.warning(f"Model type {model_type} not available. Using lgb_cost_sensitive.")
            model_type = 'lgb_cost_sensitive'
        
        model = models[model_type]
        
        # Train the model
        logger.info("Training final model on resampled data...")
        model.fit(X_resampled, y_resampled)
        
        # Evaluate on original test set (using time series split)
        tscv = create_time_series_split(pd.DataFrame(X))
        test_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            y_pred = model.predict(X_test)
            fold_eval = self.imbalance_handler.evaluate_class_specific_performance(y_test, y_pred)
            test_results.append(fold_eval)
            
            logger.info(f"Fold {fold+1} - Accuracy: {fold_eval['accuracy']:.3f}, "
                       f"Class 3 F1: {fold_eval['class_3_metrics']['f1']:.3f}, "
                       f"Class 4 F1: {fold_eval['class_4_metrics']['f1']:.3f}")
        
        # Calculate final metrics
        final_metrics = {
            'strategy': self.best_strategy,
            'sampling_method': sampling_method,
            'model_type': model_type,
            'accuracy': np.mean([r['accuracy'] for r in test_results]),
            'class_3_f1': np.mean([r['class_3_metrics']['f1'] for r in test_results]),
            'class_4_f1': np.mean([r['class_4_metrics']['f1'] for r in test_results]),
            'problematic_classes_f1': np.mean([r['problematic_classes_f1'] for r in test_results]),
            'training_samples': len(X_resampled),
            'original_samples': len(X)
        }
        
        self.best_model = model
        
        logger.info("🎉 Final model training completed!")
        logger.info(f"   Strategy: {self.best_strategy}")
        logger.info(f"   Accuracy: {final_metrics['accuracy']:.3f}")
        logger.info(f"   Class 3 F1: {final_metrics['class_3_f1']:.3f}")
        logger.info(f"   Class 4 F1: {final_metrics['class_4_f1']:.3f}")
        
        return model, final_metrics
    
    def save_model_and_results(self, model, metrics: dict, features: list):
        """
        Save the trained model and results.
        """
        # Create output directories
        model_dir = Path("models/imbalance_improved")
        metrics_dir = Path("reports/metrics/imbalance_improved")
        
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = model_dir / f"best_imbalance_model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"💾 Model saved to: {model_path}")
        
        # Save metrics
        metrics_with_features = metrics.copy()
        metrics_with_features['features'] = features
        metrics_with_features['timestamp'] = timestamp
        
        metrics_path = metrics_dir / f"imbalance_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_with_features, f, indent=4)
        logger.info(f"📊 Metrics saved to: {metrics_path}")
        
        # Save feature list
        features_path = metrics_dir / f"features_{timestamp}.json"
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=4)
        
        return model_path, metrics_path
    
    def run_complete_pipeline(self, sample_size: int = 500000) -> dict:
        """
        Run the complete imbalance-aware training pipeline.
        """
        logger.info("🚀 Starting complete imbalance-aware training pipeline...")
        
        # Step 1: Load and prepare data
        X, y, features, target_col = self.load_and_prepare_data(sample_size)
        
        # Step 2: Run imbalance analysis
        analysis_results = self.run_imbalance_analysis(X, y)
        
        # Step 3: Train best model
        model, final_metrics = self.train_best_model(X, y)
        
        if model is None:
            logger.error("Model training failed!")
            return None
        
        # Step 4: Save results
        model_path, metrics_path = self.save_model_and_results(model, final_metrics, features)
        
        # Step 5: Summary
        summary = {
            'pipeline_completed': True,
            'best_strategy': self.best_strategy,
            'final_metrics': final_metrics,
            'model_path': str(model_path),
            'metrics_path': str(metrics_path),
            'data_samples': len(X),
            'features_count': len(features),
            'target_column': target_col
        }
        
        # Print final summary
        print("\n" + "🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉".center(60, "="))
        print(f"✅ Best Strategy: {self.best_strategy}")
        print(f"✅ Final Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"✅ Class 3 F1 Score: {final_metrics['class_3_f1']:.3f}")
        print(f"✅ Class 4 F1 Score: {final_metrics['class_4_f1']:.3f}")
        print(f"✅ Model saved to: {model_path}")
        print("="*60)
        
        return summary


def main():
    """
    Main function to run the imbalance-aware training pipeline.
    """
    logger.info("Starting Advanced Class Imbalance Training Pipeline")
    
    # Initialize trainer
    trainer = ImbalanceAwareTrainer()
    
    # Run complete pipeline
    try:
        # Use smaller sample for testing, increase for production
        sample_size = 500000  # Adjust based on your computational resources
        
        summary = trainer.run_complete_pipeline(sample_size=sample_size)
        
        if summary and summary['pipeline_completed']:
            logger.info("✅ Pipeline completed successfully!")
            
            # Print key recommendations
            print("\n🔧 NEXT STEPS:")
            print("1. Review the generated reports in reports/imbalance_analysis/")
            print("2. Test the improved model on new data")
            print("3. Consider deploying the best strategy in production")
            print("4. Monitor class-specific performance metrics")
            
            # Check if significant improvement was achieved
            class_3_improvement = summary['final_metrics']['class_3_f1'] - 0.36
            class_4_improvement = summary['final_metrics']['class_4_f1'] - 0.34
            
            if class_3_improvement > 0.05 and class_4_improvement > 0.05:
                print("🎯 SIGNIFICANT IMPROVEMENT achieved for problematic classes!")
            elif class_3_improvement > 0 or class_4_improvement > 0:
                print("⚠️  PARTIAL IMPROVEMENT - consider additional strategies")
            else:
                print("❌ LIMITED IMPROVEMENT - may need different approach")
                print("   Consider: More data, different features, or ensemble methods")
        
        else:
            logger.error("❌ Pipeline failed!")
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 