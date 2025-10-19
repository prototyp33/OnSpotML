"""
Performance Gap Diagnostic and Resolution Tool

This script helps identify and address the performance gap between 
baseline (RandomForest) and main (LightGBM) models.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.feature_engineering_v2 import FeatureEngineeringPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceGapDiagnostic:
    """Comprehensive diagnostic tool for model performance gaps."""
    
    def __init__(self, data_path, config_path=None):
        self.data_path = data_path
        self.config_path = config_path or 'config/model_config.yaml'
        self.results = {}
        
    def load_data(self, sample_size=100000):
        """Load and sample data for comparison."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            df = pd.read_parquet(self.data_path)
            
            # Sample for faster comparison
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled {sample_size} rows from {len(df)} total rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_baseline_features(self, df):
        """Create simple baseline features (temporal only)."""
        logger.info("Creating baseline features...")
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        df['month'] = df['timestamp'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        baseline_features = [
            'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
            'month_sin', 'month_cos', 'is_weekend'
        ]
        
        return df, baseline_features
    
    def create_main_features(self, df, config):
        """Create full feature engineering pipeline features."""
        logger.info("Creating main model features...")
        
        try:
            pipeline = FeatureEngineeringPipeline(config)
            df_features = pipeline.fit_transform(df.copy())
            
            # Get feature names (exclude target and meta columns)
            exclude_cols = ['timestamp', 'parking_id', 'occupancy_level', 'prediction_code']
            main_features = [col for col in df_features.columns if col not in exclude_cols]
            
            return df_features, main_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            # Fallback to baseline features if pipeline fails
            return self.create_baseline_features(df)
    
    def compare_algorithms_same_features(self, df, features, target='occupancy_level'):
        """Compare RandomForest vs LightGBM on same features."""
        logger.info("Comparing algorithms with same features...")
        
        X = df[features]
        y = df[target]
        
        # Remove any NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3, test_size=len(X)//5)
        
        # RandomForest
        rf_scores = []
        lgb_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # RandomForest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_scores.append(accuracy_score(y_test, rf_pred))
            
            # LightGBM
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            lgb_scores.append(accuracy_score(y_test, lgb_pred))
        
        return {
            'randomforest_scores': rf_scores,
            'lightgbm_scores': lgb_scores,
            'rf_mean': np.mean(rf_scores),
            'lgb_mean': np.mean(lgb_scores),
            'features_used': len(features)
        }
    
    def compare_feature_complexity(self, df, baseline_features, main_features, target='occupancy_level'):
        """Compare simple vs complex features using same algorithm."""
        logger.info("Comparing feature complexity...")
        
        # Use LightGBM for fair comparison
        results = {}
        
        for feature_set_name, features in [('baseline', baseline_features), ('main', main_features)]:
            logger.info(f"Testing {feature_set_name} features ({len(features)} features)")
            
            # Check if features exist
            available_features = [f for f in features if f in df.columns]
            if len(available_features) != len(features):
                logger.warning(f"Missing features in {feature_set_name}: {set(features) - set(available_features)}")
            
            X = df[available_features]
            y = df[target]
            
            # Remove NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                logger.error(f"No valid data for {feature_set_name} features")
                continue
            
            # Time series CV
            tscv = TimeSeriesSplit(n_splits=3, test_size=len(X)//5)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    verbose=-1
                )
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            results[feature_set_name] = {
                'scores': scores,
                'mean_accuracy': np.mean(scores),
                'std_accuracy': np.std(scores),
                'num_features': len(available_features)
            }
        
        return results
    
    def diagnose_class_imbalance(self, df, target='occupancy_level'):
        """Analyze class distribution and its impact."""
        logger.info("Diagnosing class imbalance...")
        
        class_counts = df[target].value_counts().sort_index()
        class_proportions = class_counts / len(df)
        
        # Calculate imbalance ratio
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class
        
        return {
            'class_counts': class_counts.to_dict(),
            'class_proportions': class_proportions.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'num_classes': len(class_counts)
        }
    
    def test_hyperparameter_sensitivity(self, df, features, target='occupancy_level'):
        """Test if hyperparameters are causing the performance gap."""
        logger.info("Testing hyperparameter sensitivity...")
        
        X = df[features]
        y = df[target]
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Test different hyperparameter configurations
        configs = [
            {'name': 'default', 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31}},
            {'name': 'conservative', 'params': {'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 15}},
            {'name': 'aggressive', 'params': {'n_estimators': 200, 'learning_rate': 0.15, 'num_leaves': 63}},
            {'name': 'regularized', 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 
                                             'reg_alpha': 0.1, 'reg_lambda': 0.1}},
        ]
        
        results = {}
        tscv = TimeSeriesSplit(n_splits=3, test_size=len(X)//5)
        
        for config in configs:
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model = lgb.LGBMClassifier(random_state=42, verbose=-1, **config['params'])
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            results[config['name']] = {
                'scores': scores,
                'mean_accuracy': np.mean(scores),
                'params': config['params']
            }
        
        return results
    
    def identify_problematic_features(self, df, features, target='occupancy_level'):
        """Identify features that might be causing overfitting."""
        logger.info("Identifying problematic features...")
        
        X = df[features]
        y = df[target]
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Check feature correlations with target
        correlations = {}
        for feature in features:
            if feature in X.columns:
                try:
                    corr = X[feature].corr(y)
                    if not np.isnan(corr):
                        correlations[feature] = abs(corr)
                except:
                    pass
        
        # Sort by correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Check for suspiciously high correlations
        suspicious_features = [(feat, corr) for feat, corr in sorted_correlations if corr > 0.8]
        
        return {
            'feature_correlations': dict(sorted_correlations),
            'suspicious_features': suspicious_features,
            'top_10_features': sorted_correlations[:10]
        }
    
    def run_comprehensive_diagnosis(self):
        """Run complete diagnostic workflow."""
        logger.info("Starting comprehensive performance gap diagnosis...")
        
        # Load data
        df = self.load_data()
        
        # Load config
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except:
            config = {}
        
        # Create features
        df_baseline, baseline_features = self.create_baseline_features(df)
        
        try:
            df_main, main_features = self.create_main_features(df, config)
        except Exception as e:
            logger.error(f"Failed to create main features: {e}")
            df_main, main_features = df_baseline, baseline_features
        
        # Ensure we have the target column
        target_col = 'occupancy_level'
        if target_col not in df_main.columns:
            # Try alternative target names
            for alt_target in ['prediction_code', 'occupancy_class', 'target']:
                if alt_target in df_main.columns:
                    target_col = alt_target
                    break
            else:
                logger.error("No suitable target column found")
                return
        
        # Run diagnostics
        self.results['class_imbalance'] = self.diagnose_class_imbalance(df_main, target_col)
        
        self.results['algorithm_comparison'] = self.compare_algorithms_same_features(
            df_baseline, baseline_features, target_col
        )
        
        self.results['feature_complexity'] = self.compare_feature_complexity(
            df_main, baseline_features, main_features, target_col
        )
        
        self.results['hyperparameter_sensitivity'] = self.test_hyperparameter_sensitivity(
            df_baseline, baseline_features, target_col
        )
        
        self.results['problematic_features'] = self.identify_problematic_features(
            df_main, main_features, target_col
        )
        
        return self.results
    
    def generate_report(self, output_file='performance_gap_report.json'):
        """Generate comprehensive diagnostic report."""
        logger.info(f"Generating report: {output_file}")
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE GAP DIAGNOSTIC REPORT")
        print("="*60)
        
        if 'class_imbalance' in self.results:
            print(f"\n📊 CLASS IMBALANCE:")
            print(f"   Imbalance Ratio: {self.results['class_imbalance']['imbalance_ratio']:.2f}")
            print(f"   Number of Classes: {self.results['class_imbalance']['num_classes']}")
        
        if 'algorithm_comparison' in self.results:
            print(f"\n🔬 ALGORITHM COMPARISON (Same Features):")
            print(f"   RandomForest: {self.results['algorithm_comparison']['rf_mean']:.4f}")
            print(f"   LightGBM: {self.results['algorithm_comparison']['lgb_mean']:.4f}")
            print(f"   Difference: {self.results['algorithm_comparison']['rf_mean'] - self.results['algorithm_comparison']['lgb_mean']:.4f}")
        
        if 'feature_complexity' in self.results:
            print(f"\n🧠 FEATURE COMPLEXITY IMPACT:")
            for feature_set, results in self.results['feature_complexity'].items():
                print(f"   {feature_set.capitalize()}: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f} ({results['num_features']} features)")
        
        if 'problematic_features' in self.results:
            print(f"\n⚠️  SUSPICIOUS FEATURES:")
            suspicious = self.results['problematic_features']['suspicious_features']
            if suspicious:
                for feat, corr in suspicious[:5]:
                    print(f"   {feat}: {corr:.4f}")
            else:
                print("   None detected")
        
        print(f"\n📄 Full report saved to: {output_file}")

def main():
    """Main execution function."""
    # Define data path - adjust as needed
    data_paths = [
        'data/processed/parking_predictions_with_pois_local_filtered.parquet',
        'data/processed/features/features_master_table_historical.parquet',
        'data/processed/parking_predictions_processed.parquet'
    ]
    
    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if not data_path:
        print("❌ No valid data file found. Please check data paths.")
        return
    
    # Run diagnosis
    diagnostic = PerformanceGapDiagnostic(data_path)
    results = diagnostic.run_comprehensive_diagnosis()
    diagnostic.generate_report()

if __name__ == "__main__":
    main() 