"""
Performance Gap Resolution Script

This script implements specific solutions to improve the main model 
performance to match or exceed the baseline model.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import joblib
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceGapResolver:
    """Implements solutions to close the performance gap."""
    
    def __init__(self, data_path, output_dir='models/improved'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self, sample_size=200000):
        """Load and preprocess data with better handling."""
        logger.info(f"Loading data from {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        
        # Sample if too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def create_robust_baseline_features(self, df):
        """Create enhanced baseline features that work well."""
        logger.info("Creating robust baseline features...")
        
        df = df.copy()
        
        # Basic temporal features (known to work well)
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
        
        # Enhanced temporal features
        df['hour_squared'] = df['hour'] ** 2
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
        
        # Week/month patterns
        df['day_of_month'] = df['timestamp'].dt.day
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Interaction features
        df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']
        df['hour_business_interaction'] = df['hour'] * df['is_business_hours']
        
        baseline_features = [
            'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
            'month_sin', 'month_cos', 'is_weekend', 'hour_squared',
            'is_business_hours', 'is_evening', 'is_night',
            'day_of_month', 'week_of_year',
            'hour_weekend_interaction', 'hour_business_interaction'
        ]
        
        return df, baseline_features
    
    def handle_class_imbalance(self, X, y, strategy='hybrid'):
        """Handle class imbalance with various strategies."""
        logger.info(f"Handling class imbalance with {strategy} strategy")
        
        # Check original distribution
        original_dist = pd.Series(y).value_counts().sort_index()
        logger.info(f"Original distribution: {original_dist.to_dict()}")
        
        if strategy == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
        elif strategy == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            
        elif strategy == 'hybrid':
            # Hybrid: SMOTE for minority, undersample for majority
            # First oversample very minority classes
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_temp, y_temp = smote.fit_resample(X, y)
            
            # Then moderately undersample majority classes
            undersampler = RandomUnderSampler(
                sampling_strategy={
                    0: min(50000, pd.Series(y_temp).value_counts()[0]),  # Limit majority class
                    1: min(40000, pd.Series(y_temp).value_counts()[1]),
                    2: min(40000, pd.Series(y_temp).value_counts()[2])
                },
                random_state=42
            )
            X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)
            
        else:
            # No resampling
            X_resampled, y_resampled = X, y
        
        # Check new distribution
        new_dist = pd.Series(y_resampled).value_counts().sort_index()
        logger.info(f"New distribution: {new_dist.to_dict()}")
        
        return X_resampled, y_resampled
    
    def select_best_features(self, X, y, method='rfe', k=20):
        """Feature selection to reduce overfitting."""
        logger.info(f"Selecting best {k} features using {method}")
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            # Use LightGBM for RFE
            estimator = lgb.LGBMClassifier(
                n_estimators=50, 
                random_state=42, 
                verbose=-1
            )
            selector = RFE(estimator=estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.support_].tolist()
            
        else:
            # Return top k features by variance (fallback)
            variances = X.var().sort_values(ascending=False)
            selected_features = variances.head(k).index.tolist()
            X_selected = X[selected_features]
        
        logger.info(f"Selected features: {selected_features}")
        return X_selected, selected_features
    
    def train_improved_lightgbm(self, X, y, config_name='conservative'):
        """Train LightGBM with conservative parameters to reduce overfitting."""
        logger.info(f"Training improved LightGBM with {config_name} config")
        
        configs = {
            'conservative': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'num_leaves': 15,
                'max_depth': 6,
                'min_child_samples': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'class_weight': 'balanced'
            },
            'moderate': {
                'n_estimators': 150,
                'learning_rate': 0.08,
                'num_leaves': 25,
                'max_depth': 8,
                'min_child_samples': 30,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.05,
                'reg_lambda': 0.05,
                'class_weight': 'balanced'
            },
            'aggressive': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                'class_weight': 'balanced'
            }
        }
        
        params = configs.get(config_name, configs['conservative'])
        
        model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            **params
        )
        
        return model
    
    def train_ensemble_model(self, X, y):
        """Train an ensemble of RandomForest and LightGBM."""
        logger.info("Training ensemble model")
        
        # RandomForest (known to work well)
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        
        # Conservative LightGBM
        lgb_model = self.train_improved_lightgbm(X, y, 'conservative')
        
        return {'randomforest': rf, 'lightgbm': lgb_model}
    
    def evaluate_with_time_series_cv(self, model, X, y, model_name='model'):
        """Evaluate model with proper time series cross-validation."""
        logger.info(f"Evaluating {model_name} with time series CV")
        
        tscv = TimeSeriesSplit(n_splits=4, test_size=len(X)//6)
        scores = []
        f1_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if isinstance(model, dict):  # Ensemble
                # Train each model in ensemble
                predictions = []
                for name, mdl in model.items():
                    mdl.fit(X_train, y_train)
                    pred = mdl.predict_proba(X_test)
                    predictions.append(pred)
                
                # Average predictions
                avg_pred = np.mean(predictions, axis=0)
                y_pred = np.argmax(avg_pred, axis=1)
            else:
                # Single model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            
            logger.info(f"Fold {fold + 1}: Accuracy = {scores[-1]:.4f}, F1 = {f1_scores[-1]:.4f}")
        
        return {
            'accuracy_scores': scores,
            'f1_scores': f1_scores,
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores)
        }
    
    def run_comprehensive_improvement(self):
        """Run comprehensive improvement pipeline."""
        logger.info("Starting comprehensive model improvement...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Create robust features
        df_features, feature_names = self.create_robust_baseline_features(df)
        
        # Determine target column
        target_col = 'occupancy_level'
        if target_col not in df_features.columns:
            for alt_target in ['prediction_code', 'occupancy_class', 'target']:
                if alt_target in df_features.columns:
                    target_col = alt_target
                    break
            else:
                logger.error("No suitable target column found")
                return
        
        # Prepare data
        X = df_features[feature_names]
        y = df_features[target_col]
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Data shape after cleaning: {X.shape}")
        
        # Test multiple approaches
        approaches = [
            {
                'name': 'baseline_rf',
                'description': 'RandomForest with baseline features',
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'features': feature_names,
                'resampling': None
            },
            {
                'name': 'improved_lgb_conservative',
                'description': 'LightGBM with conservative parameters',
                'model': self.train_improved_lightgbm(X, y, 'conservative'),
                'features': feature_names,
                'resampling': None
            },
            {
                'name': 'lgb_with_resampling',
                'description': 'LightGBM with class balancing',
                'model': self.train_improved_lightgbm(X, y, 'conservative'),
                'features': feature_names,
                'resampling': 'hybrid'
            },
            {
                'name': 'lgb_feature_selected',
                'description': 'LightGBM with feature selection',
                'model': self.train_improved_lightgbm(X, y, 'moderate'),
                'features': 'select_best',
                'resampling': None
            },
            {
                'name': 'ensemble',
                'description': 'Ensemble of RF and LightGBM',
                'model': 'ensemble',
                'features': feature_names,
                'resampling': None
            }
        ]
        
        # Test each approach
        for approach in approaches:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing: {approach['description']}")
            logger.info(f"{'='*50}")
            
            try:
                # Prepare features
                if approach['features'] == 'select_best':
                    X_selected, selected_features = self.select_best_features(X, y, method='rfe', k=12)
                    approach['selected_features'] = selected_features
                else:
                    X_selected = X[approach['features']]
                    selected_features = approach['features']
                
                # Handle resampling
                if approach['resampling']:
                    X_resampled, y_resampled = self.handle_class_imbalance(
                        X_selected, y, approach['resampling']
                    )
                else:
                    X_resampled, y_resampled = X_selected, y
                
                # Prepare model
                if approach['model'] == 'ensemble':
                    model = self.train_ensemble_model(X_resampled, y_resampled)
                else:
                    model = approach['model']
                
                # Evaluate
                results = self.evaluate_with_time_series_cv(
                    model, X_resampled, y_resampled, approach['name']
                )
                
                # Store results
                self.results[approach['name']] = {
                    'description': approach['description'],
                    'performance': results,
                    'features_used': len(selected_features),
                    'data_shape': X_resampled.shape
                }
                
                # Save best performing models
                if results['mean_accuracy'] > 0.75:  # Threshold for saving
                    model_path = self.output_dir / f"{approach['name']}_model.pkl"
                    joblib.dump(model, model_path)
                    logger.info(f"Saved model to {model_path}")
                
            except Exception as e:
                logger.error(f"Error in approach {approach['name']}: {e}")
                continue
        
        return self.results
    
    def generate_improvement_report(self):
        """Generate comprehensive improvement report."""
        logger.info("Generating improvement report...")
        
        print("\n" + "="*70)
        print("MODEL IMPROVEMENT RESULTS")
        print("="*70)
        
        # Sort results by accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['performance']['mean_accuracy'], 
            reverse=True
        )
        
        print(f"\n{'Rank':<5} {'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Features':<10}")
        print("-" * 70)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            perf = result['performance']
            print(f"{rank:<5} {name:<25} {perf['mean_accuracy']:.4f}±{perf['std_accuracy']:.3f} "
                  f"{perf['mean_f1']:.4f}±{perf['std_f1']:.3f} {result['features_used']:<10}")
        
        # Identify best model
        if sorted_results:
            best_name, best_result = sorted_results[0]
            best_accuracy = best_result['performance']['mean_accuracy']
            
            print(f"\n🏆 BEST MODEL: {best_name}")
            print(f"   Description: {best_result['description']}")
            print(f"   Accuracy: {best_accuracy:.4f} ± {best_result['performance']['std_accuracy']:.4f}")
            print(f"   F1-Score: {best_result['performance']['mean_f1']:.4f}")
            print(f"   Features Used: {best_result['features_used']}")
            
            # Compare with original baseline (87.80%)
            baseline_accuracy = 0.8780
            improvement = best_accuracy - baseline_accuracy
            
            if improvement > 0:
                print(f"   ✅ IMPROVEMENT: +{improvement:.4f} vs baseline")
            else:
                print(f"   ❌ GAP REMAINING: {abs(improvement):.4f} below baseline")
        
        # Save detailed results
        output_file = self.output_dir / 'improvement_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        return sorted_results

def main():
    """Main execution function."""
    # Define data path
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
    
    # Run improvement process
    resolver = PerformanceGapResolver(data_path)
    results = resolver.run_comprehensive_improvement()
    best_models = resolver.generate_improvement_report()
    
    # Provide recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if best_models:
        best_name = best_models[0][0]
        
        print(f"\n1. 🎯 IMMEDIATE ACTION:")
        print(f"   Use the '{best_name}' model for production")
        print(f"   Model file: models/improved/{best_name}_model.pkl")
        
        print(f"\n2. 🔧 FURTHER IMPROVEMENTS:")
        print(f"   - Investigate data quality issues")
        print(f"   - Consider feature engineering refinements")
        print(f"   - Experiment with advanced ensemble methods")
        print(f"   - Analyze temporal data leakage patterns")
        
        print(f"\n3. 📊 MONITORING:")
        print(f"   - Set up performance monitoring in production")
        print(f"   - Track model drift over time")
        print(f"   - Retrain periodically with new data")

if __name__ == "__main__":
    main() 