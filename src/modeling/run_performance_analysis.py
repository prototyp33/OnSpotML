"""
Working Performance Gap Analysis

This script analyzes the performance gap using the actual parking data structure.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_analyze_data():
    """Load and analyze the parking data structure."""
    logger.info("Loading parking data...")
    
    df = pd.read_parquet('data/processed/parking_predictions_phase1_enriched.parquet')
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Target distribution: {df['prediction_code'].value_counts().sort_index().to_dict()}")
    
    # Sample for faster analysis
    if len(df) > 200000:
        df = df.sample(n=200000, random_state=42)
        logger.info(f"Sampled to {len(df)} rows")
    
    return df

def create_enhanced_features(df):
    """Create enhanced features for the parking data."""
    logger.info("Creating enhanced temporal features...")
    
    df = df.copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    # Time-based features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    
    # Time interactions
    df['hour_weekend'] = df['hour'] * df['is_weekend']
    df['hour_business'] = df['hour'] * df['is_business_hours']
    
    # Parking-specific features
    # Encode categorical variables
    le_tipo = LabelEncoder()
    le_tarifa = LabelEncoder()
    
    df['tipo_encoded'] = le_tipo.fit_transform(df['TIPO'].fillna('unknown'))
    df['tarifa_encoded'] = le_tarifa.fit_transform(df['TARIFA'].fillna('unknown'))
    
    # Parking location features
    df['parking_id'] = df['ID_TRAMO']
    
    # Create lag features (occupancy patterns)
    df = df.sort_values(['parking_id', 'timestamp'])
    for lag in [1, 2, 3, 6, 12]:  # 5, 10, 15, 30, 60 minutes ago
        df[f'prediction_lag_{lag}'] = df.groupby('parking_id')['prediction_code'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:  # 15, 30, 60 minute windows
        df[f'prediction_rolling_mean_{window}'] = df.groupby('parking_id')['prediction_code'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'prediction_rolling_std_{window}'] = df.groupby('parking_id')['prediction_code'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Define feature sets
    baseline_features = [
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos', 'is_weekend'
    ]
    
    enhanced_features = baseline_features + [
        'is_business_hours', 'is_evening', 'is_night', 'is_morning_rush', 'is_evening_rush',
        'hour_weekend', 'hour_business', 'tipo_encoded', 'tarifa_encoded'
    ]
    
    complex_features = enhanced_features + [
        'prediction_lag_1', 'prediction_lag_2', 'prediction_lag_3', 'prediction_lag_6',
        'prediction_rolling_mean_3', 'prediction_rolling_mean_6', 'prediction_rolling_std_3'
    ]
    
    return df, baseline_features, enhanced_features, complex_features

def evaluate_model_approach(df, features, target='prediction_code', model_type='rf', model_name=''):
    """Evaluate a specific model approach."""
    logger.info(f"Evaluating {model_name} with {len(features)} features")
    
    # Prepare data
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    X = df[available_features].fillna(0)  # Fill NaN values
    y = df[target]
    
    # Remove any remaining NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        logger.error("No valid data after cleaning")
        return None
    
    logger.info(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3, test_size=len(X)//5)
    scores = []
    f1_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Choose model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'lgb_conservative':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=15,
                max_depth=6,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
        elif model_type == 'lgb_default':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        scores.append(accuracy)
        f1_scores.append(f1)
        
        logger.info(f"Fold {fold + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
    
    return {
        'accuracy_scores': scores,
        'f1_scores': f1_scores,
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'features_used': len(available_features)
    }

def run_comprehensive_analysis():
    """Run comprehensive performance gap analysis."""
    logger.info("Starting comprehensive performance analysis...")
    
    # Load data
    df = load_and_analyze_data()
    
    # Create features
    df_features, baseline_features, enhanced_features, complex_features = create_enhanced_features(df)
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'baseline_rf',
            'description': 'RandomForest with basic temporal features',
            'model_type': 'rf',
            'features': baseline_features
        },
        {
            'name': 'enhanced_rf',
            'description': 'RandomForest with enhanced features',
            'model_type': 'rf',
            'features': enhanced_features
        },
        {
            'name': 'baseline_lgb_default',
            'description': 'LightGBM default with basic features',
            'model_type': 'lgb_default',
            'features': baseline_features
        },
        {
            'name': 'baseline_lgb_conservative',
            'description': 'LightGBM conservative with basic features',
            'model_type': 'lgb_conservative',
            'features': baseline_features
        },
        {
            'name': 'enhanced_lgb_conservative',
            'description': 'LightGBM conservative with enhanced features',
            'model_type': 'lgb_conservative',
            'features': enhanced_features
        },
        {
            'name': 'complex_lgb_conservative',
            'description': 'LightGBM conservative with complex features',
            'model_type': 'lgb_conservative',
            'features': complex_features
        }
    ]
    
    # Run scenarios
    results = {}
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {scenario['description']}")
        logger.info(f"{'='*60}")
        
        try:
            result = evaluate_model_approach(
                df_features,
                scenario['features'],
                model_type=scenario['model_type'],
                model_name=scenario['name']
            )
            
            if result:
                results[scenario['name']] = {
                    'description': scenario['description'],
                    'performance': result
                }
                
        except Exception as e:
            logger.error(f"Error in scenario {scenario['name']}: {e}")
            continue
    
    return results

def generate_analysis_report(results):
    """Generate comprehensive analysis report."""
    logger.info("Generating analysis report...")
    
    print("\n" + "="*80)
    print("PERFORMANCE GAP ANALYSIS RESULTS")
    print("="*80)
    
    if not results:
        print("❌ No successful results to report")
        return
    
    # Sort by accuracy
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['performance']['mean_accuracy'],
        reverse=True
    )
    
    print(f"\n{'Rank':<5} {'Model':<30} {'Accuracy':<15} {'F1-Score':<15} {'Features':<10}")
    print("-" * 80)
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        perf = result['performance']
        print(f"{rank:<5} {name:<30} {perf['mean_accuracy']:.4f}±{perf['std_accuracy']:.3f} "
              f"{perf['mean_f1']:.4f}±{perf['std_f1']:.3f} {perf['features_used']:<10}")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Find best RF vs best LGB
    rf_results = [(name, result) for name, result in sorted_results if 'rf' in name]
    lgb_results = [(name, result) for name, result in sorted_results if 'lgb' in name]
    
    if rf_results and lgb_results:
        best_rf = rf_results[0]
        best_lgb = lgb_results[0]
        
        print(f"\n🔍 ALGORITHM COMPARISON:")
        print(f"   Best RandomForest: {best_rf[1]['performance']['mean_accuracy']:.4f} ({best_rf[0]})")
        print(f"   Best LightGBM: {best_lgb[1]['performance']['mean_accuracy']:.4f} ({best_lgb[0]})")
        
        if best_rf[1]['performance']['mean_accuracy'] > best_lgb[1]['performance']['mean_accuracy']:
            diff = best_rf[1]['performance']['mean_accuracy'] - best_lgb[1]['performance']['mean_accuracy']
            print(f"   ✅ RandomForest outperforms LightGBM by {diff:.4f}")
        else:
            diff = best_lgb[1]['performance']['mean_accuracy'] - best_rf[1]['performance']['mean_accuracy']
            print(f"   ✅ LightGBM outperforms RandomForest by {diff:.4f}")
    
    # Feature complexity analysis
    baseline_results = [(name, result) for name, result in sorted_results if 'baseline' in name]
    enhanced_results = [(name, result) for name, result in sorted_results if 'enhanced' in name]
    complex_results = [(name, result) for name, result in sorted_results if 'complex' in name]
    
    print(f"\n🧠 FEATURE COMPLEXITY IMPACT:")
    if baseline_results:
        avg_baseline = np.mean([r[1]['performance']['mean_accuracy'] for r in baseline_results])
        print(f"   Baseline features: {avg_baseline:.4f} average")
    if enhanced_results:
        avg_enhanced = np.mean([r[1]['performance']['mean_accuracy'] for r in enhanced_results])
        print(f"   Enhanced features: {avg_enhanced:.4f} average")
    if complex_results:
        avg_complex = np.mean([r[1]['performance']['mean_accuracy'] for r in complex_results])
        print(f"   Complex features: {avg_complex:.4f} average")
    
    # Best overall
    if sorted_results:
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 RECOMMENDED MODEL: {best_name}")
        print(f"   Description: {best_result['description']}")
        print(f"   Accuracy: {best_result['performance']['mean_accuracy']:.4f} ± {best_result['performance']['std_accuracy']:.4f}")
        print(f"   F1-Score: {best_result['performance']['mean_f1']:.4f}")
    
    # Save results
    output_file = 'performance_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {output_file}")

def main():
    """Main execution function."""
    try:
        results = run_comprehensive_analysis()
        generate_analysis_report(results)
        
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print("\nThis analysis helps identify:")
        print("1. Whether RandomForest or LightGBM works better for your data")
        print("2. Impact of feature complexity on performance")
        print("3. Optimal hyperparameter settings")
        print("4. Best practices for your specific use case")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 