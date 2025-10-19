import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import optuna
from datetime import datetime, timedelta
import joblib
from collections import Counter
import json
import yaml
from pathlib import Path
import sys

# Add the project root directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.feature_engineering_v2 import FeatureEngineeringPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_main_model")

# --- Diagnostic Functions ---
def diagnose_data_splits(df, df_train=None, df_val=None, df_test=None):
    """Check for data leakage in splits"""
    logger.info("Running data split diagnostics...")
    
    # Verify temporal ordering
    if 'timestamp' in df.columns:
        logger.info(f"Overall data timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        train_min_ts = df_train['timestamp'].min() if df_train is not None and not df_train.empty else None
        train_max_ts = df_train['timestamp'].max() if df_train is not None and not df_train.empty else None
        val_min_ts = df_val['timestamp'].min() if df_val is not None and not df_val.empty else None
        val_max_ts = df_val['timestamp'].max() if df_val is not None and not df_val.empty else None
        test_min_ts = df_test['timestamp'].min() if df_test is not None and not df_test.empty else None
        test_max_ts = df_test['timestamp'].max() if df_test is not None and not df_test.empty else None

        if df_train is not None and not df_train.empty:
            logger.info(f"Training data timestamp range: {train_min_ts} to {train_max_ts}")
        if df_val is not None and not df_val.empty:
            logger.info(f"Validation data timestamp range: {val_min_ts} to {val_max_ts}")
            if train_max_ts and val_min_ts and train_max_ts >= val_min_ts:
                logger.error("❌ TEMPORAL LEAKAGE: Validation period starts before or at the end of training period!")
        if df_test is not None and not df_test.empty:
            logger.info(f"Test data timestamp range: {test_min_ts} to {test_max_ts}")
            if df_val is not None and not df_val.empty and val_max_ts and test_min_ts and val_max_ts >= test_min_ts:
                logger.error("❌ TEMPORAL LEAKAGE: Test period starts before or at the end of validation period!")
            elif df_train is not None and not df_train.empty and train_max_ts and test_min_ts and train_max_ts >= test_min_ts:
                 logger.error("❌ TEMPORAL LEAKAGE: Test period starts before or at the end of training period (no validation set used)!")

    # Check for duplicate rows between train/val/test
    if df_train is not None and df_val is not None and not df_train.empty and not df_val.empty:
        duplicates_train_val = pd.merge(df_train, df_val, how='inner', on=df_train.columns.tolist())
        if not duplicates_train_val.empty:
            logger.error(f"❌ ROW LEAKAGE: {len(duplicates_train_val)} duplicate rows between train and validation sets!")

    if df_train is not None and df_test is not None and not df_train.empty and not df_test.empty:
        duplicates_train_test = pd.merge(df_train, df_test, how='inner', on=df_train.columns.tolist())
        if not duplicates_train_test.empty:
            logger.error(f"❌ ROW LEAKAGE: {len(duplicates_train_test)} duplicate rows between train and test sets!")

    if df_val is not None and df_test is not None and not df_val.empty and not df_test.empty:
        duplicates_val_test = pd.merge(df_val, df_test, how='inner', on=df_val.columns.tolist())
        if not duplicates_val_test.empty:
            logger.error(f"❌ ROW LEAKAGE: {len(duplicates_val_test)} duplicate rows between validation and test sets!")
    logger.info("Data split diagnostics complete.")

def check_feature_leakage(df, target_col='occupancy_level'):
    """Identify features that might leak target information"""
    logger.info("Running feature leakage diagnostics...")
    suspicious_features = []
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame for feature leakage check.")
        return suspicious_features

    for col in df.columns:
        if col == target_col or col == 'timestamp' or col == 'parking_id': # Exclude target and identifiers
            continue
            
        # Check correlation with target for numeric features
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
            try:
                correlation = df[col].corr(df[target_col])
                if abs(correlation) > 0.9:
                    suspicious_features.append((col, correlation))
                    logger.warning(f"⚠️ HIGH CORRELATION with target: {col} (correlation = {correlation:.4f})")
            except Exception as e:
                logger.debug(f"Could not calculate correlation for {col}: {e}")
    
    # Check for perfect predictors (if a feature value always maps to a single target value)
    # This check can be computationally expensive for high cardinality features on large datasets.
    # Consider sampling or skipping for very large data if performance is an issue.
    # For now, this is a simplified check for lower cardinality features.
    if len(df) < 1_000_000: # Only run for smaller (sampled) dataframes to avoid performance issues
        for col in df.columns:
            if col == target_col or col == 'timestamp' or col == 'parking_id':
                continue
            try:
                if df.groupby(col)[target_col].nunique().max() == 1 and df[col].nunique() < len(df) and df[col].nunique() > 1:
                    #nunique < len(df) to avoid columns with all unique values being flagged
                    #nunique > 1 to avoid constant columns being flagged as perfect predictors if they predict the only class present for that value
                    logger.warning(f"⚠️ POTENTIAL PERFECT PREDICTOR: Feature '{col}' might perfectly predict target for its values.")
                    suspicious_features.append((col, "potential perfect predictor"))
            except Exception as e:
                logger.debug(f"Could not perform perfect predictor check for {col}: {e}")
    else:
        logger.info("Skipping perfect predictor check due to large dataset size.")

    logger.info(f"Feature leakage diagnostics complete. Found {len(suspicious_features)} suspicious features.")
    return suspicious_features

# --- VALOR Feature Investigation ---
def investigate_valor_feature(df_input):
    """Comprehensive analysis of VALOR feature"""
    df = df_input.copy() # Work on a copy
    
    # Ensure timestamp is datetime for splitting
    if 'timestamp' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logger.error("Timestamp column missing or not datetime for VALOR investigation. Skipping.")
        return None, None, None
        
    # Temporal split similar to main training setup (using a fixed date from your example)
    # For this specific investigation, we use the date from your snippet.
    # If this function were to be generalized, this split point might need to be dynamic.
    try:
        train_end_date = pd.Timestamp('2023-05-11 09:10:53') 
        train_df = df[df['timestamp'] <= train_end_date]
        test_df = df[df['timestamp'] > train_end_date]
    except Exception as e:
        logger.error(f"Error during temporal split in VALOR investigation: {e}. Splitting by 80/20 rule.")
        # Fallback to 80/20 split if fixed date causes issues (e.g. not in sampled data range)
        df = df.sort_values('timestamp')
        train_end_idx = int(len(df) * 0.8)
        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[train_end_idx:]

    if train_df.empty:
        logger.warning("Training set is empty after temporal split in VALOR investigation. Cannot proceed.")
        return None, None, None
    if 'VALOR' not in train_df.columns or 'occupancy_level' not in train_df.columns:
        logger.error("'VALOR' or 'occupancy_level' not in training DataFrame for VALOR investigation.")
        return None, None, None

    logger.info("=== VALOR Feature Investigation ===")
    
    # Basic statistics
    logger.info(f"\nVALOR in training set (for VALOR investigation split):")
    logger.info(f"Shape: {train_df['VALOR'].shape}")
    logger.info(f"Unique values: {train_df['VALOR'].nunique()}")
    logger.info(f"Data type: {train_df['VALOR'].dtype}")
    min_val_train, max_val_train = train_df['VALOR'].min(), train_df['VALOR'].max()
    logger.info(f"Range: {min_val_train} to {max_val_train}")
    
    # Correlation analysis
    correlation = train_df['VALOR'].corr(train_df['occupancy_level'])
    logger.info(f"\nVALOR correlation with occupancy_level (in VALOR train split): {correlation:.4f}")
    
    # Distribution analysis
    logger.info(f"\nVALOR distribution in training set (VALOR train split):")
    logger.info(f"\n{train_df['VALOR'].describe()}")
    
    # Check for perfect prediction patterns
    valor_by_class = train_df.groupby('occupancy_level')['VALOR'].agg(['mean', 'std', 'min', 'max']).round(4)
    logger.info(f"\nVALOR statistics by occupancy class (VALOR train split):")
    logger.info(f"\n{valor_by_class}")
    
    # Check if VALOR perfectly separates classes
    perfect_separation = True
    # Reduce iterations for performance on potentially large number of unique VALOR values
    # Check only if any value from one class is also present in ANY other class.
    # This is a simplified check for practical purposes.
    unique_occupancy_levels = train_df['occupancy_level'].unique()
    for class_id in unique_occupancy_levels:
        class_valores = set(train_df[train_df['occupancy_level'] == class_id]['VALOR'].unique())
        other_valores = set(train_df[train_df['occupancy_level'] != class_id]['VALOR'].unique())
        
        if not class_valores.isdisjoint(other_valores):
            perfect_separation = False
            logger.info(f"VALOR overlap found for class {class_id}. Not a perfect separator.")
            break
    
    logger.info(f"\nDoes VALOR perfectly separate classes (in VALOR train split)? {perfect_separation}")
    
    # Temporal stability check
    logger.info(f"\nVALOR temporal patterns (VALOR investigation split):")
    logger.info(f"Training VALOR range: {min_val_train} to {max_val_train}")
    if not test_df.empty and 'VALOR' in test_df.columns:
        min_val_test, max_val_test = test_df['VALOR'].min(), test_df['VALOR'].max()
        logger.info(f"Test VALOR range: {min_val_test} to {max_val_test}")
    else:
        logger.info("Test set for VALOR investigation is empty or VALOR missing, skipping test range.")
    logger.info("=== VALOR Feature Investigation Complete ===")
    return correlation, valor_by_class, perfect_separation

# --- Configuration ---
DATA_PATH = 'data/processed/features/features_master_table_historical.parquet'
TARGET_COLUMN = 'actual_state'
TIMESTAMP_COLUMN = 'timestamp'
MODEL_OUTPUT_DIR = 'models/main'
PLOTS_OUTPUT_DIR = 'reports/figures/main'
METRICS_OUTPUT_DIR = 'reports/metrics/main'

# Time series split configuration
N_SPLITS = 4
TEST_SIZE = '3M'  # 3 months test set
GAP = '1M'  # 1 month gap between train and test

# SMOTE configuration
SMOTE_RATIO = 0.5  # Ratio of minority class samples to generate

# Ensure output directories exist
for dir_path in [MODEL_OUTPUT_DIR, PLOTS_OUTPUT_DIR, METRICS_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def load_config(config_path='config/model_config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return {
            'data_path': DATA_PATH,
            'model_path': MODEL_OUTPUT_DIR,
            'feature_engineering': {
                'scale_features': True,
                'parallel_processing': {
                    'enabled': False
                }
            },
            'model': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'class_weight': 'balanced'
            }
        }

def load_data(data_path):
    """Load and preprocess the data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    return df

def preprocess_data(df):
    """Preprocess the data."""
    logger.info("Preprocessing data...")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Rename columns to match expected names
    column_mapping = {
        'ID_TRAMO': 'parking_id',
        'actual_state': 'occupancy_level'
    }
    df = df.rename(columns=column_mapping)
    
    # Create occupancy rate based on the actual_state (0-6 scale)
    # Assuming actual_state represents occupancy levels from 0 (empty) to 6 (full)
    if 'occupancy_rate' not in df.columns:
        df['occupancy_rate'] = df['occupancy_level'] / 6.0
    
    logger.info(f"Dataset shape after preprocessing: {df.shape}")
    logger.info(f"Occupancy level distribution:\n{df['occupancy_level'].value_counts().sort_index()}")
    
    return df

def save_model(model, model_path):
    """Save the trained model."""
    logger.info(f"Saving model to {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def plot_feature_importance(importance_dict, title, filename):
    """Plot feature importance."""
    plt.figure(figsize=(12, 6))
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importances)
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def load_and_sample_data(data_path, sample_size=2_000_000):
    """Load data with intelligent sampling to maintain class balance"""
    import pandas as pd
    import numpy as np

    logger.info(f"Loading data from {data_path}...")

    # Read full data (unavoidable with parquet)
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} records")

    # Get class distribution
    class_counts = df['actual_state'].value_counts()
    logger.info(f"Original class distribution:\n{class_counts}")

    # Intelligent sampling: maintain class balance while reducing total size
    if len(df) > sample_size:
        total_samples = len(df)
        
        # Calculate target samples per class to maintain roughly balanced distribution
        # But cap the maximum samples per class to prevent memory issues
        max_samples_per_class = min(sample_size // len(class_counts), 500_000)
        min_samples_per_class = min(max_samples_per_class // 4, 50_000)  # Ensure minority classes get enough samples

        sampled_dfs = []
        for class_label, count in class_counts.items():
            class_df = df[df['actual_state'] == class_label]

            if count <= min_samples_per_class:
                # Keep all samples for very rare classes
                sampled_dfs.append(class_df)
                logger.info(f"Class {class_label}: keeping all {count:,} samples")
            else:
                # Sample proportionally but ensure minimum representation
                target_samples = max(min_samples_per_class, 
                                   min(max_samples_per_class, int(sample_size * count / total_samples)))
                sample_ratio = target_samples / count
                sampled_class = class_df.sample(n=target_samples, random_state=42)
                sampled_dfs.append(sampled_class)
                logger.info(f"Class {class_label}: sampled {target_samples:,} from {count:,} samples")

        df = pd.concat(sampled_dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

        logger.info(f"Sampled to {len(df):,} records")
        logger.info(f"Final class distribution:\n{df['actual_state'].value_counts()}")

    return df

def select_features(df):
    """Select features for the main model."""
    logger.info("Selecting features...")
    
    # Temporal features
    temporal_features = [
        'hour', 'dayofweek', 'dayofyear', 'month', 'year',
        'weekofyear', 'quarter', 'is_weekend',
        'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos',
        'is_holiday', 'is_event_day'
    ]
    
    # Lag features
    lag_features = [
        'actual_state_lag_1h',
        'actual_state_lag_6h',
        'actual_state_lag_12h',
        'actual_state_lag_24h',
        'actual_state_lag_48h',
        'actual_state_lag_168h'
    ]
    
    # Additional features from the dataset
    additional_features = [
        'VALOR'  # This might be related to occupancy measurements
    ]
    
    # Combine all features
    all_features = temporal_features + lag_features + additional_features
    
    # Check which features exist in the DataFrame
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    logger.info(f"Selected {len(available_features)} features for modeling")
    logger.info(f"Available features: {available_features}")
    
    return available_features

def create_time_series_split(df):
    """Create time series split for cross-validation."""
    logger.info(f"Creating TimeSeriesSplit with {N_SPLITS} splits...")
    
    # Calculate split points
    total_days = (df[TIMESTAMP_COLUMN].max() - df[TIMESTAMP_COLUMN].min()).days
    split_size = total_days // (N_SPLITS + 1)  # +1 for the initial training set
    
    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=split_size,
        gap=split_size // 4  # Gap between train and test
    )
    
    return tscv

def apply_smote(X, y):
    """Apply SMOTE with memory-efficient sampling strategy for multi-class"""
    # Get class counts
    class_counts = dict(pd.Series(y).value_counts())
    majority_class_count = max(class_counts.values())
    total_samples = sum(class_counts.values())
    
    logger.info(f"Class distribution before SMOTE: {class_counts}")
    
    # Very conservative approach: Only upsample extremely rare classes
    # Target 5% of majority class for classes with <0.5% of total samples
    target_count = int(majority_class_count * 0.05)  # ~1.5M samples per class
    
    sampling_strategy = {
        cls: target_count 
        for cls, count in class_counts.items() 
        if count < (total_samples * 0.005)  # Only classes with <0.5% of total samples
    }
    
    logger.info(f"SMOTE sampling strategy: {sampling_strategy}")
    
    if not sampling_strategy:
        logger.info("No resampling needed - all classes already balanced")
        return X, y
    
    # Convert to memory-efficient types before SMOTE
    X = X.astype('float32')
    
    # Process SMOTE in chunks for memory efficiency
    try:
        # Initialize SMOTE with minimal memory usage
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            n_jobs=1,  # Use single thread to reduce memory
            k_neighbors=3  # Minimal number of neighbors
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Verify results
        final_counts = dict(pd.Series(y_resampled).value_counts())
        logger.info(f"Class distribution after SMOTE: {final_counts}")
        
        return X_resampled, y_resampled
    except MemoryError:
        logger.warning("Memory error during SMOTE. Falling back to class weights only.")
        return X, y

def calculate_class_weights(y):
    """Calculate class weights based on class distribution."""
    class_counts = Counter(y)
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    return class_weights

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for hyperparameter optimization."""
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_class': len(np.unique(y_train)),
        
        # Parameters to optimize
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
    # Always use class weights
    class_weights = calculate_class_weights(y_train)
    param['class_weight'] = class_weights
    
    model = lgb.LGBMClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )
    
    # Use weighted F1 score as the optimization metric
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    return f1

def train_and_evaluate_model(df, features, tscv):
    """Train and evaluate the model using time series cross-validation."""
    logger.info("Starting model training and evaluation...")
    
    X = df[features]
    y = df[TARGET_COLUMN]
    
    # Store results
    all_preds = []
    all_true = []
    fold_metrics = []
    feature_importance = []
    
    # Train and evaluate for each fold
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        logger.info(f"Processing fold {fold + 1}/{N_SPLITS}")
        
        # Process in chunks to save memory
        chunk_size = 1000000  # 1M rows per chunk
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Create validation set from the last 20% of training data
        val_size = int(len(X_train) * 0.2)
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        
        # Apply SMOTE to training data
        X_train_resampled, y_train_resampled = apply_smote(X_train_final, y_train_final)
        
        # Optimize hyperparameters with reduced number of trials
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, X_train_resampled, y_train_resampled, X_val, y_val),
            n_trials=20  # Reduced from 50
        )
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_class': len(np.unique(y_train)),
            'n_jobs': 1  # Use single thread to reduce memory
        })
        
        # Add class weights to best parameters
        class_weights = calculate_class_weights(y_train_resampled)
        best_params['class_weight'] = class_weights
        
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )
        
        # Evaluate in chunks
        y_pred = []
        for i in range(0, len(X_test), chunk_size):
            chunk_pred = model.predict(X_test.iloc[i:i+chunk_size])
            y_pred.extend(chunk_pred)
        
        # Calculate comprehensive metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'class_distribution': Counter(y_test)
        })
        
        # Store predictions and true values
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        
        # Store feature importance
        feature_importance.append(model.feature_importances_)
        
        # Save model
        model_path = os.path.join(MODEL_OUTPUT_DIR, f'model_fold_{fold + 1}.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Saved model for fold {fold + 1} to {model_path}")
        
        # Free memory
        del X_train, y_train, X_test, y_test
        del X_train_final, y_train_final, X_val, y_val
        del X_train_resampled, y_train_resampled
        del model
    
    return all_preds, all_true, fold_metrics, feature_importance

def generate_evaluation_plots(all_preds, all_true, fold_metrics, feature_importance):
    """Generate evaluation plots and save metrics."""
    logger.info("Generating evaluation plots and metrics...")
    
    # Convert predictions and true values to numpy arrays
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Calculate overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_preds, average=None
    )
    
    # Create metrics summary with converted numpy arrays to lists
    metrics_summary = {
        'overall': {
            'accuracy': float(accuracy_score(all_true, all_preds)),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'per_fold': [
            {
                'fold': m['fold'],
                'accuracy': float(m['accuracy']),
                'precision': m['precision'].tolist() if isinstance(m['precision'], np.ndarray) else m['precision'],
                'recall': m['recall'].tolist() if isinstance(m['recall'], np.ndarray) else m['recall'],
                'f1': m['f1'].tolist() if isinstance(m['f1'], np.ndarray) else m['f1'],
                'support': m['support'].tolist() if isinstance(m['support'], np.ndarray) else m['support'],
                'class_distribution': dict(m['class_distribution'])
            }
            for m in fold_metrics
        ]
    }
    
    # Save metrics
    metrics_path = os.path.join(METRICS_OUTPUT_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    class_dist = Counter(all_true)
    plt.bar(class_dist.keys(), class_dist.values())
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, 'class_distribution.png'))
    plt.close()
    
    # Plot per-class metrics
    classes = np.unique(all_true)
    metrics = ['precision', 'recall', 'f1']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, metrics_summary['overall'][metric], width, label=metric)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Metrics')
    plt.xticks(x + width, classes)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, 'per_class_metrics.png'))
    plt.close()
    
    # Plot feature importance
    feature_importance_df = pd.DataFrame(feature_importance)
    mean_importance = feature_importance_df.mean(axis=0).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    mean_importance.plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, 'feature_importance.png'))
    plt.close()
    
    # Plot learning curves for each fold
    plt.figure(figsize=(12, 6))
    for fold in range(len(fold_metrics)):
        metrics = fold_metrics[fold]
        plt.plot(metrics['f1'], label=f'Fold {fold + 1}')
    
    plt.title('F1 Score per Fold')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, 'learning_curves.png'))
    plt.close()

def create_coarse_labels(occupancy_level):
    """Convert 7-class occupancy levels into 3 coarse classes."""
    if occupancy_level in [0, 1]:
        return 'Low'        # Classes 0,1 → Low occupancy
    elif occupancy_level in [2, 3]:
        return 'Medium'     # Classes 2,3 → Medium occupancy  
    else:
        return 'High'       # Classes 4,5,6 → High occupancy

def create_fine_labels(occupancy_level, coarse_class):
    """Create fine-grained labels within each coarse class."""
    if coarse_class == 'Low':
        return occupancy_level  # Keep original labels 0,1
    elif coarse_class == 'Medium':
        return occupancy_level - 2  # Map 2,3 to 0,1
    else:  # High
        return occupancy_level - 4  # Map 4,5,6 to 0,1,2

class HierarchicalClassifier:
    """Two-stage hierarchical classifier for parking occupancy prediction."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.coarse_model = None
        self.fine_models = {}
        self.feature_importance = {}
        
    def fit(self, X, y):
        """Train the hierarchical classifier."""
        # Create coarse labels
        y_coarse = pd.Series(y).apply(create_coarse_labels)
        
        # Train coarse classifier with reduced complexity
        self.coarse_model = lgb.LGBMClassifier(
            n_estimators=200,  # Reduced from 1000
            learning_rate=0.1,
            num_leaves=15,     # Reduced from 31
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        self.coarse_model.fit(X, y_coarse)
        
        # Train fine classifiers for each coarse class
        for coarse_class in ['Low', 'Medium', 'High']:
            mask = (y_coarse == coarse_class)
            if mask.sum() > 50:  # Reduced threshold from 100
                X_fine = X[mask]
                y_fine = pd.Series(y[mask]).apply(
                    lambda x: create_fine_labels(x, coarse_class)
                )
                
                self.fine_models[coarse_class] = lgb.LGBMClassifier(
                    n_estimators=100,  # Reduced from 500
                    learning_rate=0.05,
                    num_leaves=10,     # Reduced from 31
                    class_weight='balanced',
                    random_state=42,
                    verbose=-1
                )
                self.fine_models[coarse_class].fit(X_fine, y_fine)
        
        # Store feature importance
        self.feature_importance['coarse'] = dict(zip(
            X.columns,
            self.coarse_model.feature_importances_
        ))
        for coarse_class, model in self.fine_models.items():
            self.feature_importance[coarse_class] = dict(zip(
                X.columns,
                model.feature_importances_
            ))
    
    def predict(self, X):
        """Make predictions using the hierarchical classifier."""
        # Get coarse predictions
        coarse_preds = self.coarse_model.predict(X)
        
        # Initialize fine predictions
        fine_preds = np.zeros(len(X), dtype=int)
        
        # Make fine predictions for each coarse class
        for coarse_class in ['Low', 'Medium', 'High']:
            mask = (coarse_preds == coarse_class)
            if mask.sum() > 0 and coarse_class in self.fine_models:
                fine_preds[mask] = self.fine_models[coarse_class].predict(X[mask])
        
        # Convert back to original 7-class scale
        final_preds = np.zeros(len(X), dtype=int)
        for i, (coarse, fine) in enumerate(zip(coarse_preds, fine_preds)):
            if coarse == 'Low':
                final_preds[i] = fine
            elif coarse == 'Medium':
                final_preds[i] = fine + 2
            else:  # High
                final_preds[i] = fine + 4
        
        return final_preds
    
    def predict_proba(self, X):
        """Get probability estimates for each class."""
        # Get coarse probabilities
        coarse_probs = self.coarse_model.predict_proba(X)
        
        # Initialize fine probabilities
        fine_probs = np.zeros((len(X), 7))
        
        # Get fine probabilities for each coarse class
        for i, coarse_class in enumerate(['Low', 'Medium', 'High']):
            mask = (coarse_preds == coarse_class)
            if mask.sum() > 0 and coarse_class in self.fine_models:
                if coarse_class == 'Low':
                    fine_probs[mask, :2] = self.fine_models[coarse_class].predict_proba(X[mask])
                elif coarse_class == 'Medium':
                    fine_probs[mask, 2:4] = self.fine_models[coarse_class].predict_proba(X[mask])
                else:  # High
                    fine_probs[mask, 4:] = self.fine_models[coarse_class].predict_proba(X[mask])
        
        # Combine probabilities
        final_probs = np.zeros((len(X), 7))
        for i in range(len(X)):
            for j in range(7):
                if j < 2:  # Low
                    final_probs[i, j] = coarse_probs[i, 0] * fine_probs[i, j]
                elif j < 4:  # Medium
                    final_probs[i, j] = coarse_probs[i, 1] * fine_probs[i, j]
                else:  # High
                    final_probs[i, j] = coarse_probs[i, 2] * fine_probs[i, j]
        
        return final_probs

def train_model(X, y, config=None):
    """Train the hierarchical classifier model."""
    # Initialize and train the hierarchical classifier
    model = HierarchicalClassifier(config)
    model.fit(X, y)
    
    return model, model.feature_importance

def evaluate_model(model, X_test, y_test):
    """Evaluate the hierarchical classifier."""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate per-class metrics
    class_report = classification_report(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

def main():
    """Main function to run the model training pipeline."""
    # Load configuration
    config = load_config()
    
    # Load and preprocess data with smaller sample for faster training
    logger.info("Loading and sampling data for efficient training...")
    df_full = load_and_sample_data(config['data_path'], sample_size=500_000)  # Reduced to 500K records
    df = preprocess_data(df_full.copy()) # Work on a copy for diagnostics

    # Apply robust feature engineering using FeatureEngineeringPipeline
    logger.info("Applying FeatureEngineeringPipeline for robust feature creation...")
    # Ensure the main config has a section for feature_engineering if pipeline expects it
    # The pipeline has defaults, but good to align with main config structure
    fe_config = config.get('feature_engineering', {})
    # If scale_features is a top-level config affecting the pipeline, pass it in a way the pipeline understands
    # For now, FeatureEngineeringPipeline uses its internal config for scaling if 'scale_features': True is in its own config section.
    # We might need to harmonize config structure later if FeatureEngineeringPipeline expects specific keys from the main model_config.yaml
    feature_pipeline = FeatureEngineeringPipeline(config) # Pass the whole config, pipeline will pick what it needs
    df = feature_pipeline.fit_transform(df)
    logger.info("FeatureEngineeringPipeline applied successfully.")
    
    # --- Run Diagnostics (Early) ---
    investigate_valor_feature(df) # Investigate VALOR before other leakage checks or feature selection
    check_feature_leakage(df, target_col='occupancy_level')
    # Note: diagnose_data_splits will be called after splitting
    # --- End Diagnostics ---

    # Select features
    features = select_features(df)
    
    # Prepare data for training
    X = df[features]
    y = df['occupancy_level']
    
    # Convert to memory-efficient data types
    logger.info("Converting to memory-efficient data types...")
    for col in X.columns:
        if X[col].dtype == 'float64':
            X.loc[:, col] = X[col].astype('float32')
        elif X[col].dtype == 'int64':
            X.loc[:, col] = X[col].astype('int16')
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Split data into train and test sets
    # --- TEMPORAL SPLIT --- 
    logger.info("Performing temporal train/test split...")
    df_sorted = df.sort_values('timestamp')
    train_end_idx = int(len(df_sorted) * 0.8) # 80% for training
    
    df_train = df_sorted.iloc[:train_end_idx]
    df_test = df_sorted.iloc[train_end_idx:]

    X_train = df_train[features]
    y_train = df_train['occupancy_level']
    X_test = df_test[features]
    y_test = df_test['occupancy_level']

    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    diagnose_data_splits(df_full, df_train, df_test=df_test)
    # --- END TEMPORAL SPLIT ---

    # Train the hierarchical model
    logger.info("Training hierarchical classifier...")
    model, feature_importance = train_model(X_train, y_train, config['model'])
    
    # Evaluate the model
    logger.info("Evaluating model...")
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])
    
    # Create output directories if they don't exist
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('reports/metrics', exist_ok=True)
    
    # Plot confusion matrix if enabled
    if config['evaluation']['plot_confusion_matrix']:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            evaluation_results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(7),
            yticklabels=range(7)
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('reports/figures/confusion_matrix.png')
        plt.close()
    
    # Plot feature importance if enabled
    if config['evaluation']['plot_feature_importance']:
        # Plot feature importance for coarse classifier
        plot_feature_importance(
            feature_importance['coarse'],
            title='Feature Importance (Coarse Classifier)',
            filename='reports/figures/feature_importance_coarse.png'
        )
        
        # Plot feature importance for fine classifiers
        for coarse_class, importance in feature_importance.items():
            if coarse_class != 'coarse':
                plot_feature_importance(
                    importance,
                    title=f'Feature Importance ({coarse_class} Fine Classifier)',
                    filename=f'reports/figures/feature_importance_{coarse_class.lower()}.png'
                )
    
    # Save evaluation metrics
    metrics = {
        'accuracy': float(evaluation_results['accuracy']),
        'f1_score': float(evaluation_results['f1_score']),
        'classification_report': evaluation_results['classification_report']
    }
    
    with open('reports/metrics/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save the model
    save_model(model, config['model_path'])
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {config['model_path']}")
    print(f"Evaluation metrics saved to: reports/metrics/evaluation_metrics.json")
    print(f"Plots saved to: reports/figures/")

if __name__ == "__main__":
    main() 