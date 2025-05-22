import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import optuna
from datetime import datetime, timedelta
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_main_model")

# --- Configuration ---
DATA_PATH = 'data/processed/features_master_table_historical_FULL.parquet'
TARGET_COLUMN = 'actual_state'
TIMESTAMP_COLUMN = 'timestamp'
MODEL_OUTPUT_DIR = 'models/main'
PLOTS_OUTPUT_DIR = 'reports/figures/main'
METRICS_OUTPUT_DIR = 'reports/metrics/main'

# Time series split configuration
N_SPLITS = 4
TEST_SIZE = '3M'  # 3 months test set
GAP = '1M'  # 1 month gap between train and test

# Ensure output directories exist
for dir_path in [MODEL_OUTPUT_DIR, PLOTS_OUTPUT_DIR, METRICS_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the data for modeling."""
    logger.info(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
        logger.info(f"Loaded {len(df)} records.")
        
        # Convert timestamp to datetime
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
        
        # Sort by timestamp
        df = df.sort_values(TIMESTAMP_COLUMN)
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def select_features(df):
    """Select features for the main model."""
    logger.info("Selecting features...")
    
    # Temporal features
    temporal_features = [
        'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos',
        'is_weekend',
        'is_public_holiday',
        'is_school_holiday'
    ]
    
    # POI features
    poi_features = []
    for category in ['sustenance', 'shop', 'education', 'health', 'transport', 
                    'leisure', 'tourism', 'parking', 'finance']:
        for radius in [100, 200, 500]:
            # Use log-transformed counts and presence flags
            poi_features.extend([
                f'poi_{category}_log1p_count_{radius}m',
                f'poi_{category}_present_{radius}m'
            ])
    
    # Combine all features
    all_features = temporal_features + poi_features
    
    # Check which features exist in the DataFrame
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    logger.info(f"Selected {len(available_features)} features for modeling")
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
    
    # Add class weight if needed
    if trial.suggest_categorical('use_class_weight', [True, False]):
        param['class_weight'] = 'balanced'
    
    model = lgb.LGBMClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Use F1 score as the optimization metric
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
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create validation set from the last 20% of training data
        val_size = int(len(X_train) * 0.2)
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, X_train_final, y_train_final, X_val, y_val),
            n_trials=50
        )
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_class': len(np.unique(y_train))
        })
        
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        })
        
        # Store predictions and true values
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        
        # Store feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_,
            'fold': fold + 1
        })
        feature_importance.append(importance)
        
        # Save model
        model_path = os.path.join(MODEL_OUTPUT_DIR, f'model_fold_{fold + 1}.txt')
        model.booster_.save_model(model_path)
        
        # Save best parameters
        params_path = os.path.join(MODEL_OUTPUT_DIR, f'best_params_fold_{fold + 1}.json')
        joblib.dump(best_params, params_path)
    
    return all_preds, all_true, fold_metrics, feature_importance

def generate_evaluation_plots(all_preds, all_true, fold_metrics, feature_importance):
    """Generate evaluation plots and save metrics."""
    logger.info("Generating evaluation plots and metrics...")
    
    # Overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(all_true, all_preds),
        'f1_weighted': f1_score(all_true, all_preds, average='weighted'),
        'f1_macro': f1_score(all_true, all_preds, average='macro')
    }
    
    # Save overall metrics
    metrics_df = pd.DataFrame([overall_metrics])
    metrics_df.to_csv(os.path.join(METRICS_OUTPUT_DIR, 'overall_metrics.csv'))
    
    # Save fold metrics
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(os.path.join(METRICS_OUTPUT_DIR, 'fold_metrics.csv'))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Plot feature importance
    feature_importance_df = pd.concat(feature_importance)
    mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    mean_importance.head(20).plot(kind='bar')
    plt.title('Top 20 Feature Importance (Mean across folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, 'feature_importance.png'))
    plt.close()
    
    # Save detailed classification report
    report = classification_report(all_true, all_preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(METRICS_OUTPUT_DIR, 'classification_report.csv'))

def main():
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Select features
        features = select_features(df)
        
        # Create time series split
        tscv = create_time_series_split(df)
        
        # Train and evaluate model
        all_preds, all_true, fold_metrics, feature_importance = train_and_evaluate_model(
            df, features, tscv
        )
        
        # Generate evaluation plots and metrics
        generate_evaluation_plots(all_preds, all_true, fold_metrics, feature_importance)
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 