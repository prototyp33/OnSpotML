import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Disable PyArrow string backend as a temporary diagnostic step
pd.options.mode.string_storage = "python"

# Setup logging
# NOTE: Changed logger name and output paths for clarity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_temporal_baseline") # Changed logger name

# --- Configuration ---
DATA_PATH = 'data/processed/parking_predictions_with_pois_local_filtered.parquet' # Still needs POI file to load
TARGET_COLUMN = 'prediction_code' 
TIMESTAMP_COLUMN = 'timestamp'
MODEL_OUTPUT_DIR = 'models/temporal_baseline' # Changed output dir
PLOTS_OUTPUT_DIR = 'reports/figures/temporal_baseline' # Changed output dir
N_SPLITS = 3 

# Ensure output directories exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
logger.info(f"Loading data from {DATA_PATH}...")
try:
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df)} records.")
    
    # Drop geometry column if it exists
    if 'geometry_parsed' in df.columns:
        df = df.drop(columns=['geometry_parsed'])
        logger.info("Dropped 'geometry_parsed' column.")
    elif 'geometry' in df.columns: 
        df = df.drop(columns=['geometry'])
        logger.info("Dropped 'geometry' column.")
except FileNotFoundError:
    logger.error(f"Data file not found at {DATA_PATH}. Please ensure the previous steps ran successfully.")
    exit()
except Exception as e:
    logger.error(f"Error loading data: {e}")
    exit()

# Convert timestamp column to datetime
logger.info(f"Converting {TIMESTAMP_COLUMN} to datetime...")
if TIMESTAMP_COLUMN not in df.columns:
    logger.error(f"Timestamp column '{TIMESTAMP_COLUMN}' not found.")
    exit()
try:
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN]) 
except Exception as e:
    logger.error(f"Error converting timestamp column: {e}")
    exit()

# Check if target column exists
if TARGET_COLUMN not in df.columns:
    logger.error(f"Target column '{TARGET_COLUMN}' not found.")
    exit()

# Determine number of classes
num_classes = df[TARGET_COLUMN].nunique()
logger.info(f"Target column '{TARGET_COLUMN}' found with {num_classes} unique classes: {sorted(df[TARGET_COLUMN].unique())}")
if num_classes <= 1:
    logger.error("Target column has only one class. Cannot train classifier.")
    exit()

# --- Feature Engineering (Temporal Features Only) ---
logger.info("Generating temporal features...")
try:    
    df['hour'] = df[TIMESTAMP_COLUMN].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['dayofweek'] = df[TIMESTAMP_COLUMN].dt.dayofweek 
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    df['month'] = df[TIMESTAMP_COLUMN].dt.month
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12) 
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    logger.info("Temporal features generated.")
except Exception as e:
    logger.error(f"Error generating temporal features: {e}")
    exit()

# --- POI Feature Transformation (Log and Binary Presence) ---
# NOTE: We still run this section to ensure the columns exist if needed later,
#       but we WON'T select them as features for this baseline model.
logger.info("(Skipping POI transformations for temporal baseline - features will be excluded later)")
# original_poi_count_features = [col for col in df.columns if col.startswith('poi_') and 'count' in col]
# if not original_poi_count_features:
#     logger.warning("No original POI count features found to transform.")
# else:
#     logger.info(f"Found {len(original_poi_count_features)} POI count features.")
    # for col in original_poi_count_features:
    #     presence_col_name = f"{col}_present"
    #     df[presence_col_name] = (df[col] > 0).astype(int)
    #     df[col] = np.log1p(df[col])
    # logger.info("POI feature transformations complete.")

# --- Feature Selection ---
logger.info("Selecting features (TEMPORAL ONLY baseline)...")

# Select ONLY temporal features
temporal_features = [
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'is_weekend',
]

# POI features are explicitly EXCLUDED for this baseline
poi_features = [] 
base_features = temporal_features # Use only temporal

logger.info(f"Total potential baseline features to consider: {len(base_features)}")

# Check which features actually exist in the DataFrame
available_features = [f for f in base_features if f in df.columns]
missing_features = [f for f in base_features if f not in df.columns]

if missing_features:
    logger.warning(f"The following potential baseline features were not found and will be excluded:")
    for f in missing_features:
        logger.warning(f" - {f}")

logger.info(f"Using {len(available_features)} available features for TEMPORAL baseline model:")
for f in available_features:
    logger.info(f" - {f}")
features = available_features

if not features:
    logger.error("No features available for training after selection. Exiting.")
    exit()
    
# --- Handle NaNs (if any) ---
initial_rows = len(df)
original_columns = df.columns.tolist()

# Check for NaNs in features and target
columns_to_check = features + [TARGET_COLUMN]
nan_check_df = df[columns_to_check]
nan_rows = nan_check_df.isnull().any(axis=1)
num_nan_rows = nan_rows.sum()

if num_nan_rows > 0:
    logger.warning(f"Found {num_nan_rows} rows with NaN values in features or target.")
    df = df.dropna(subset=columns_to_check)
    logger.info(f"Dropped {initial_rows - len(df)} rows containing NaNs.")
    if df.empty:
        logger.error("DataFrame is empty after dropping NaNs. Exiting.")
        exit()
else:
    logger.info("No NaN values found in selected features or target.")
    
# Ensure all feature columns are numeric before variance check and training
try:
    df[features] = df[features].astype(float) 
except Exception as e:
    logger.error(f"Could not convert all features to numeric type: {e}")
    exit()
    
# --- Validate Feature Variance ---
logger.info("Validating feature variance...")
constant_features = []
for col in features:
    if df[col].nunique(dropna=True) <= 1:
        logger.warning(f"Feature '{col}' has no variance (or only one unique value). It will be dropped.")
        constant_features.append(col)

if constant_features:
    features = [f for f in features if f not in constant_features]
    logger.info(f"Removed {len(constant_features)} constant features.")
    if not features:
        logger.error("No features remaining after removing constant ones. Exiting.")
        exit()
        
logger.info(f"Final features for training ({len(features)}): {features}")

# --- Temporal Train/Test Split using TimeSeriesSplit ---
logger.info(f"Setting up TimeSeriesSplit with {N_SPLITS} splits...")

# Data needs to be sorted by time for TimeSeriesSplit
df = df.sort_values(TIMESTAMP_COLUMN)

X = df[features]
y = df[TARGET_COLUMN]

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# --- Model Training & Evaluation Loop ---
logger.info("Starting model training and evaluation loop...")

all_preds = []
all_y_test = []
fold_metrics = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    logger.info(f"--- Fold {fold + 1}/{N_SPLITS} ---")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    logger.info(f"Train indices: {len(train_index)} ({df.iloc[train_index][TIMESTAMP_COLUMN].min()} to {df.iloc[train_index][TIMESTAMP_COLUMN].max()})")
    logger.info(f"Test indices: {len(test_index)} ({df.iloc[test_index][TIMESTAMP_COLUMN].min()} to {df.iloc[test_index][TIMESTAMP_COLUMN].max()})")

    if X_train.empty or X_test.empty:
        logger.warning(f"Skipping Fold {fold + 1} due to empty train or test set.")
        continue

    # Define LightGBM Classifier Parameters
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss', 
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42 + fold, 
        'boosting_type': 'gbdt',
        'num_class': num_classes 
        # Consider adding class_weight='balanced' here if needed
    }

    model = lgb.LGBMClassifier(**params)

    # Early stopping
    eval_set = [(X_test, y_test)] 
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    logger.info(f"Training model for Fold {fold + 1}...")
    model.fit(X_train, y_train,
              eval_set=eval_set,
              callbacks=callbacks)

    # --- Evaluation for the Fold ---
    logger.info(f"Evaluating model on Fold {fold + 1} test set...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Fold {fold + 1} Test Accuracy: {acc:.4f}")
    
    # Store results for overall evaluation
    all_preds.extend(preds)
    all_y_test.extend(y_test)
    fold_metrics.append({'fold': fold + 1, 'accuracy': acc})
    
    # Save the model from the last fold only
    if fold == N_SPLITS - 1:
        # NOTE: Changed model save path
        model_path = os.path.join(MODEL_OUTPUT_DIR, f'lgbm_temporal_baseline_classifier_fold{fold+1}.txt')
        model.booster_.save_model(model_path)
        logger.info(f"Saved final fold model to {model_path}")
        
        # --- Feature Importance (from last fold model) ---
        logger.info("Calculating feature importance for the last fold model...")
        try:
            current_features = model.booster_.feature_name()
            importance_df = pd.DataFrame({
                'feature': current_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("Top 20 Feature Importances (Last Fold):")
            print(importance_df.head(20))

            # Plot feature importance
            plt.figure(figsize=(10, 8))
            # Adjust plot size if fewer features
            num_features_to_plot = min(20, len(importance_df))
            sns.barplot(x='importance', y='feature', data=importance_df.head(num_features_to_plot)) 
            plt.title(f'LightGBM Feature Importance (Temporal Baseline - Fold {fold + 1})') # Changed title
            plt.tight_layout()
            # NOTE: Changed plot save path
            plot_path = os.path.join(PLOTS_OUTPUT_DIR, f'temporal_feature_importance_fold{fold+1}.png')
            plt.savefig(plot_path)
            logger.info(f"Saved feature importance plot to {plot_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Could not calculate or plot feature importance: {e}")

# --- Overall Evaluation ---
if all_y_test:
    logger.info("--- Overall Evaluation (Temporal Baseline - Aggregated over folds) ---") # Changed title
    overall_accuracy = accuracy_score(all_y_test, all_preds)
    logger.info(f"Overall Test Accuracy: {overall_accuracy:.4f}")
    
    print("Overall Classification Report (Temporal Baseline):") # Changed title
    target_names = [f'Class {i}' for i in sorted(df[TARGET_COLUMN].unique())]
    print(classification_report(all_y_test, all_preds, target_names=target_names))
    
    print("Fold Accuracy Summary:")
    print(pd.DataFrame(fold_metrics))
else:
    logger.warning("No evaluation results generated as no folds completed successfully.")

logger.info("Temporal Baseline classification training script finished.") # Changed final message 