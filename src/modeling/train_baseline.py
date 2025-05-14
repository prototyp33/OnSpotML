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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATA_PATH = 'data/processed/parking_predictions_with_pois_local_filtered.parquet'
TARGET_COLUMN = 'prediction_code' # Confirmed categorical target
TIMESTAMP_COLUMN = 'timestamp'
MODEL_OUTPUT_DIR = 'models/baseline'
PLOTS_OUTPUT_DIR = 'reports/figures/baseline' # For feature importance plot
N_SPLITS = 3 # Number of splits for TimeSeriesSplit

# Ensure output directories exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
logging.info(f"Loading data from {DATA_PATH}...")
try:
    df = pd.read_parquet(DATA_PATH)
    logging.info(f"Loaded {len(df)} records.")
    
    # Drop geometry column if it exists
    if 'geometry_parsed' in df.columns:
        df = df.drop(columns=['geometry_parsed'])
        logging.info("Dropped 'geometry_parsed' column.")
    elif 'geometry' in df.columns: 
        df = df.drop(columns=['geometry'])
        logging.info("Dropped 'geometry' column.")
        
except FileNotFoundError:
    logging.error(f"Data file not found at {DATA_PATH}. Please ensure the previous steps ran successfully.")
    exit()
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit()

# Convert timestamp column to datetime
logging.info(f"Converting {TIMESTAMP_COLUMN} to datetime...")
if TIMESTAMP_COLUMN not in df.columns:
    logging.error(f"Timestamp column '{TIMESTAMP_COLUMN}' not found.")
    exit()
try:
    # Assuming timezone info is present and correct, keep it
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN]) 
except Exception as e:
    logging.error(f"Error converting timestamp column: {e}")
    exit()

# Check if target column exists
if TARGET_COLUMN not in df.columns:
    logging.error(f"Target column '{TARGET_COLUMN}' not found.")
    exit()

# Determine number of classes
num_classes = df[TARGET_COLUMN].nunique()
logging.info(f"Target column '{TARGET_COLUMN}' found with {num_classes} unique classes: {sorted(df[TARGET_COLUMN].unique())}")
if num_classes <= 1:
    logging.error("Target column has only one class. Cannot train classifier.")
    exit()

# --- Feature Engineering (Temporal Features Only) ---
logging.info("Generating temporal features...")
try:
    # Sort by timestamp just in case, although not strictly needed for these features
    # df = df.sort_values(TIMESTAMP_COLUMN)
    
    # Cyclical time features
    df['hour'] = df[TIMESTAMP_COLUMN].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['dayofweek'] = df[TIMESTAMP_COLUMN].dt.dayofweek # Monday=0, Sunday=6
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    df['month'] = df[TIMESTAMP_COLUMN].dt.month
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12) # Adjust month to 0-11 for calc
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    # Weekend flag
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Drop intermediate columns if desired
    # df = df.drop(columns=['hour', 'dayofweek', 'month'])
    
    logging.info("Temporal features generated.")
except Exception as e:
    logging.error(f"Error generating temporal features: {e}")
    exit()

# --- POI Feature Transformation (Log and Binary Presence) ---
logging.info("Transforming POI count features...")
original_poi_count_features = [col for col in df.columns if col.startswith('poi_') and 'count' in col]

if not original_poi_count_features:
    logging.warning("No original POI count features found to transform.")
else:
    logging.info(f"Found {len(original_poi_count_features)} POI count features to transform.") # Simplified log
    for col in original_poi_count_features:
        presence_col_name = f"{col}_present"
        df[presence_col_name] = (df[col] > 0).astype(int)
        # logging.info(f"Created binary presence feature: {presence_col_name}") # Reduce verbosity
        df[col] = np.log1p(df[col])
        # logging.info(f"Applied log1p transformation to: {col}") # Reduce verbosity
    logging.info(f"Created {len(original_poi_count_features)} binary presence features and applied log1p to counts.")
    logging.info("POI feature transformations complete.")

# --- Feature Selection ---
logging.info("Selecting features...")

# Start with temporal features
temporal_features = [
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'is_weekend',
]

# Dynamically discover transformed POI count features and new binary presence features
log_transformed_poi_counts = [col for col in df.columns if col.startswith('poi_') and 'count' in col and not col.endswith('_present')]
binary_poi_presence_features = [col for col in df.columns if col.startswith('poi_') and col.endswith('_present')]

poi_features = log_transformed_poi_counts + binary_poi_presence_features

if not poi_features:
    logging.warning("No POI features (log-transformed counts or binary presence) found after transformation.")
else:
    logging.info(f"Found {len(poi_features)} POI features for model training (log-transformed counts and binary presence).")

base_features = temporal_features + poi_features

logging.info(f"Total potential baseline features to consider: {len(base_features)}")

# Check which features actually exist in the DataFrame 
available_features = [f for f in base_features if f in df.columns]
missing_features = [f for f in base_features if f not in df.columns]

if missing_features:
    # This shouldn't happen with dynamic discovery unless temporal features have issues
    logging.warning(f"The following features were unexpectedly not found and will be excluded:")
    for f in missing_features:
        logging.warning(f" - {f}")

logging.info(f"Using {len(available_features)} available features for baseline model:")
# Limit printing if too many features
if len(available_features) < 80:
     for f in available_features:
         logging.info(f" - {f}")
else:
     logging.info(f"(Feature list too long to print, using {len(available_features)} features)")

features = available_features

if not features:
    logging.error("No features available for training after selection. Exiting.")
    exit()
    
# --- Handle NaNs (if any) ---
# Lag features omitted, so less likely to have NaNs unless present in original data
initial_rows = len(df)
original_columns = df.columns.tolist()

# Check for NaNs in features and target
columns_to_check = features + [TARGET_COLUMN]
nan_check_df = df[columns_to_check]
nan_rows = nan_check_df.isnull().any(axis=1)
num_nan_rows = nan_rows.sum()

if num_nan_rows > 0:
    logging.warning(f"Found {num_nan_rows} rows with NaN values in features or target.")
    # Simple strategy: Drop rows with NaNs for baseline
    df = df.dropna(subset=columns_to_check)
    logging.info(f"Dropped {initial_rows - len(df)} rows containing NaNs.")
    if df.empty:
        logging.error("DataFrame is empty after dropping NaNs. Exiting.")
        exit()
else:
    logging.info("No NaN values found in selected features or target.")
    
# Ensure all feature columns are numeric before variance check and training
try:
    df[features] = df[features].astype(float) # Attempt conversion
except Exception as e:
    logging.error(f"Could not convert all features to numeric type: {e}")
    # More granular check could be added here to identify specific non-numeric columns
    exit()
    
# --- Validate Feature Variance ---
logging.info("Validating feature variance...")
constant_features = []
for col in features:
    if df[col].nunique(dropna=True) <= 1:
        logging.warning(f"Feature '{col}' has no variance (or only one unique value). It will be dropped.")
        constant_features.append(col)

if constant_features:
    features = [f for f in features if f not in constant_features]
    logging.info(f"Removed {len(constant_features)} constant features.")
    if not features:
        logging.error("No features remaining after removing constant ones. Exiting.")
        exit()
        
logging.info(f"Final features for training ({len(features)}): {features}")

# --- Temporal Train/Test Split using TimeSeriesSplit ---
logging.info(f"Setting up TimeSeriesSplit with {N_SPLITS} splits...")

# Data needs to be sorted by time for TimeSeriesSplit
df = df.sort_values(TIMESTAMP_COLUMN)

X = df[features]
y = df[TARGET_COLUMN]

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# --- Model Training & Evaluation Loop ---
logging.info("Starting model training and evaluation loop...")

all_preds = []
all_y_test = []
fold_metrics = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    logging.info(f"--- Fold {fold + 1}/{N_SPLITS} ---")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    logging.info(f"Train indices: {len(train_index)} ({df.iloc[train_index][TIMESTAMP_COLUMN].min()} to {df.iloc[train_index][TIMESTAMP_COLUMN].max()})")
    logging.info(f"Test indices: {len(test_index)} ({df.iloc[test_index][TIMESTAMP_COLUMN].min()} to {df.iloc[test_index][TIMESTAMP_COLUMN].max()})")

    if X_train.empty or X_test.empty:
        logging.warning(f"Skipping Fold {fold + 1} due to empty train or test set.")
        continue

    # Define LightGBM Classifier Parameters
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss', # or 'multi_error'
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42 + fold, # Vary seed per fold
        'boosting_type': 'gbdt',
        'num_class': num_classes # Crucial for multiclass
    }

    model = lgb.LGBMClassifier(**params)

    # Early stopping
    # Note: Early stopping uses the *first* set in eval_set by default for stopping criteria
    eval_set = [(X_test, y_test)] 
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)] # verbose=True for details

    logging.info(f"Training model for Fold {fold + 1}...")
    model.fit(X_train, y_train,
              eval_set=eval_set,
              callbacks=callbacks)

    # --- Evaluation for the Fold ---
    logging.info(f"Evaluating model on Fold {fold + 1} test set...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info(f"Fold {fold + 1} Test Accuracy: {acc:.4f}")
    
    # Store results for overall evaluation
    all_preds.extend(preds)
    all_y_test.extend(y_test)
    fold_metrics.append({'fold': fold + 1, 'accuracy': acc})
    
    # Save the model from the last fold only (or implement logic to save best model)
    if fold == N_SPLITS - 1:
        model_path = os.path.join(MODEL_OUTPUT_DIR, f'lgbm_baseline_classifier_fold{fold+1}.txt')
        model.booster_.save_model(model_path)
        logging.info(f"Saved final fold model to {model_path}")
        
        # --- Feature Importance (from last fold model) ---
        logging.info("\nCalculating feature importance for the last fold model...")
        try:
            # Need to handle cases where features might have changed if constant ones were dropped
            # Ensure feature names match the model's internal names
            current_features = model.booster_.feature_name()
            importance_df = pd.DataFrame({
                'feature': current_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logging.info("Top 20 Feature Importances (Last Fold):")
            print(importance_df.head(20))

            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(20)) # Plot top 20
            plt.title(f'LightGBM Feature Importance (Last Fold - Fold {fold + 1})')
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_OUTPUT_DIR, f'feature_importance_fold{fold+1}.png')
            plt.savefig(plot_path)
            logging.info(f"Saved feature importance plot to {plot_path}")
            plt.close()

        except Exception as e:
            logging.error(f"Could not calculate or plot feature importance: {e}")

# --- Overall Evaluation ---
if all_y_test:
    logging.info("\n--- Overall Evaluation (Aggregated over folds) ---")
    overall_accuracy = accuracy_score(all_y_test, all_preds)
    logging.info(f"Overall Test Accuracy: {overall_accuracy:.4f}")
    
    print("\nOverall Classification Report:")
    # Ensure target_names are sorted correctly if you want labels
    target_names = [f'Class {i}' for i in sorted(df[TARGET_COLUMN].unique())]
    print(classification_report(all_y_test, all_preds, target_names=target_names))
    
    print("\nFold Accuracy Summary:")
    print(pd.DataFrame(fold_metrics))
else:
    logging.warning("No evaluation results generated as no folds completed successfully.")

logging.info("Baseline classification training script finished.") 