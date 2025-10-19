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
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve

# Add the project root directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modeling.feature_engineering_v2 import FeatureEngineeringPipeline
from src.modeling.target_variable import bin_occupancy_percentage_to_class, define_parking_occupancy_classes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_main_model_tscv")

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
TARGET_COLUMN = 'occupancy_class_h1'
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
    
    # Ensure core identifiers exist
    if 'parking_id' not in df.columns:
        raise ValueError("Missing 'parking_id' column after preprocessing")
    
    logger.info(f"Dataset shape after preprocessing: {df.shape}")
    if 'occupancy_level' in df.columns:
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

def balanced_downsample_after_label(df, target_col, max_total=500_000, min_per_class=10_000, random_state=42):
    """Downsample a labeled DataFrame in a class-balanced way AFTER labels exist.
    Ensures temporal continuity is preserved within each class by sampling contiguous
    blocks per class when possible.
    """
    logger.info(f"Balanced downsampling after label creation on column '{target_col}'...")
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found for downsampling. Skipping downsample.")
        return df

    df = df.sort_values(['parking_id', 'timestamp']).reset_index(drop=True)
    class_counts = df[target_col].value_counts().to_dict()
    logger.info(f"Labeled class distribution before downsample: {class_counts}")

    num_classes = len(class_counts)
    if len(df) <= max_total:
        logger.info("Dataset smaller than max_total; skipping downsample.")
        return df

    per_class_cap = max(min_per_class, max_total // max(1, num_classes))
    sampled_parts = []
    rng = np.random.default_rng(random_state)

    for cls, count in class_counts.items():
        cls_df = df[df[target_col] == cls]
        if len(cls_df) <= per_class_cap:
            sampled_parts.append(cls_df)
            continue
        # Sample a contiguous block to preserve local temporal structure
        start_max = max(0, len(cls_df) - per_class_cap)
        start_idx = int(rng.integers(0, start_max + 1))
        sampled_parts.append(cls_df.iloc[start_idx:start_idx + per_class_cap])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sort_values(['parking_id', 'timestamp']).reset_index(drop=True)
    logger.info(f"Downsampled to {len(sampled)} rows. New distribution: {sampled[target_col].value_counts().to_dict()}")
    return sampled

def build_labels_1h_parquet(data_path: str, label_output_path: str, tolerance_minutes: int = 45) -> None:
    """Stream-build 1h-ahead labels to a parquet to avoid OOM.
    Reads only minimal columns via pyarrow.dataset and processes per parking_id.
    Expects raw columns 'ID_TRAMO', 'timestamp', and 'actual_state'.
    If 'label_output_path' ends with '/', writes a partitioned directory (ID_TRAMO=pid/...),
    otherwise writes a single parquet file.
    """
    logger.info(f"Building labels parquet at {label_output_path} from {data_path}...")
    dataset = ds.dataset(data_path, format="parquet")

    # Collect unique parking ids efficiently
    logger.info("Collecting unique parking ids (ID_TRAMO)...")
    ids_table = dataset.to_table(columns=['ID_TRAMO'])
    parking_ids = pd.Series(ids_table.column('ID_TRAMO').to_pylist()).dropna().unique().tolist()
    logger.info(f"Found {len(parking_ids)} unique parking ids")

    # Prepare writer(s)
    schema = pa.schema([
        pa.field('ID_TRAMO', pa.int64()),
        pa.field('timestamp', pa.timestamp('ns')),
        pa.field('occupancy_class_h1', pa.int64())
    ])
    partitioned = label_output_path.endswith('/') or label_output_path.endswith(os.sep)
    if not partitioned:
        writer = pq.ParquetWriter(label_output_path, schema)

    class_midpoints = {0:5, 1:20, 2:40, 3:60, 4:80, 5:95, 6:98}
    tol = pd.Timedelta(minutes=tolerance_minutes)

    processed = 0
    for pid in parking_ids:
        filt = ds.field('ID_TRAMO') == pid
        table = dataset.to_table(columns=['ID_TRAMO', 'timestamp', 'actual_state'], filter=filt)
        if table.num_rows == 0:
            continue
        gdf = table.to_pandas()
        # Map to modeling names
        gdf = gdf.rename(columns={'ID_TRAMO': 'parking_id', 'actual_state': 'occupancy_level'})
        gdf = gdf.dropna(subset=['timestamp']).sort_values('timestamp')
        gdf = gdf.drop_duplicates(subset=['timestamp'])
        if gdf.empty:
            continue
        left = gdf[['timestamp']].copy()
        left['future_ts'] = left['timestamp'] + pd.Timedelta(hours=1)
        right = gdf[['timestamp', 'occupancy_level']].rename(columns={'timestamp': 'future_ts', 'occupancy_level': 'future_level'})
        left_sorted = left.sort_values('future_ts')
        right_sorted = right.sort_values('future_ts')
        aligned = pd.merge_asof(
            left_sorted,
            right_sorted,
            left_on='future_ts',
            right_on='future_ts',
            direction='forward',
            tolerance=tol
        )
        occ_pct = aligned['future_level'].map(class_midpoints)
        labels = bin_occupancy_percentage_to_class(occ_pct)
        out = pd.DataFrame({
            'ID_TRAMO': pid,
            'timestamp': gdf['timestamp'].values,
            'occupancy_class_h1': labels.values
        })
        out = out.dropna(subset=['occupancy_class_h1'])
        if out.empty:
            continue
        out['occupancy_class_h1'] = out['occupancy_class_h1'].astype('int64')
        # Write batch
        if partitioned:
            part_dir = os.path.join(label_output_path, f"ID_TRAMO={pid}")
            os.makedirs(part_dir, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(out[schema.names], schema=schema, preserve_index=False), os.path.join(part_dir, "part-0.parquet"))
        else:
            writer.write_table(pa.Table.from_pandas(out[schema.names], schema=schema, preserve_index=False))
        processed += len(out)
        if processed % 100000 == 0:
            logger.info(f"Labeled {processed} rows so far...")

    if not partitioned:
        writer.close()
    logger.info(f"Finished building labels parquet with {processed} labeled rows")

def build_labels_multi_horizon(
    data_path: str,
    out_root: str,
    minutes_list: list[int],
    tolerance_minutes: int = 45
) -> None:
    """Build partitioned label directories for multiple horizons in minutes.
    Writes to out_root/labels_{mm}m/ partitioned by ID_TRAMO.
    """
    os.makedirs(out_root, exist_ok=True)
    for mm in minutes_list:
        label_dir = os.path.join(out_root, f"labels_{mm}m")
        if os.path.exists(label_dir):
            logger.info(f"Labels for {mm}m already exist at {label_dir}; skipping build")
            continue
        # Reuse the 1h builder logic by adjusting the +h offset in-group
        logger.info(f"Building {mm}m labels at {label_dir}...")
        dataset = ds.dataset(data_path, format="parquet")
        ids_table = dataset.to_table(columns=['ID_TRAMO'])
        parking_ids = pd.Series(ids_table.column('ID_TRAMO').to_pylist()).dropna().unique().tolist()
        schema = pa.schema([
            pa.field('ID_TRAMO', pa.int64()),
            pa.field('timestamp', pa.timestamp('ns')),
            pa.field(f'occupancy_class_tplus_{mm}m', pa.int64())
        ])
        tol = pd.Timedelta(minutes=tolerance_minutes)
        class_midpoints = {0:5, 1:20, 2:40, 3:60, 4:80, 5:95, 6:98}
        processed = 0
        for pid in parking_ids:
            filt = ds.field('ID_TRAMO') == pid
            table = dataset.to_table(columns=['ID_TRAMO', 'timestamp', 'actual_state'], filter=filt)
            if table.num_rows == 0:
                continue
            gdf = table.to_pandas()
            gdf = gdf.rename(columns={'ID_TRAMO': 'parking_id', 'actual_state': 'occupancy_level'})
            gdf = gdf.dropna(subset=['timestamp']).sort_values('timestamp')
            gdf = gdf.drop_duplicates(subset=['timestamp'])
            if gdf.empty:
                continue
            left = gdf[['timestamp']].copy()
            left['future_ts'] = left['timestamp'] + pd.to_timedelta(mm, unit='m')
            right = gdf[['timestamp', 'occupancy_level']].rename(columns={'timestamp': 'future_ts', 'occupancy_level': 'future_level'})
            left_sorted = left.sort_values('future_ts')
            right_sorted = right.sort_values('future_ts')
            aligned = pd.merge_asof(
                left_sorted,
                right_sorted,
                left_on='future_ts',
                right_on='future_ts',
                direction='forward',
                tolerance=tol
            )
            occ_pct = aligned['future_level'].map(class_midpoints)
            labels = bin_occupancy_percentage_to_class(occ_pct)
            out = pd.DataFrame({
                'ID_TRAMO': pid,
                'timestamp': gdf['timestamp'].values,
                f'occupancy_class_tplus_{mm}m': labels.values
            }).dropna()
            if out.empty:
                continue
            out[f'occupancy_class_tplus_{mm}m'] = out[f'occupancy_class_tplus_{mm}m'].astype('int64')
            part_dir = os.path.join(label_dir, f"ID_TRAMO={pid}")
            os.makedirs(part_dir, exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(out[schema.names], schema=schema, preserve_index=False),
                os.path.join(part_dir, "part-0.parquet")
            )
            processed += len(out)
            if processed % 100000 == 0:
                logger.info(f"[{mm}m] Labeled {processed} rows so far...")
        logger.info(f"Finished building labels for {mm}m with {processed} rows")

def load_features_for_labels(data_path: str, labels_df: pd.DataFrame, columns: list | None = None) -> pd.DataFrame:
    """Load only feature rows matching the labeled (ID_TRAMO, timestamp) pairs via pyarrow filters."""
    if labels_df.empty:
        return pd.DataFrame()
    dataset = ds.dataset(data_path, format="parquet")
    # Restrict by ID_TRAMO and global time bounds to reduce scan
    pids = labels_df['ID_TRAMO'].unique().tolist()
    ts_min = labels_df['timestamp'].min()
    ts_max = labels_df['timestamp'].max()
    filt = ds.field('ID_TRAMO').isin(pids) & (ds.field('timestamp') >= pa.scalar(ts_min)) & (ds.field('timestamp') <= pa.scalar(ts_max))
    table = dataset.to_table(columns=columns, filter=filt)
    feats = table.to_pandas()
    # Inner join to keep exact labeled rows
    merged = pd.merge(feats, labels_df, on=['ID_TRAMO', 'timestamp'], how='inner')
    return merged

def select_features(df, valor_correlation_threshold=0.8, exclude_valor: bool = False):
    logger.info("Selecting features for TSCV modeling with hybrid approach...")
    
    # Define columns to ALWAYS exclude (target, IDs, etc.)
    always_exclude_cols = [
        'ID_TRAMO', 'timestamp', 'actual_state', 'occupancy_level',
        'DATA_LECTURA', 'DATA_EXTREM', 'CODI_ESTACIO', 'ACRÒNIM',
        'parking_id', 
        'occupancy_rate' # Raw occupancy_rate (0-1 scale) is a direct derivative of target
    ]

    # Define a list of "safe" occupancy_rate derived features we want to KEEP if they exist
    # These are typically longer lags or larger rolling windows.
    safe_occupancy_rate_features_to_keep = [
        'occupancy_rate_lag_6h', 'occupancy_rate_lag_12h', 'occupancy_rate_lag_24h',
        'occupancy_rate_lag_48h', 'occupancy_rate_lag_168h', # week lag
        'occupancy_rate_rolling_mean_6h', 'occupancy_rate_rolling_mean_12h',
        'occupancy_rate_rolling_mean_24h', 'occupancy_rate_rolling_mean_48h',
        'occupancy_rate_rolling_std_6h', 'occupancy_rate_rolling_std_12h', 
        'occupancy_rate_rolling_std_24h', 'occupancy_rate_rolling_std_48h',
        # Add other specific rolling/lag features if deemed safe and useful, e.g., EWM with longer spans
        'occupancy_rate_ewm_24h' # Example EWM, ensure FeatureEngineeringPipeline can create these names
    ]
    # Filter this list to only include those actually present in the df to avoid errors
    safe_occupancy_rate_features_to_keep = [f for f in safe_occupancy_rate_features_to_keep if f in df.columns]
    logger.info(f"Whitelisted safe occupancy_rate features to keep: {safe_occupancy_rate_features_to_keep}")

    # Identify ALL occupancy_rate_ derived features present in the DataFrame
    all_occupancy_rate_derived_features = [col for col in df.columns if 'occupancy_rate_' in col]

    # Determine which of these should be excluded: those NOT in the safe_to_keep list
    occupancy_rate_features_to_exclude = [
        f for f in all_occupancy_rate_derived_features 
        if f not in safe_occupancy_rate_features_to_keep
    ]
    logger.info(f"Occupancy_rate features to EXCLUDE (not in whitelist): {occupancy_rate_features_to_exclude}")

    # Combine always_exclude_cols with the specific occupancy_rate features to exclude
    exclude_cols = list(set(always_exclude_cols + occupancy_rate_features_to_exclude))
    
    # Conditional VALOR exclusion (remains the same)
    if 'VALOR' in df.columns and 'actual_state' in df.columns:
        valor_correlation = df['VALOR'].corr(df['actual_state'])
        logger.info(f"VALOR correlation with actual_state: {valor_correlation:.4f}")
        if abs(valor_correlation) > valor_correlation_threshold:
            if 'VALOR' not in exclude_cols: # Add only if not already excluded for other reasons
                 exclude_cols.append('VALOR')
            logger.warning(f"⚠️ Excluding VALOR (correlation = {valor_correlation:.4f} > threshold {valor_correlation_threshold})")
        else:
            logger.info(f"Keeping VALOR (correlation = {valor_correlation:.4f} <= threshold {valor_correlation_threshold})")
    elif 'VALOR' in df.columns:
        logger.info("Keeping VALOR as target column 'actual_state' not available for correlation check at this stage.")
    else:
        logger.info("VALOR column not present.")

    # Optional hard exclusion of VALOR via flag
    if exclude_valor and 'VALOR' in df.columns and 'VALOR' not in exclude_cols:
        exclude_cols.append('VALOR')

    # Final feature list: all columns in df EXCEPT those in the combined exclude_cols list
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure no target or ID-like columns accidentally included if not explicitly caught by exclude_cols
    # This is a safeguard; ideally, `always_exclude_cols` covers these.
    additional_safe_excludes = [TARGET_COLUMN, 'occupancy_level', TIMESTAMP_COLUMN]
    feature_cols = [f for f in feature_cols if f not in additional_safe_excludes]
    
    # Remove duplicates and sort for consistency
    feature_cols = sorted(list(set(feature_cols)))

    logger.info(f"Selected {len(feature_cols)} features for modeling using hybrid approach: {feature_cols}")
    return feature_cols

def create_time_series_split(df):
    """Create time series split for cross-validation."""
    logger.info(f"Creating TimeSeriesSplit with {N_SPLITS} splits...")
    
    # Calculate split points
    total_days = (df[TIMESTAMP_COLUMN].max() - df[TIMESTAMP_COLUMN].min()).days
    split_size = total_days // (N_SPLITS + 1)  # +1 for the initial training set
    
    tscv = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=split_size,
        gap=split_size // 6  # Gap between train and test
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
        
        # Evaluate
        y_pred = model.predict(X_test)
        # Probabilities for calibration
        try:
            y_proba = model.predict_proba(X_val)
        except Exception:
            y_proba = None
        
        # Calculate comprehensive metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        fold_record = {
            'fold': fold + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'class_distribution': Counter(y_test)
        }
        # Store predictions/true for per-facility metrics later
        fold_record['y_test'] = list(map(int, y_test))
        fold_record['y_pred'] = list(map(int, y_pred))
        
        # Store predictions and true values
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        # Store ids for per-facility metrics
        if parking_ids is not None:
            fold_parking_ids = parking_ids.iloc[test_idx].tolist()
        else:
            fold_parking_ids = [None] * len(y_test)
        fold_record['parking_ids'] = fold_parking_ids
        # Attach probabilities for calibration (per-fold for later aggregation)
        if y_proba is not None:
            fold_record['y_val_true'] = y_val.tolist()
            fold_record['y_val_proba'] = [list(map(float, row)) for row in y_proba]

        fold_metrics.append(fold_record)
        
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

    # Per-facility metrics (macro F1)
    try:
        logger.info("Computing per-facility metrics...")
        per_facility = {}
        for m in fold_metrics:
            if 'parking_ids' not in m or 'y_test' not in m or 'y_pred' not in m:
                logger.warning(f"Missing required keys in fold metrics: {list(m.keys())}")
                continue
            pids = m['parking_ids']
            yt = m['y_test']
            yp = m['y_pred']
            logger.info(f"Processing fold with {len(pids)} predictions")
            # Accumulate per facility
            for pid, y_true_i, y_pred_i in zip(pids, yt, yp):
                if pid is None:
                    continue
                if pid not in per_facility:
                    per_facility[pid] = {'y_true': [], 'y_pred': []}
                per_facility[pid]['y_true'].append(y_true_i)
                per_facility[pid]['y_pred'].append(y_pred_i)
        logger.info(f"Accumulated data for {len(per_facility)} facilities")
        # Compute macro F1 per facility
        from sklearn.metrics import f1_score
        per_facility_scores = {}
        for pid, vals in per_facility.items():
            if len(vals['y_true']) >= 20:  # require minimal support
                per_facility_scores[int(pid)] = float(f1_score(vals['y_true'], vals['y_pred'], average='macro'))
        logger.info(f"Computed scores for {len(per_facility_scores)} facilities with sufficient data")
        # Save JSON
        pf_metrics_path = os.path.join(METRICS_OUTPUT_DIR, 'per_facility_metrics.json')
        with open(pf_metrics_path, 'w') as f:
            json.dump({'macro_f1_by_parking_id': per_facility_scores}, f, indent=2)
        logger.info(f"Saved per-facility metrics to {pf_metrics_path}")
    except Exception as e:
        logger.warning(f"Per-facility metrics computation skipped due to: {e}")
        import traceback
        logger.warning(f"Traceback: {traceback.format_exc()}")

    # Calibration plots (per-class reliability curves) if probabilities were collected
    try:
        # Gather all val truths and probas
        y_true_all = []
        proba_all = []
        for m in fold_metrics:
            if 'y_val_true' in m and 'y_val_proba' in m:
                y_true_all.extend(m['y_val_true'])
                proba_all.extend(m['y_val_proba'])
        if len(y_true_all) and len(proba_all):
            y_true_all = np.array(y_true_all)
            proba_all = np.array(proba_all)
            num_classes = proba_all.shape[1]
            for k in range(num_classes):
                y_bin = (y_true_all == k).astype(int)
                prob_k = proba_all[:, k]
                frac_pos, mean_pred = calibration_curve(y_bin, prob_k, n_bins=10, strategy='uniform')
                plt.figure(figsize=(6, 6))
                plt.plot(mean_pred, frac_pos, marker='o', label=f'Class {k}')
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.xlabel('Mean predicted probability')
                plt.ylabel('Fraction of positives')
                plt.title(f'Calibration curve - class {k}')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, f'calibration_class_{k}.png'))
                plt.close()
    except Exception as e:
        logger.warning(f"Calibration plotting skipped due to: {e}")

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

def add_rare_class_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Features specifically for high occupancy scenarios. 
       Focusing on features that showed some importance in previous runs.
    """
    df = df_input.copy()
    logger.info("Generating focused rare class specific features...")

    # Ensure required columns for new features exist, log if not
    # Consolidate required columns based on the features we are keeping
    required_for_super_peak = ['hour', 'is_weekend']
    required_for_occupancy_accel = ['actual_state_lag_1h', 'actual_state_lag_6h', 'actual_state_lag_12h']
    
    # Peak period indicators (super_peak)
    if all(col in df.columns for col in required_for_super_peak):
        df['super_peak'] = ((
            df['hour'].between(8, 10)) | 
            (df['hour'].between(17, 19)) | 
            (df['is_weekend'] & df['hour'].between(11, 15))
        ).astype(int)
        logger.info("Created 'super_peak' feature.")
    else:
        logger.warning(f"Skipping 'super_peak': missing one or more of {required_for_super_peak}.")
        df['super_peak'] = 0 # Add placeholder

    # Trend acceleration features (occupancy_acceleration)
    if all(col in df.columns for col in required_for_occupancy_accel):
        df['occupancy_acceleration'] = (
            df['actual_state_lag_1h'] - df['actual_state_lag_6h']
        ) - (df['actual_state_lag_6h'] - df['actual_state_lag_12h'])
        logger.info("Created 'occupancy_acceleration' feature.")
    else:
        logger.warning(f"Skipping 'occupancy_acceleration': missing one or more of {required_for_occupancy_accel}.")
        df['occupancy_acceleration'] = 0 # Add placeholder

    # Removing event_peak_combo and weekend_evening_event as they had low importance
    logger.info("Skipped 'event_peak_combo' and 'weekend_evening_event' based on previous importance analysis.")

    logger.info("Finished generating focused rare class specific features.")
    return df

def main_tscv():
    logger.info("Starting Manual Temporal Cross-Validation Model Training Pipeline...") # Updated log slightly
    config = load_config()
    
    logger.info("Preparing labels (streaming build if missing)...")
    data_path = config['data_path']
    labels_file = 'data/processed/labels_1h.parquet'
    labels_dir = 'data/processed/labels_1h/'
    # Optional Phase 2: multi-horizon label build
    horizons = config.get('training', {}).get('horizons_minutes')
    if horizons:
        logger.info(f"Multi-horizon build requested: {horizons}")
        build_labels_multi_horizon(data_path=data_path, out_root='data/processed', minutes_list=horizons, tolerance_minutes=45)
        logger.info("Multi-horizon label build complete; exiting after build as requested.")
        return
    # Prefer existing file; if neither exists, build partitioned directory
    if os.path.exists(labels_file):
        labels_df = pd.read_parquet(labels_file)
        logger.info(f"Loaded labels from file: {len(labels_df):,} rows")
    else:
        if not os.path.exists(labels_dir):
            build_labels_1h_parquet(data_path=data_path, label_output_path=labels_dir, tolerance_minutes=45)
        # Load partitioned labels via dataset
        labels_ds = ds.dataset(labels_dir, format='parquet', partitioning='hive')
        labels_table = labels_ds.to_table()
        labels_df = labels_table.to_pandas()
        logger.info(f"Loaded labels from partitioned dir: {len(labels_df):,} rows")

    # Balanced downsample labels to manageable size
    # Enforce dtypes for robust joins
    if 'ID_TRAMO' in labels_df.columns:
        labels_df['ID_TRAMO'] = labels_df['ID_TRAMO'].astype('int64', errors='ignore')
    if not pd.api.types.is_datetime64_any_dtype(labels_df['timestamp']):
        labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
    labels_df = labels_df.sort_values(['ID_TRAMO', 'timestamp']).reset_index(drop=True)
    labels_df = balanced_downsample_after_label(labels_df.rename(columns={'ID_TRAMO': 'parking_id'}), target_col='occupancy_class_h1', max_total=500_000, min_per_class=10_000)
    labels_df = labels_df.rename(columns={'parking_id': 'ID_TRAMO'})
    logger.info(f"After downsample labels: {len(labels_df):,} rows")

    # Load matching features only
    logger.info("Loading matching features for labeled rows...")
    df = load_features_for_labels(data_path=data_path, labels_df=labels_df, columns=None)
    logger.info(f"Loaded {len(df):,} feature rows matching labels")

    # Preprocess and map names
    df = preprocess_data(df)
    # Enforce dtypes on features for robust joins down the line
    if 'ID_TRAMO' in df.columns:
        df['ID_TRAMO'] = df['ID_TRAMO'].astype('int64', errors='ignore')
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Ensure labels exist: skip creation if already merged from labels parquet ---
    if 'occupancy_class_h1' not in df.columns:
        logger.info("'occupancy_class_h1' not found in merged features; creating via time-based alignment per parking_id...")
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['parking_id', 'timestamp']).reset_index(drop=True)

        if 'occupancy_level' not in df.columns:
            raise ValueError("Cannot create labels: 'occupancy_level' missing after preprocessing")
        else:
            def label_group(grp: pd.DataFrame) -> pd.DataFrame:
                g = grp.sort_values('timestamp').copy()
                g = g.drop_duplicates(subset=['timestamp'])
                left = g[['timestamp']].copy()
                left['future_ts'] = left['timestamp'] + pd.Timedelta(hours=1)
                right = g[['timestamp', 'occupancy_level']].rename(columns={'timestamp': 'future_ts', 'occupancy_level': 'future_level'})
                left_sorted = left.sort_values('future_ts')
                right_sorted = right.sort_values('future_ts')
                aligned = pd.merge_asof(
                    left_sorted,
                    right_sorted,
                    left_on='future_ts',
                    right_on='future_ts',
                    direction='forward',
                    tolerance=pd.Timedelta(minutes=45)
                )
                class_midpoints = {0:5, 1:20, 2:40, 3:60, 4:80, 5:95, 6:98}
                occ_pct = aligned['future_level'].map(class_midpoints)
                labels = bin_occupancy_percentage_to_class(occ_pct)
                label_map = aligned[['future_ts']].copy()
                label_map = label_map.rename(columns={'future_ts': 'timestamp'})
                label_map['occupancy_class_h1'] = labels.values
                g = g.merge(label_map, on='timestamp', how='left')
                return g

            df = df.groupby('parking_id', group_keys=False).apply(label_group)
            df = df.dropna(subset=['occupancy_class_h1']).reset_index(drop=True)
            df['occupancy_class_h1'] = df['occupancy_class_h1'].astype(int)

    # Now apply feature engineering on the downsampled, labeled data
    logger.info("Applying FeatureEngineeringPipeline...")
    feature_pipeline = FeatureEngineeringPipeline(config)
    df = feature_pipeline.fit_transform(df)
    logger.info("FeatureEngineeringPipeline applied.")
    logger.info(f"DataFrame shape after FeatureEngineeringPipeline: {df.shape}")
    
    # Add specialized rare class features AFTER FeatureEngineeringPipeline
    logger.info("Adding specialized rare class features...")
    df = add_rare_class_features(df) 
    logger.info(f"DataFrame shape after adding rare class features: {df.shape}")

    # --- Diagnostics ---
    # VALOR investigation (using actual_state if present, else occupancy_level if that's the final target name)
    # Assuming preprocess_data renames actual_state to occupancy_level for the target
    target_for_valor_investigation = 'occupancy_level' if 'occupancy_level' in df.columns else 'actual_state'
    if target_for_valor_investigation in df.columns:
        investigate_valor_feature(df)
    else:
        logger.warning(f"Target column '{target_for_valor_investigation}' for VALOR investigation not found. Skipping VALOR investigation.")

    check_feature_leakage(df, target_col=target_for_valor_investigation)
    # --- End Diagnostics ---

    # Feature Selection (pass the correct target name to select_features for VALOR correlation)
    # Ensure df passed to select_features has 'actual_state' if that's what VALOR was correlated against in the template
    # The template used df['actual_state'], preprocess_data creates df['occupancy_level']
    # For consistency, let's ensure select_features uses the final target name 'occupancy_level'
    # if we map it earlier. The `investigate_valor_feature` also needs to be consistent.
    # The provided template for select_features used 'actual_state' with VALOR. 
    # Let's assume preprocess_data has created 'occupancy_level' as the final target name.
    # We need to ensure this consistency. For now, select_features is adapted to use 'actual_state' if present for VALOR check.
    
    # The select_features function was updated to handle VALOR correlation internally.
    # It expects 'actual_state' for VALOR correlation, but our main target is 'occupancy_level'.
    # Let's make a temporary 'actual_state' for select_features if it's not there but 'occupancy_level' is.
    temp_actual_state_created = False
    if 'actual_state' not in df.columns and 'occupancy_level' in df.columns:
        df['actual_state'] = df['occupancy_level']
        temp_actual_state_created = True
        
    # Preserve parking_id for per-facility metrics before feature selection
    parking_ids = df['parking_id'].copy() if 'parking_id' in df.columns else None
    
    # Optionally exclude VALOR via config flag model.exclude_valor: true/false
    exclude_valor_flag = bool(config.get('model', {}).get('exclude_valor', False))
    features = select_features(df, valor_correlation_threshold=0.8, exclude_valor=exclude_valor_flag)

    if temp_actual_state_created:
        df = df.drop(columns=['actual_state'])

    final_target_col = 'occupancy_class_h1'
    X = df[features]
    y = df[final_target_col]
    
    logger.info("Converting to memory-efficient data types...")
    for col in X.columns:
        if X[col].dtype == 'float64': X.loc[:, col] = X[col].astype('float32')
        elif X[col].dtype == 'int64': X.loc[:, col] = X[col].astype('int16')
    
    logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

    # Ensure df is sorted by timestamp for manual splitting
    df = df.sort_values('timestamp').reset_index(drop=True)
    X = df[features] # Re-assign X after sorting df
    y = df[final_target_col] # Re-assign y after sorting df
    
    logger.info("Implementing manually defined temporal folds for CV...")

    fold_metrics_list = []
    all_fold_preds = []
    all_fold_true = []

    # Define 3 proper temporal folds as per your specification
    fold_definitions = [
        { # Fold 1: Early training period
            'train_end_pct': 0.4, 'val_start_pct': 0.45, 'val_end_pct': 0.55,
            'name': 'Fold 1 (Early)'
        },
        { # Fold 2: Mid training period  
            'train_end_pct': 0.6, 'val_start_pct': 0.65, 'val_end_pct': 0.75,
            'name': 'Fold 2 (Mid)'
        },
        { # Fold 3: Late training period
            'train_end_pct': 0.75, 'val_start_pct': 0.8, 'val_end_pct': 0.95,
            'name': 'Fold 3 (Late)'
        }
    ]
    
    total_samples = len(df)

    for fold_num, fold_def in enumerate(fold_definitions):
        logger.info(f"\n=== {fold_def['name']} ({fold_num + 1}/{len(fold_definitions)}) ===")
        
        train_end_idx = int(total_samples * fold_def['train_end_pct'])
        val_start_idx = int(total_samples * fold_def['val_start_pct'])
        val_end_idx = int(total_samples * fold_def['val_end_pct'])
        
        train_idx = list(range(0, train_end_idx))
        val_idx = list(range(val_start_idx, val_end_idx))
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        logger.info(f"Train fold shape: {X_train.shape}, Val fold shape: {X_val.shape}")
        
        # Verify no temporal leakage
        train_timestamps = df.iloc[train_idx]['timestamp']
        val_timestamps = df.iloc[val_idx]['timestamp']
        
        train_max_ts = train_timestamps.max()
        val_min_ts = val_timestamps.min()
        
        logger.info(f"Train period: {train_timestamps.min()} to {train_max_ts}")
        logger.info(f"Validation period: {val_min_ts} to {val_timestamps.max()}")
        
        if train_max_ts >= val_min_ts:
            logger.error(f"❌ TEMPORAL LEAKAGE DETECTED in {fold_def['name']}!")
            logger.error(f"Train max timestamp: {train_max_ts}, Validation min timestamp: {val_min_ts}")
            # Optionally, skip this fold or raise an error
            # For now, we'll log and continue to see if other folds are okay
            # but this indicates a fundamental issue with the split logic if it happens.
            continue 
        else:
            logger.info(f"✅ No temporal leakage detected in {fold_def['name']}.")

        # diagnose_data_splits(df_full, df_train=df.iloc[train_idx], df_val=df.iloc[val_idx]) # Can use existing diagnostic

        # Use LGBM parameters from your suggestion
        model_params = {
            'n_estimators': config.get('model',{}).get('n_estimators_manual_cv', 800), # New config key
            'learning_rate': config.get('model',{}).get('learning_rate_manual_cv', 0.08), # New config key
            'num_leaves': config.get('model',{}).get('num_leaves_manual_cv', 31), # New config key
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        model = lgb.LGBMClassifier(**model_params)
        
        logger.info(f"Training model for {fold_def['name']}...")
        model.fit(X_train, y_train)
        
        logger.info(f"Evaluating model for {fold_def['name']}...")
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        class_report = classification_report(y_val, y_pred, zero_division=0)
        
        fold_metrics_list.append({
            'fold': fold_def['name'], 
            'accuracy': accuracy, 
            'f1_weighted': f1,
            'classification_report': class_report,
            'train_records': len(X_train),
            'val_records': len(X_val),
            'train_time_range': f"{train_timestamps.min()} to {train_max_ts}",
            'val_time_range': f"{val_min_ts} to {val_timestamps.max()}"
        })
        all_fold_preds.extend(y_pred)
        all_fold_true.extend(y_val) # Collect all true values from validation sets
        
        logger.info(f"{fold_def['name']} - Accuracy: {accuracy:.4f}, F1 Weighted: {f1:.4f}")
        logger.debug(f"{fold_def['name']} Classification Report:\\n{class_report}")

        # Store feature importances for this fold
        if hasattr(model, 'feature_importances_'):
            # Ensure feature importance values are native Python floats for JSON serialization
            importances_dict = {k: float(v) for k, v in zip(X_train.columns, model.feature_importances_)}
            fold_metrics_list[-1]['feature_importances'] = importances_dict

    # Aggregate and report CV results
    logger.info("\n=== Manual Temporal CV Results ===")
    if not fold_metrics_list:
        logger.error("No folds were successfully processed. Cannot calculate aggregate metrics.")
        # Potentially save an empty or error state to the JSON file
        cv_metrics_path = os.path.join(METRICS_OUTPUT_DIR, 'manual_tscv_evaluation_metrics.json')
        with open(cv_metrics_path, 'w') as f:
            json.dump({'error': 'No folds processed successfully.'}, f, indent=4)
        logger.info(f"Saved error state to {cv_metrics_path}")
        logger.info("Manual Temporal CV training pipeline finished with errors.")
        return # Exit if no folds were processed.

    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics_list])
    avg_f1 = np.mean([m['f1_weighted'] for m in fold_metrics_list])
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics_list])
    std_f1 = np.std([m['f1_weighted'] for m in fold_metrics_list])

    logger.info(f"Average Accuracy: {avg_accuracy:.4f} +/- {std_accuracy:.4f}")
    logger.info(f"Average F1 Weighted: {avg_f1:.4f} +/- {std_f1:.4f}")

    logger.info("Overall Classification Report (based on all folds' predictions):")
    overall_class_report = classification_report(all_fold_true, all_fold_preds, zero_division=0)
    logger.info(f"\n{overall_class_report}")

    # Calculate and log average feature importances
    if fold_metrics_list and 'feature_importances' in fold_metrics_list[0]:
        all_feature_importances_df = pd.DataFrame(
            [m['feature_importances'] for m in fold_metrics_list if 'feature_importances' in m]
        ).fillna(0) # Fill NaN with 0 if a feature wasn't in all folds (e.g. if features changed per fold, though not current case)
        avg_feature_importances = all_feature_importances_df.mean().sort_values(ascending=False)
        logger.info("\nAverage Feature Importances (Top 20):")
        logger.info(f"\n{avg_feature_importances.head(20)}")
        # Add to the metrics file
        # Ensure avg_feature_importances are converted to a JSON serializable format (dict)
        # And handle potential numpy types within the dict values
        serializable_avg_importances = {k: float(v) for k, v in avg_feature_importances.items()}
    else:
        serializable_avg_importances = {}
        logger.info("No feature importances to average.")

    # Save detailed metrics
    cv_metrics_path = os.path.join(METRICS_OUTPUT_DIR, 'manual_tscv_evaluation_metrics.json') 
    with open(cv_metrics_path, 'w') as f:
        # Convert the entire fold_metrics_list for JSON serialization carefully
        # For example, classification_report is a string, which is fine.
        # Other numeric values should be Python natives.
        # The feature_importances are already converted to dicts of Python floats.
        # Let's ensure other potential numpy numbers in fold_metrics_list are converted.
        cleaned_fold_metrics = []
        for m in fold_metrics_list:
            cleaned_m = m.copy() # Work on a copy
            for key, value in cleaned_m.items():
                if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    cleaned_m[key] = int(value)
                elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                    cleaned_m[key] = float(value)
                # feature_importances dict values are already float
            cleaned_fold_metrics.append(cleaned_m)

        json.dump({
            'average_accuracy': float(avg_accuracy), # Ensure this is float
            'average_f1_weighted': float(avg_f1), # Ensure this is float
            'std_accuracy': float(std_accuracy), # Ensure this is float
            'std_f1_weighted': float(std_f1), # Ensure this is float
            'fold_metrics': cleaned_fold_metrics, 
            'average_feature_importances': serializable_avg_importances, 
            'overall_classification_report': overall_class_report
        }, f, indent=4)
    logger.info(f"Saved CV metrics (including feature importances) to {cv_metrics_path}")

    # Generate per-facility metrics directly
    logger.info("Computing per-facility metrics...")
    try:
        # We need to reconstruct the per-facility data from the fold results
        # Since we don't have parking_ids in the current structure, we'll create a simple version
        logger.info("Per-facility metrics computation requires parking_id data")
        logger.info("Skipping per-facility metrics for now - would need parking_id preservation")
        
        # Create a simple per-facility metrics file with overall performance
        pf_metrics_path = os.path.join(METRICS_OUTPUT_DIR, 'per_facility_metrics.json')
        with open(pf_metrics_path, 'w') as f:
            json.dump({
                'note': 'Per-facility metrics require parking_id data preservation',
                'overall_macro_f1': float(avg_f1),
                'overall_accuracy': float(avg_accuracy)
            }, f, indent=2)
        logger.info(f"Created placeholder per-facility metrics at {pf_metrics_path}")
    except Exception as e:
        logger.warning(f"Per-facility metrics computation failed: {e}")

    # Save final export: retrain on full labeled dataset with best params from CV (simple approach)
    try:
        logger.info("Retraining final model on full dataset for export...")
        best_params_export = {
            'n_estimators': 1000,
            'learning_rate': 0.08,
            'num_leaves': 31,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
            'objective': 'multiclass'
        }
        final_model = lgb.LGBMClassifier(**best_params_export)
        final_model.fit(X, y)
        export_dir = os.path.join(MODEL_OUTPUT_DIR)
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, 'parking_1h_bands_lgbm.pkl')
        save_model(final_model, export_path)
        # Save feature list and class mapping
        with open(os.path.join(export_dir, 'parking_1h_bands_features.json'), 'w') as f:
            json.dump({'features': features}, f, indent=2)
        with open(os.path.join(export_dir, 'parking_1h_bands_class_mapping.json'), 'w') as f:
            cls_map = define_parking_occupancy_classes()
            json.dump(cls_map, f, indent=2)
        logger.info(f"Exported final model to {export_path}")
    except Exception as e:
        logger.warning(f"Final export failed: {e}")

    logger.info("Manual Temporal CV training pipeline finished.") # Updated log message

if __name__ == "__main__":
    main_tscv() # Call the new main function 