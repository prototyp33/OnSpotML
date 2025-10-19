import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import yaml
from pathlib import Path
import mlflow
import mlflow.lightgbm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config/training_config.yaml'):
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
            
def select_features(config: dict) -> list:
    """Combines feature lists from the config."""
    features = []
    for key, f_list in config['features'].items():
        if key != 'potential_leakage':
             if isinstance(f_list, list):
                features.extend(f_list)
    return features

# --- Utility: log confusion matrix to MLflow ---
def _log_confusion_matrix(y_true, y_pred, class_names, run_id):
    """Create confusion matrix heatmap and log as MLflow artifact."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plot_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, "plots")
    os.remove(plot_path)
    logging.info("Confusion matrix plotted and logged.")

def train_baseline_model():
    """
    Main function to load data, train, and evaluate the baseline model,
    now with MLflow experiment tracking.
    """
    try:
        config = load_config()
        cfg_data = config['data']
        cfg_model = config['model']
        
        # --- MLflow Setup ---
        # Set an experiment name. If it doesn't exist, MLflow creates it.
        mlflow.set_experiment("OnSpotML Baseline Training")
        # Start an MLflow run. All subsequent logging will be part of this run.
        with mlflow.start_run() as run:
            logging.info(f"MLflow run started. Run ID: {run.info.run_id}")
            
            # --- Log Parameters ---
            # Log key configuration parameters to make runs reproducible.
            mlflow.log_params(cfg_data)
            mlflow.log_params(cfg_model['params'])
            mlflow.log_param("n_splits", cfg_model['n_splits'])

            # 1. Load Data
            features_path = Path(cfg_data['features_path'])
            logging.info(f"Loading pre-computed features from {features_path}...")
            df = pd.read_parquet(features_path)
            
            # (The rest of the data prep remains the same)
            target_col = cfg_data['target_column']
            timestamp_col = cfg_data['timestamp_column']
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)
            features_to_use = select_features(config)
            cols_to_drop = config['features'].get('potential_leakage', [])
            df = df.drop(columns=cols_to_drop, errors='ignore')
            df = df.dropna(subset=features_to_use + [target_col])
            
            X = df[features_to_use]
            y = df[target_col]
            class_names = sorted(y.unique())

            # 2. Train and Evaluate
            tscv = TimeSeriesSplit(n_splits=cfg_model['n_splits'])
            all_preds, all_y_test = [], []
            
            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                # ... (training logic remains the same) ...
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                params = cfg_model['params']
                params['num_class'] = len(class_names)
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                # --- Log Metric per Fold ---
                mlflow.log_metric(f"fold_{fold+1}_accuracy", acc)
                all_preds.extend(preds)
                all_y_test.extend(y_test)

            # 3. Log Overall Results and Artifacts
            if all_y_test:
                overall_accuracy = accuracy_score(all_y_test, all_preds)
                weighted_f1 = f1_score(all_y_test, all_preds, average='weighted')
                
                logging.info(f"Overall Test Accuracy: {overall_accuracy:.4f}")
                logging.info(f"Overall Weighted F1-Score: {weighted_f1:.4f}")

                # --- Log Final Metrics ---
                mlflow.log_metric("overall_accuracy", overall_accuracy)
                mlflow.log_metric("weighted_f1_score", weighted_f1)
                
                # --- Per-class metrics ---
                prec, rec, f1, _ = precision_recall_fscore_support(
                    all_y_test, all_preds, labels=class_names, zero_division=0
                )
                for idx, cls in enumerate(class_names):
                    mlflow.log_metric(f"class_{cls}_precision", prec[idx])
                    mlflow.log_metric(f"class_{cls}_recall", rec[idx])
                    mlflow.log_metric(f"class_{cls}_f1", f1[idx])

                # Confusion matrix artifact
                _log_confusion_matrix(all_y_test, all_preds, class_names, run.info.run_id)

                # --- Log Model Artifact ---
                # Convert integer columns to float64 to avoid MLflow warnings
                input_example = X_train.head().copy()
                for col in input_example.select_dtypes(include=['int']).columns:
                    input_example[col] = input_example[col].astype('float64')
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path="model",
                    input_example=input_example
                )

                # --- Log Plot Artifact ---
                importance_df = pd.DataFrame({
                    'feature': model.booster_.feature_name(),
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=importance_df.head(20))
                plt.title('LightGBM Feature Importance (Last Fold)')
                plt.tight_layout()
                
                # Save plot to a temporary file to be logged as an artifact
                plot_path = "feature_importance.png"
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path, "plots") # Log the plot to a "plots" folder
                plt.close()
                os.remove(plot_path) # Clean up the temp file

    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    train_baseline_model()