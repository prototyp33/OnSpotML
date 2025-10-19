import pandas as pd
import logging
import yaml
from pathlib import Path
import mlflow
import json
import numpy as np
import argparse
import os

# Ensure imports resolve when script is executed from project root
# (Path already imported above)
import sys
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from advanced_class_imbalance_handler import AdvancedClassImbalanceHandler  # type: ignore
import advanced_class_imbalance_handler as acih

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str = 'config/training_config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_features(config: dict) -> list:
    features = []
    for key, f_list in config['features'].items():
        if key != 'potential_leakage' and isinstance(f_list, list):
            features.extend(f_list)
    return features


# Helper to convert any NumPy types (scalars, arrays) into Python built-ins so that
# they can be safely serialized by ``yaml.safe_dump``.

def _make_yaml_serializable(obj):
    """Recursively convert NumPy objects → native Python types (int, float, list)."""

    if isinstance(obj, dict):
        return {str(_make_yaml_serializable(k)): _make_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [ _make_yaml_serializable(v) for v in obj ]

    # NumPy scalar types → Python scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # NumPy arrays → list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def run_imbalance_experiment() -> None:
    """Run comprehensive imbalance evaluation and log everything to MLflow."""

    parser = argparse.ArgumentParser(description="Run imbalance experiment")
    parser.add_argument("--fast", action="store_true", help="Run quick debug mode with subsampling and minimal config")
    args = parser.parse_args()

    cfg = load_config()
    cfg_data = cfg['data']

    mlflow.set_experiment("Advanced Imbalance Handling")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow run started (imbalance exp). Run ID: {run_id}")

        # --- Load dataset ---
        features_path = Path(cfg_data['features_path'])
        logging.info(f"Reading features table → {features_path}")
        df = pd.read_parquet(features_path)

        if args.fast:
            logging.info("FAST DEBUG MODE enabled – subsampling dataset for quicker runtime …")
            df = df.head(200_000)  # use first 200k rows (~ <10%)
            os.environ["FAST_DEBUG"] = "1"

        target_col = cfg_data['target_column']
        feature_cols = select_features(cfg)
        df = df.dropna(subset=feature_cols + [target_col])

        X = df[feature_cols]
        y = df[target_col]
        logging.info(f"Prepared data: X shape {X.shape}, class distribution:\n{y.value_counts(normalize=True)}")

        # --- Comprehensive evaluation ---
        if args.fast:
            # Reduce folds and restrict strategies
            acih.KEEP_SAMPLERS = {"adasyn", "smote_tomek"}
            acih.KEEP_MODELS = {"lgb_cost_sensitive"}
            globals()['KEEP_SAMPLERS'] = acih.KEEP_SAMPLERS
            globals()['KEEP_MODELS'] = acih.KEEP_MODELS
            n_splits = 2  # minimum allowed by TimeSeriesSplit
        else:
            n_splits = 2

        handler = AdvancedClassImbalanceHandler(random_state=42)
        eval_results = handler.run_comprehensive_evaluation(X, y, n_splits=n_splits)

        # --- Persist raw results (ensure YAML-serializable) ---
        serializable_results = _make_yaml_serializable(eval_results)
        results_path = Path("imbalance_results.yaml")
        results_path.write_text(yaml.safe_dump(serializable_results, sort_keys=False))
        mlflow.log_artifact(str(results_path), artifact_path="reports")
        results_path.unlink(missing_ok=True)

        # --- Log best strategy & metrics ---
        best_strategy = eval_results.get('best_strategy')
        best_perf = eval_results.get('best_performance', {})
        if best_strategy:
            mlflow.log_param("best_strategy", best_strategy)
        if best_perf:
            # Flatten / filter only scalar metrics (loggable by MLflow)
            scalar_metrics = {}
            for k, v in best_perf.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    scalar_metrics[k] = float(v)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float, np.integer, np.floating)):
                            scalar_metrics[f"{k}_{sub_k}"] = float(sub_v)
            if scalar_metrics:
                mlflow.log_metrics(scalar_metrics)

        # --- Human-readable report ---
        report_txt = handler.generate_improvement_report()
        mlflow.log_text(report_txt, "reports/improvement_report.txt")

        # --------------------------------------------------
        # 5.  Train the best strategy on the **entire** data
        #     and log the fitted model so SHAP can consume it
        # --------------------------------------------------
        best_strategy_name = eval_results.get("best_strategy")
        if best_strategy_name is None:
            logging.warning("No best strategy found – skipping model logging.")
        else:
            logging.info(f"Refitting best strategy '{best_strategy_name}' on full dataset to log model…")

            try:
                sampling_name, model_name = best_strategy_name.rsplit("_", 1)

                # Re-create sampling & model objects
                # (reuse handler helpers)
                resampled_map = handler.apply_advanced_smote_variants(X, y, allowed={sampling_name})
                if sampling_name not in resampled_map:
                    raise ValueError(f"Sampling strategy '{sampling_name}' not found when recreating.")

                X_res, y_res = resampled_map[sampling_name]

                class_weights = handler.calculate_dynamic_class_weights(y)
                models_map = handler.create_cost_sensitive_models(class_weights)
                if model_name not in models_map:
                    raise ValueError(f"Model '{model_name}' not found when recreating.")

                final_model = models_map[model_name]
                if args.fast and hasattr(final_model, 'set_params'):
                    final_model.set_params(n_estimators=50)

                sample_weight_final = np.array([class_weights.get(lbl, 1.0) for lbl in y_res])
                try:
                    final_model.fit(X_res, y_res, sample_weight=sample_weight_final)
                except TypeError:
                    final_model.fit(X_res, y_res)

                mlflow.sklearn.log_model(
                    sk_model=final_model,
                    artifact_path="best-imbalance-model",
                    input_example=X.head()
                )
                logging.info("Best model logged to MLflow (artifact path 'best-imbalance-model').")
            except Exception as ml_e:
                logging.error(f"Failed to log best model: {ml_e}")

        logging.info("Imbalance experiment completed and logged to MLflow.")


if __name__ == "__main__":
    run_imbalance_experiment() 