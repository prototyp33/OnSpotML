import os
from pathlib import Path
import yaml
import json
import mlflow
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- USER CONFIG ----------------
RUN_ID = "ed2cdfa9b15f4a2c9185fa1b5cca84de"
CFG_PATH = Path("config/training_config.yaml")
SAMPLE_ROWS = 10000  # subsample for shap to keep runtime reasonable
# ---------------------------------------------

OUTPUT_DIR = Path("reports/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    if RUN_ID.startswith("<"):
        raise ValueError("Please set RUN_ID to your MLflow run id before running this script.")

    cfg = load_config(CFG_PATH)
    features_path = Path(cfg["data"]["features_path"])

    logging.info(f"Loading feature table from {features_path}")
    df = pd.read_parquet(features_path)

    # build feature list from config
    feature_cols = []
    for key, lst in cfg["features"].items():
        if key != "potential_leakage" and isinstance(lst, list):
            feature_cols.extend(lst)

    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(df) > SAMPLE_ROWS:
        df = df.sample(SAMPLE_ROWS, random_state=0)
    X = df[feature_cols]

    model_uri = f"runs:/{RUN_ID}/best-imbalance-model"
    logging.info(f"Loading model from {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    clf = model._model_impl  # underlying estimator supporting predict_proba

    logging.info("Computing SHAP values (TreeExplainer)…")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    class_labels = list(clf.classes_)
    if 3 in class_labels:
        cls_idx = class_labels.index(3)
        plt.title("SHAP summary – class 3")
        shap.summary_plot(shap_values[cls_idx], X, show=False, max_display=25, plot_size=(12, 8))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "summary_class3.png", dpi=300)
        plt.close()
        logging.info("Saved summary_class3.png")

    # Global bar chart across all classes
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=25, plot_size=(12, 6))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_bar.png", dpi=300)
    plt.close()
    logging.info("Saved global_bar.png")

    # save shap values array metadata for further inspection
    meta_path = OUTPUT_DIR / "shap_metadata.json"
    meta = {"num_rows": len(X), "num_features": len(feature_cols), "classes": class_labels}
    meta_path.write_text(json.dumps(meta, indent=2))

    logging.info("SHAP analysis complete. Files written to reports/shap/.")


if __name__ == "__main__":
    main() 