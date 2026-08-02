"""
explainability.py
Generates SHAP explanations for the tree-based forecasting model
(XGBoost, LightGBM, or Random Forest — whichever tree model is available;
SHAP's TreeExplainer needs a tree-based model, so Linear Regression is
never chosen here even if it happened to score best).

Produces:
    models/shap_summary.png       (global feature importance, all test rows)
    models/shap_waterfall.png     (single-prediction breakdown, first test row)
    models/shap_feature_importance.csv
    models/shap_values.pkl        (raw shap values, for the dashboard to reuse)

Run:
    python src/explainability.py
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TEST_DAYS = 60
TREE_MODEL_PRIORITY = ["XGBoost", "LightGBM", "Random Forest"]


def load_explainer_model():
    all_models = joblib.load(MODELS_DIR / "all_models.pkl")
    for name in TREE_MODEL_PRIORITY:
        if name in all_models:
            return name, all_models[name]
    raise ValueError("No tree-based model available for SHAP explanation")


def get_test_split():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    with open(MODELS_DIR / "best_model_name.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    test_days = min(TEST_DAYS, max(len(df) // 5, 1))
    split_idx = len(df) - test_days
    test_df = df.iloc[split_idx:]
    return test_df[feature_cols], test_df, feature_cols


def run_shap_analysis():
    model_name, model = load_explainer_model()
    logger.info(f"Explaining model: {model_name}")

    X_test, test_df, feature_cols = get_test_split()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Global summary plot: which features matter most overall ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {MODELS_DIR / 'shap_summary.png'}")

    # --- Waterfall plot: how features pushed ONE prediction up/down ---
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {MODELS_DIR / 'shap_waterfall.png'}")

    # --- Feature importance ranking table (mean |SHAP value|) ---
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(MODELS_DIR / "shap_feature_importance.csv", index=False)
    logger.info("\nTop 5 features by mean |SHAP value|:\n" + importance_df.head(5).to_string(index=False))

    joblib.dump(
        {"shap_values": shap_values, "X_test": X_test, "model_name": model_name},
        MODELS_DIR / "shap_values.pkl",
    )
    logger.info(f"\nSaved raw SHAP values -> {MODELS_DIR / 'shap_values.pkl'}")
    return importance_df


if __name__ == "__main__":
    run_shap_analysis()
