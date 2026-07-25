"""
explainability.py
Generates SHAP explanations for the tree-based models (Random Forest / XGBoost).
SHAP doesn't support Prophet directly, so if Prophet is the "best model" this
module explains the XGBoost model instead (kept trained as a reference/explainer model).

Produces:
- models/shap_summary.png     (global feature importance, all test rows)
- models/shap_waterfall.png   (single-prediction breakdown, first test row)
- models/shap_values.pkl      (raw shap values, for the dashboard to reuse)

Run:
    python src/explainability.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TARGET_COL = "units_sold"
TEST_DAYS = 60


def load_explainer_model():
    """SHAP TreeExplainer needs a tree model. Prefer XGBoost, fall back to Random Forest."""
    all_models = joblib.load(MODELS_DIR / "all_models.pkl")
    if "XGBoost" in all_models:
        return "XGBoost", all_models["XGBoost"]
    if "Random Forest" in all_models:
        return "Random Forest", all_models["Random Forest"]
    raise ValueError("No tree-based model available for SHAP explanation")


def get_test_split():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    with open(MODELS_DIR / "best_model_name.json") as f:
        import json
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    split_idx = len(df) - TEST_DAYS
    test_df = df.iloc[split_idx:]
    return test_df[feature_cols], test_df, feature_cols


def run_shap_analysis():
    model_name, model = load_explainer_model()
    print(f"Explaining model: {model_name}")

    X_test, test_df, feature_cols = get_test_split()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Global summary plot: which features matter most overall ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {MODELS_DIR / 'shap_summary.png'}")

    # --- Waterfall plot: how features pushed ONE prediction up/down ---
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {MODELS_DIR / 'shap_waterfall.png'}")

    # --- Feature importance ranking table (mean |SHAP value|) ---
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(MODELS_DIR / "shap_feature_importance.csv", index=False)
    print("\nTop 5 features by mean |SHAP value|:")
    print(importance_df.head(5).to_string(index=False))

    joblib.dump({"shap_values": shap_values, "X_test": X_test, "model_name": model_name},
                MODELS_DIR / "shap_values.pkl")
    print(f"\nSaved raw SHAP values -> {MODELS_DIR / 'shap_values.pkl'}")

    return importance_df


if __name__ == "__main__":
    run_shap_analysis()
