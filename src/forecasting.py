"""
forecasting.py
Trains forecasting models on the engineered daily features and compares them
on a chronological train/test split (last N days held out as test set —
this is a time series, so it is never shuffled).

Models:
    1. Linear Regression       - simple baseline, interpretable coefficients
    2. Random Forest Regressor - non-linear, handles feature interactions
    3. XGBoost Regressor       - boosted trees, usually the strongest tabular performer
    4. LightGBM Regressor      - optional; skipped automatically if not installed

Run:
    python src/forecasting.py
Input:
    data/processed/features.csv
Output:
    models/best_model.pkl
    models/all_models.pkl
    models/model_comparison.csv
    models/best_model_name.json
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "units_sold"
TEST_DAYS = 60
FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "year", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_30", "rolling_std_30",
]
# Note: avg_order_value is a same-day derived KPI (revenue / invoices), so it
# is deliberately excluded from FEATURE_COLS — using it would leak same-day
# revenue information into a same-day units_sold prediction.


def load_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])


def chronological_split(df: pd.DataFrame, test_days=TEST_DAYS):
    """Time series split: train on everything before the last `test_days`, test on the rest.
    Never randomly shuffle time series data — that leaks future info into training."""
    test_days = min(test_days, max(len(df) // 5, 1))
    split_idx = len(df) - test_days
    return df.iloc[:split_idx], df.iloc[split_idx:]


def walk_forward_splits(df: pd.DataFrame, n_folds=5, min_train_days=90):
    """Expanding-window walk-forward validation — the same style of validation
    used in Kaggle's M5 (Walmart) and Rossmann Store Sales forecasting
    competitions, and how real demand-forecasting teams validate before
    trusting a model. Fold 1 trains on the earliest data and tests on the
    next chunk; fold 2 trains on everything up to fold 1's test end and
    tests on the following chunk; and so on. Unlike a single train/test
    split, this shows whether a model is consistently good across different
    time periods (e.g. does it still work across a holiday season?) rather
    than just lucky/unlucky on one particular 60-day window."""
    usable_days = len(df) - min_train_days
    fold_size = max(usable_days // n_folds, 1)
    n_folds = min(n_folds, max(usable_days // fold_size, 1))

    for fold in range(n_folds):
        train_end = min_train_days + fold * fold_size
        test_end = min(train_end + fold_size, len(df))
        if train_end >= test_end:
            break
        yield df.iloc[:train_end], df.iloc[train_end:test_end]


def evaluate(y_true, y_pred) -> dict:
    return {
        "R2": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
    }


def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, evaluate(y_test, preds), preds


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, evaluate(y_test, preds), preds


def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, evaluate(y_test, preds), preds


def train_lightgbm(X_train, y_train, X_test, y_test):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.03, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, evaluate(y_test, preds), preds


TRAINERS = {
    "Linear Regression": train_linear_regression,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
}


def run_walk_forward_validation(df: pd.DataFrame, model_name: str, n_folds=5) -> pd.DataFrame:
    """Re-trains `model_name` from scratch on each expanding-window fold and
    collects per-fold metrics. Run only on the best single-split model —
    training every model on every fold isn't necessary to make the point,
    and keeps this fast on a CPU-only machine."""
    trainer = TRAINERS[model_name]
    fold_metrics = []
    for fold_i, (train_df, test_df) in enumerate(walk_forward_splits(df, n_folds=n_folds), start=1):
        X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
        X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
        _, metrics, _ = trainer(X_train, y_train, X_test, y_test)
        metrics.update({"fold": fold_i, "train_days": len(train_df), "test_days": len(test_df)})
        fold_metrics.append(metrics)
    return pd.DataFrame(fold_metrics)


def run_all_models():
    logger.info("Loading features...")
    df = load_features()
    train_df, test_df = chronological_split(df)
    logger.info(f"  train: {len(train_df)} days, test: {len(test_df)} days")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    results, trained_models = {}, {}

    logger.info("Training Linear Regression...")
    model, metrics, _ = train_linear_regression(X_train, y_train, X_test, y_test)
    results["Linear Regression"], trained_models["Linear Regression"] = metrics, model

    logger.info("Training Random Forest...")
    model, metrics, _ = train_random_forest(X_train, y_train, X_test, y_test)
    results["Random Forest"], trained_models["Random Forest"] = metrics, model

    logger.info("Training XGBoost...")
    model, metrics, _ = train_xgboost(X_train, y_train, X_test, y_test)
    results["XGBoost"], trained_models["XGBoost"] = metrics, model

    logger.info("Training LightGBM (optional)...")
    try:
        model, metrics, _ = train_lightgbm(X_train, y_train, X_test, y_test)
        results["LightGBM"], trained_models["LightGBM"] = metrics, model
    except ImportError:
        logger.info("  lightgbm not installed; skipping (not required — 3 models is enough).")

    comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    comparison_df = comparison_df.sort_values("R2", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    logger.info("\nModel comparison (sorted by R2, best first):\n" + comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    logger.info(f"\nBest model: {best_model_name}")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(trained_models, MODELS_DIR / "all_models.pkl")

    with open(MODELS_DIR / "best_model_name.json", "w") as f:
        json.dump({"best_model": best_model_name, "feature_cols": FEATURE_COLS}, f)

    logger.info(f"Saved best model -> {MODELS_DIR / 'best_model.pkl'}")

    logger.info(
        f"\nRunning walk-forward cross-validation on {best_model_name} "
        f"(5 expanding-window folds — same validation style used in Kaggle's "
        f"M5/Rossmann retail forecasting competitions)..."
    )
    cv_df = run_walk_forward_validation(df, best_model_name)
    cv_df.to_csv(MODELS_DIR / "walk_forward_cv.csv", index=False)
    logger.info("\nWalk-forward CV results (per fold):\n" + cv_df.to_string(index=False))
    logger.info(
        f"\nMean R2 across folds: {cv_df['R2'].mean():.4f}  (std: {cv_df['R2'].std():.4f}) "
        f"— a small std means the model performs consistently across different "
        f"time periods, not just well on one lucky test window."
    )

    return comparison_df, trained_models


if __name__ == "__main__":
    run_all_models()
