"""
forecasting.py
Trains 4 forecasting models on the engineered features and compares them
on a chronological train/test split (last 60 days held out as test set,
since this is a time series — never shuffle).

Models:
    1. Linear Regression       - simple baseline, interpretable coefficients
    2. Random Forest Regressor - non-linear, handles feature interactions
    3. XGBoost Regressor       - boosted trees, usually strongest tabular performer
    4. Prophet                 - purpose-built time-series model (trend+seasonality)

Run:
    python src/forecasting.py
Input:
    data/processed/features.csv   (for LR / RF / XGB)
    data/processed/daily_totals.csv (for Prophet, which wants raw ds/y)
Output:
    models/best_model.pkl
    models/model_comparison.csv
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

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


def load_features():
    return pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])


def chronological_split(df, test_days=TEST_DAYS):
    """Time series split: train on everything before the last `test_days`, test on the rest.
    NEVER randomly shuffle time series data — that would leak future info into training."""
    split_idx = len(df) - test_days
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    return train, test


def evaluate(y_true, y_pred):
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
    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, evaluate(y_test, preds), preds


def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, evaluate(y_test, preds), preds


def train_prophet(daily_df, test_days=TEST_DAYS):
    from prophet import Prophet

    prophet_df = daily_df[["date", TARGET_COL]].rename(columns={"date": "ds", TARGET_COL: "y"})
    train = prophet_df.iloc[: len(prophet_df) - test_days]
    test = prophet_df.iloc[len(prophet_df) - test_days :]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=test_days)
    forecast = model.predict(future)
    preds = forecast.tail(test_days)["yhat"].values

    return model, evaluate(test["y"].values, preds), preds


def run_all_models():
    print("Loading features...")
    df = load_features()
    train_df, test_df = chronological_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    results = {}
    trained_models = {}

    print("Training Linear Regression...")
    lr_model, lr_metrics, _ = train_linear_regression(X_train, y_train, X_test, y_test)
    results["Linear Regression"] = lr_metrics
    trained_models["Linear Regression"] = lr_model

    print("Training Random Forest...")
    rf_model, rf_metrics, _ = train_random_forest(X_train, y_train, X_test, y_test)
    results["Random Forest"] = rf_metrics
    trained_models["Random Forest"] = rf_model

    print("Training XGBoost...")
    xgb_model, xgb_metrics, _ = train_xgboost(X_train, y_train, X_test, y_test)
    results["XGBoost"] = xgb_metrics
    trained_models["XGBoost"] = xgb_model

    print("Training Prophet...")
    try:
        daily_df = pd.read_csv(PROCESSED_DIR / "daily_totals.csv", parse_dates=["date"])
        prophet_model, prophet_metrics, _ = train_prophet(daily_df)
        results["Prophet"] = prophet_metrics
        trained_models["Prophet"] = prophet_model
    except Exception as e:
        print(f"  Prophet failed ({e}); skipping — comparison will proceed with the other 3 models.")

    comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    comparison_df = comparison_df.sort_values("R2", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison (sorted by R2, best first):")
    print(comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # Prophet models aren't sklearn-API and are saved with their own method via joblib too (works fine)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(trained_models, MODELS_DIR / "all_models.pkl")

    with open(MODELS_DIR / "best_model_name.json", "w") as f:
        json.dump({"best_model": best_model_name, "feature_cols": FEATURE_COLS}, f)

    print(f"Saved best model -> {MODELS_DIR / 'best_model.pkl'}")
    return comparison_df, trained_models


if __name__ == "__main__":
    run_all_models()
