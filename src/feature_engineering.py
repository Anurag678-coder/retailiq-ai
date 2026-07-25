"""
feature_engineering.py
Builds model-ready features on the daily aggregated sales series:
- lag features (sales N days ago)
- rolling window stats (mean / std over trailing windows)
- calendar features (day of week, month, is_weekend, is_month_start/end)

Run:
    python src/feature_engineering.py
Input:
    data/processed/daily_totals.csv
Output:
    data/processed/features.csv
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 30]
TARGET_COL = "units_sold"


def load_daily():
    return pd.read_csv(PROCESSED_DIR / "daily_totals.csv", parse_dates=["date"])


def add_calendar_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek       # 0=Mon
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(df, target_col=TARGET_COL, lags=LAGS):
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col=TARGET_COL, windows=ROLLING_WINDOWS):
    for w in windows:
        # shift(1) first so the window never includes the current day (avoids leakage)
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w).std()
    return df


def build_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  dropped {n_before - len(df)} rows with NaN from lag/rolling windows (start-of-series warmup)")
    return df


if __name__ == "__main__":
    daily = load_daily()
    print(f"Loaded {len(daily)} daily rows")

    print("Building features...")
    features = build_features(daily)

    features.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(f"Saved features: {features.shape[0]} rows x {features.shape[1]} cols "
          f"-> {PROCESSED_DIR / 'features.csv'}")
    print("Columns:", list(features.columns))
