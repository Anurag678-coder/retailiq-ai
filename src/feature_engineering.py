"""
feature_engineering.py
Builds two feature sets from the cleaned transaction data:

1. Daily time-series features (for forecasting):
   lag features, rolling window stats, calendar features, average order value.
2. Customer-level features (for segmentation):
   Recency, Frequency, Monetary, simple Customer Lifetime Value,
   purchase frequency (orders/month), and average basket size.

Run:
    python src/feature_engineering.py
Input:
    data/processed/daily_totals.csv
    data/processed/sales_clean.csv
Output:
    data/processed/features.csv
    data/processed/customer_features.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 30]
TARGET_COL = "units_sold"


# ---------------------------------------------------------------------------
# Daily / time-series features
# ---------------------------------------------------------------------------
def load_daily() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "daily_totals.csv", parse_dates=["date"])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Average order value — a standard retail KPI, cheap to compute and
    useful both as a model feature and as a dashboard metric."""
    df["avg_order_value"] = (df["revenue"] / df["num_invoices"].replace(0, np.nan)).round(2)
    df["avg_order_value"] = df["avg_order_value"].fillna(0)
    return df


def add_lag_features(df: pd.DataFrame, target_col=TARGET_COL, lags=LAGS) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col=TARGET_COL, windows=ROLLING_WINDOWS) -> pd.DataFrame:
    for w in windows:
        # shift(1) first so the window never includes the current day (avoids leakage)
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w).std()
    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_business_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"  dropped {n_before - len(df)} rows with NaN from lag/rolling warmup period")
    return df


# ---------------------------------------------------------------------------
# Customer-level features (RFM + CLV + basket size)
# ---------------------------------------------------------------------------
def load_clean_transactions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "sales_clean.csv", parse_dates=["InvoiceDate"])


def build_customer_features(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """One row per identified customer (guest/missing-CustomerID rows are
    excluded here — segmentation and CLV require knowing who the customer is).
    Some dataset mirrors encode a missing CustomerID as the literal value 0
    instead of NaN — excluded too, or it shows up as one fake "customer" with
    an impossibly high order count that distorts the top segment."""
    df = df[df["CustomerID"].notna() & (df["CustomerID"] != 0)].copy()

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    per_invoice = df.groupby(["CustomerID", "InvoiceNo"]).agg(
        invoice_date=("InvoiceDate", "min"),
        invoice_value=("TotalPrice", "sum"),
        basket_size=("StockCode", "nunique"),
    ).reset_index()

    customer_span_days = (
        df.groupby("CustomerID")["InvoiceDate"].agg(lambda s: (s.max() - s.min()).days)
    )

    features = per_invoice.groupby("CustomerID").agg(
        recency=("invoice_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("invoice_value", "sum"),
        avg_basket_size=("basket_size", "mean"),
    ).reset_index()

    features["avg_basket_size"] = features["avg_basket_size"].round(1)
    features["monetary"] = features["monetary"].round(2)

    # Simple CLV: average order value x purchase frequency x active lifespan (months, min 1).
    # This is a historical/observed CLV, not a predictive one — good enough for
    # a portfolio project and easy to explain in a viva without a survival model.
    active_months = (customer_span_days.reindex(features["CustomerID"]).values / 30.0)
    active_months = np.clip(active_months, 1, None)
    features["active_months"] = np.round(active_months, 1)
    features["purchase_frequency_per_month"] = (
        features["frequency"] / features["active_months"]
    ).round(2)
    avg_order_value = (features["monetary"] / features["frequency"]).round(2)
    features["clv_simple"] = (avg_order_value * features["purchase_frequency_per_month"] * features["active_months"]).round(2)

    return features


if __name__ == "__main__":
    daily = load_daily()
    logger.info(f"Loaded {len(daily)} daily rows")

    logger.info("Building daily time-series features...")
    daily_features = build_daily_features(daily)
    daily_features.to_csv(PROCESSED_DIR / "features.csv", index=False)
    logger.info(
        f"Saved features: {daily_features.shape[0]} rows x {daily_features.shape[1]} cols "
        f"-> {PROCESSED_DIR / 'features.csv'}"
    )

    logger.info("Building customer-level features (RFM + CLV + basket size)...")
    transactions = load_clean_transactions()
    customer_features = build_customer_features(transactions)
    customer_features.to_csv(PROCESSED_DIR / "customer_features.csv", index=False)
    logger.info(
        f"Saved customer features: {len(customer_features):,} customers "
        f"-> {PROCESSED_DIR / 'customer_features.csv'}"
    )
