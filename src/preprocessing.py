"""
preprocessing.py
Cleans the raw synthetic sales data:
- removes exact duplicate rows
- imputes missing units_sold via category-level median
- caps extreme outliers (winsorization on units_sold per category)
- recomputes revenue after cleaning
- aggregates to a single daily time series (all stores/categories combined)
  AND keeps a store/category-level cleaned table for drill-down use.

Run:
    python src/preprocessing.py
Input:
    data/raw/sales_data.csv
Output:
    data/processed/sales_clean.csv        (store/category level, cleaned)
    data/processed/daily_totals.csv       (aggregated single time series)
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    df = pd.read_csv(RAW_DIR / "sales_data.csv", parse_dates=["date"])
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"  removed {before - len(df)} exact duplicate rows")
    return df


def impute_missing(df):
    missing_before = df["units_sold"].isna().sum()
    df["units_sold"] = df.groupby("category")["units_sold"].transform(
        lambda s: s.fillna(s.median())
    )
    print(f"  imputed {missing_before} missing units_sold values (category median)")
    return df


def cap_outliers(df, lower_q=0.01, upper_q=0.99):
    def cap_group(s):
        lo, hi = s.quantile(lower_q), s.quantile(upper_q)
        return s.clip(lo, hi)

    before_max = df["units_sold"].max()
    df["units_sold"] = df.groupby("category")["units_sold"].transform(cap_group)
    df["units_sold"] = df["units_sold"].round().astype(int)
    print(f"  capped outliers: max units_sold {before_max:.0f} -> {df['units_sold'].max()}")
    return df


def recompute_revenue(df):
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
    return df


def clean_sales_data(df):
    print("Cleaning sales data...")
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = cap_outliers(df)
    df = recompute_revenue(df)
    df = df.sort_values(["date", "store", "category"]).reset_index(drop=True)
    return df


def build_daily_totals(df_clean):
    """Aggregate to a single daily series — this is what forecasting models train on."""
    daily = (
        df_clean.groupby("date", as_index=False)
        .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


if __name__ == "__main__":
    raw = load_raw()
    print(f"Loaded {len(raw)} raw rows")

    clean = clean_sales_data(raw)
    clean.to_csv(PROCESSED_DIR / "sales_clean.csv", index=False)
    print(f"Saved cleaned data: {len(clean)} rows -> {PROCESSED_DIR / 'sales_clean.csv'}")

    daily = build_daily_totals(clean)
    daily.to_csv(PROCESSED_DIR / "daily_totals.csv", index=False)
    print(f"Saved daily totals: {len(daily)} rows -> {PROCESSED_DIR / 'daily_totals.csv'}")
