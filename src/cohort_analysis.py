"""
cohort_analysis.py
Monthly cohort retention analysis — a standard, easy-to-explain technique
used across almost every real retail analytics project. Answers: "of the
customers who made their first purchase in month X, how many came back in
each following month?"

No new libraries needed — just pandas groupby/pivot, so it's simple to
understand and explain in a viva, while still being a feature most beginner
projects skip (which is exactly why it stands out on a resume).

Run:
    python src/cohort_analysis.py
Input:
    data/processed/sales_clean.csv
Output:
    data/processed/cohort_retention.csv   (cohort month x period number -> % retained)
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_clean_transactions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "sales_clean.csv", parse_dates=["InvoiceDate"])


def build_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["CustomerID"].notna()].copy()

    # Each customer's "cohort" = the month of their very first purchase.
    df["order_month"] = df["InvoiceDate"].dt.to_period("M")
    df["cohort_month"] = df.groupby("CustomerID")["order_month"].transform("min")

    # How many months after their first purchase did this order happen?
    df["period_number"] = (
        (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
        + (df["order_month"].dt.month - df["cohort_month"].dt.month)
    )

    cohort_data = (
        df.groupby(["cohort_month", "period_number"])["CustomerID"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period_number", values="CustomerID")
    cohort_sizes = cohort_pivot.iloc[:, 0]  # period 0 = everyone in that cohort

    retention = cohort_pivot.divide(cohort_sizes, axis=0).round(4) * 100
    retention.index = retention.index.astype(str)
    retention = retention.reset_index().rename(columns={"cohort_month": "cohort_month"})

    return retention


if __name__ == "__main__":
    logger.info("Loading cleaned transactions...")
    df = load_clean_transactions()

    logger.info("Building monthly cohort retention table...")
    retention = build_cohort_retention(df)
    retention.to_csv(PROCESSED_DIR / "cohort_retention.csv", index=False)

    logger.info(f"  {len(retention)} cohorts")
    logger.info(f"Saved -> {PROCESSED_DIR / 'cohort_retention.csv'}")
