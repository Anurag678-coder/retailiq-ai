"""
preprocessing.py
Cleans the raw Online Retail II transaction data.

Real transactional data is messy: cancelled orders, missing customer IDs
(guest/offline sales), negative or zero quantities, free/zero-priced rows
(damages, samples, adjustments), and duplicate rows. This module turns that
into a clean, analysis-ready transaction table.

Run:
    python src/preprocessing.py
Input:
    data/raw/online_retail.csv   (see src/download_data.py)
Output:
    data/processed/sales_clean.csv   (transaction-level, cleaned)
    data/processed/daily_totals.csv  (aggregated single daily time series)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE_CANDIDATES = ["online_retail.csv", "online_retail_II.csv", "Online Retail.csv"]

# Real-world exports use inconsistent column naming across UCI/Kaggle mirrors
# (e.g. "Invoice" vs "InvoiceNo", "Price" vs "UnitPrice", "Customer ID" vs
# "CustomerID"). Normalize everything to one canonical schema.
COLUMN_ALIASES = {
    "invoice": "InvoiceNo",
    "invoiceno": "InvoiceNo",
    "stockcode": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "price": "UnitPrice",
    "unitprice": "UnitPrice",
    "customerid": "CustomerID",
    "customer id": "CustomerID",
    "country": "Country",
}


def find_raw_file() -> Path:
    for name in RAW_FILE_CANDIDATES:
        p = RAW_DIR / name
        if p.exists():
            return p
    csvs = list(RAW_DIR.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        "No raw dataset found in data/raw/. Run `python src/download_data.py` first, "
        "or manually place the Online Retail II CSV at data/raw/online_retail.csv"
    )


def load_raw() -> pd.DataFrame:
    path = find_raw_file()
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
    df.columns = [COLUMN_ALIASES.get(c.strip().lower(), c.strip()) for c in df.columns]
    required = {"InvoiceNo", "StockCode", "Description", "Quantity",
                "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"  removed {before - len(df)} exact duplicate rows")
    return df


def remove_cancelled_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """Cancelled orders have an InvoiceNo starting with 'C'. They represent
    returns/reversals, not sales, so they're excluded from the sales pipeline."""
    before = len(df)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    is_cancelled = df["InvoiceNo"].str.startswith("C")
    df = df[~is_cancelled].copy()
    logger.info(f"  removed {is_cancelled.sum()} rows from cancelled invoices")
    return df


def remove_invalid_quantity(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["Quantity"] > 0].copy()
    logger.info(f"  removed {before - len(df)} rows with Quantity <= 0")
    return df


def remove_invalid_price(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["UnitPrice"] > 0].copy()
    logger.info(f"  removed {before - len(df)} rows with UnitPrice <= 0")
    return df


def remove_missing_description(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["Description"].notna()].copy()
    df["Description"] = df["Description"].str.strip()
    logger.info(f"  removed {before - len(df)} rows with missing Description")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # CustomerID is nullable: guest/offline sales legitimately have no ID.
    # Kept as Int64 (nullable int) rather than dropped here — segmentation.py
    # drops the nulls itself, since it specifically needs identified customers.
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def cap_outliers(df: pd.DataFrame, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Real retail data has genuine wholesale bulk orders mixed in with small
    quantity anomalies/typos. Winsorize (cap, don't drop) Quantity and
    UnitPrice at the 1st/99th percentile so a handful of extreme rows don't
    distort daily aggregates or model training."""
    q_before_max = df["Quantity"].max()
    p_before_max = df["UnitPrice"].max()

    q_lo, q_hi = df["Quantity"].quantile([lower_q, upper_q])
    p_lo, p_hi = df["UnitPrice"].quantile([lower_q, upper_q])

    df["Quantity"] = df["Quantity"].clip(q_lo, q_hi)
    df["UnitPrice"] = df["UnitPrice"].clip(p_lo, p_hi)

    logger.info(f"  capped Quantity: max {q_before_max:.0f} -> {df['Quantity'].max():.0f}")
    logger.info(f"  capped UnitPrice: max {p_before_max:.2f} -> {df['UnitPrice'].max():.2f}")
    return df


def add_total_price(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalPrice"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning sales data...")
    df = remove_duplicates(df)
    df = remove_missing_description(df)
    df = remove_cancelled_invoices(df)
    df = remove_invalid_quantity(df)
    df = remove_invalid_price(df)
    df = convert_dtypes(df)
    df = cap_outliers(df)
    df = add_total_price(df)
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    return df


def build_daily_totals(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to a single daily series — this is what forecasting trains on."""
    daily = (
        df_clean.assign(date=df_clean["InvoiceDate"].dt.date)
        .groupby("date", as_index=False)
        .agg(
            units_sold=("Quantity", "sum"),
            revenue=("TotalPrice", "sum"),
            num_invoices=("InvoiceNo", "nunique"),
            num_customers=("CustomerID", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


if __name__ == "__main__":
    raw = load_raw()
    logger.info(f"Loaded {len(raw):,} raw rows")

    clean = clean_sales_data(raw)
    clean.to_csv(PROCESSED_DIR / "sales_clean.csv", index=False)
    logger.info(f"Saved cleaned data: {len(clean):,} rows -> {PROCESSED_DIR / 'sales_clean.csv'}")

    daily = build_daily_totals(clean)
    daily.to_csv(PROCESSED_DIR / "daily_totals.csv", index=False)
    logger.info(f"Saved daily totals: {len(daily):,} rows -> {PROCESSED_DIR / 'daily_totals.csv'}")
