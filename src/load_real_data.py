"""
load_real_data.py
Transforms the real UCI "Online Retail" dataset (data/external/online_retail.csv)
into the three raw files the rest of the RetailIQ pipeline expects:

    data/raw/sales_data.csv          (date, store, category, units_sold, unit_price, revenue, is_promo)
    data/raw/customer_transactions.csv (customer_id, transaction_date, amount)
    data/raw/basket_transactions.csv   (basket_id, items)

Source: UCI Machine Learning Repository — "Online Retail" dataset
Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33
Real transactions from a UK-based online gift retailer, Dec 2010 - Dec 2011.
License: CC BY 4.0.

Run:
    python src/load_real_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = BASE_DIR / "data" / "external"
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_COUNTRIES = 6   # everything else bucketed into "Other"
TOP_N_PRODUCTS_FOR_BASKETS = 60  # keeps Apriori's one-hot matrix tractable


# ---------------------------------------------------------------------
# Category bucketing — the real dataset has no category column, only a
# free-text Description per product. We bucket by keyword so the rest of
# the pipeline (which groups by category) still works meaningfully.
# ---------------------------------------------------------------------
CATEGORY_RULES = [
    ("Christmas & Seasonal", ["CHRISTMAS", "XMAS", "ADVENT", "EASTER", "HALLOWEEN"]),
    ("Home Decor & Lighting", ["LIGHT", "LANTERN", "CANDLE", "HOLDER", "DECORATION", "ORNAMENT", "MIRROR", "CLOCK"]),
    ("Kitchen & Dining", ["MUG", "CAKE", "TIN", "JAR", "PLATE", "CUTLERY", "BOWL", "TEA", "KITCHEN", "BAKING"]),
    ("Bags & Storage", ["BAG", "BOX", "BASKET", "STORAGE"]),
    ("Cards & Stationery", ["CARD", "NOTEBOOK", "PAPER", "GIFT WRAP", "RIBBON"]),
    ("Toys & Games", ["TOY", "GAME", "DOLL", "PUZZLE", "SKITTLE"]),
    ("Bath & Garden", ["BATH", "GARDEN", "SPONGE", "SOAP"]),
]


def categorize(description: str) -> str:
    if not isinstance(description, str):
        return "Other"
    text = description.upper()
    for category, keywords in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category
    return "Other"


def load_source():
    path = EXTERNAL_DIR / "online_retail.csv"
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


# ---------------------------------------------------------------------
# 1. sales_data.csv
# ---------------------------------------------------------------------
def build_sales_data(df):
    top_countries = df["Country"].value_counts().head(TOP_N_COUNTRIES).index.tolist()
    df = df.copy()
    df["store"] = df["Country"].where(df["Country"].isin(top_countries), "Other")

    # Map category off the unique descriptions first (much faster than row-wise apply)
    unique_desc = df["Description"].dropna().unique()
    cat_map = {d: categorize(d) for d in unique_desc}
    df["category"] = df["Description"].map(cat_map).fillna("Other")

    df["date"] = df["InvoiceDate"].dt.date

    grouped = (
        df.groupby(["date", "store", "category"])
        .agg(units_sold=("Quantity", "sum"), revenue=("Revenue", "sum"))
        .reset_index()
    )
    grouped["unit_price"] = (grouped["revenue"] / grouped["units_sold"].replace(0, np.nan)).round(2)
    grouped["unit_price"] = grouped["unit_price"].fillna(0)
    grouped["revenue"] = grouped["revenue"].round(2)
    grouped["is_promo"] = 0  # not available in the source data
    grouped["date"] = pd.to_datetime(grouped["date"])

    cols = ["date", "store", "category", "units_sold", "unit_price", "revenue", "is_promo"]
    return grouped[cols].sort_values(["date", "store", "category"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 2. customer_transactions.csv — one row per invoice (a real "transaction")
# ---------------------------------------------------------------------
def build_customer_transactions(df):
    per_invoice = (
        df.groupby(["CustomerID", "InvoiceNo"])
        .agg(transaction_date=("InvoiceDate", "min"), amount=("Revenue", "sum"))
        .reset_index()
    )
    per_invoice = per_invoice.rename(columns={"CustomerID": "customer_id"})
    per_invoice["customer_id"] = "CUST_" + per_invoice["customer_id"].astype(int).astype(str)
    per_invoice["amount"] = per_invoice["amount"].round(2)
    per_invoice = per_invoice[per_invoice["amount"] > 0]  # drop zero/negative-value invoices
    return per_invoice[["customer_id", "transaction_date", "amount"]].sort_values("transaction_date")


# ---------------------------------------------------------------------
# 3. basket_transactions.csv — for Apriori, restricted to top-N products
#    so the one-hot encoded matrix stays a manageable size.
# ---------------------------------------------------------------------
def build_basket_transactions(df):
    top_products = df["Description"].value_counts().head(TOP_N_PRODUCTS_FOR_BASKETS).index.tolist()
    filtered = df[df["Description"].isin(top_products)]

    baskets = (
        filtered.groupby("InvoiceNo")["Description"]
        .apply(lambda items: sorted(set(items)))
        .reset_index()
    )
    baskets = baskets[baskets["Description"].apply(len) >= 2]  # need 2+ items for association rules
    baskets["items"] = baskets["Description"].apply(lambda lst: ",".join(lst))
    baskets = baskets.rename(columns={"InvoiceNo": "basket_id"})
    return baskets[["basket_id", "items"]]


if __name__ == "__main__":
    print("Loading real UCI Online Retail dataset...")
    df = load_source()
    print(f"  {len(df)} raw transaction lines, {df['InvoiceDate'].dt.date.nunique()} unique days, "
          f"{df['CustomerID'].nunique()} customers, {df['InvoiceNo'].nunique()} invoices")

    print("Building sales_data.csv (date x store x category)...")
    sales = build_sales_data(df)
    sales.to_csv(RAW_DIR / "sales_data.csv", index=False)
    print(f"  -> {len(sales)} rows, stores={sales['store'].unique().tolist()}")
    print(f"  -> categories={sales['category'].unique().tolist()}")

    print("Building customer_transactions.csv (one row per invoice)...")
    cust = build_customer_transactions(df)
    cust.to_csv(RAW_DIR / "customer_transactions.csv", index=False)
    print(f"  -> {len(cust)} rows, {cust['customer_id'].nunique()} unique customers")

    print("Building basket_transactions.csv (top products, for Apriori)...")
    baskets = build_basket_transactions(df)
    baskets.to_csv(RAW_DIR / "basket_transactions.csv", index=False)
    print(f"  -> {len(baskets)} baskets with 2+ items from the top {TOP_N_PRODUCTS_FOR_BASKETS} products")

    print("\nDone. Real data written to data/raw/. Now re-run the rest of the pipeline:")
    print("  python src/preprocessing.py")
    print("  python src/feature_engineering.py")
    print("  python src/forecasting.py")
    print("  python src/explainability.py")
    print("  python src/segmentation.py")
    print("  python src/recommendation.py")
