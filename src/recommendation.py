"""
recommendation.py
Simple, explainable product recommendations built from the cleaned
transaction data. No LLM, no vector database, no RAG — just:

    1. Frequently Bought Together — Apriori association rules on real baskets
       (grouped by InvoiceNo).
    2. Popular Products — highest total quantity sold.
    3. Top Revenue Products — highest total revenue.
    4. Top Products by Segment — best sellers within each customer segment
       (requires segmentation.py to have run first; skipped otherwise).

Run:
    python src/recommendation.py
Input:
    data/processed/sales_clean.csv
    data/processed/customer_segments.csv   (optional, for segment breakdown)
Output:
    data/processed/association_rules.csv
    data/processed/popular_products.csv
    data/processed/top_revenue_products.csv
    data/processed/segment_top_products.csv   (if customer_segments.csv exists)
"""

import logging
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.2
TOP_N = 20

# Real data has 4,000+ unique products. Apriori's one-hot "basket matrix" is
# (invoices x products) — running it on every product is slow and memory-
# heavy on a CPU-only laptop with no real payoff (long-tail products barely
# co-occur often enough to form meaningful rules anyway). Restricting to the
# top N best-selling products keeps it fast while still surfacing the
# business-relevant "frequently bought together" patterns.
TOP_N_PRODUCTS_FOR_APRIORI = 200


def load_clean_transactions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "sales_clean.csv", parse_dates=["InvoiceDate"])


def get_top_products(df: pd.DataFrame, n=TOP_N_PRODUCTS_FOR_APRIORI) -> set:
    return set(
        df.groupby("Description")["Quantity"].sum()
        .sort_values(ascending=False).head(n).index
    )


def build_baskets(df: pd.DataFrame, top_products: set = None) -> list:
    """One basket per invoice = the set of distinct products bought together.
    If `top_products` is given, baskets are restricted to those products first
    (keeps Apriori tractable on CPU — see TOP_N_PRODUCTS_FOR_APRIORI above)."""
    if top_products is not None:
        df = df[df["Description"].isin(top_products)]
    baskets = df.groupby("InvoiceNo")["Description"].apply(lambda s: list(set(s))).tolist()
    return [b for b in baskets if len(b) > 1]  # single-item baskets can't form a rule


def run_apriori(transactions: list, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE) -> pd.DataFrame:
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found at this min_support — try lowering it.")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["confidence", "support"], ascending=False).reset_index(drop=True)

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]


def popular_products(df: pd.DataFrame, top_n=TOP_N) -> pd.DataFrame:
    return (
        df.groupby("Description", as_index=False)["Quantity"].sum()
        .sort_values("Quantity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def top_revenue_products(df: pd.DataFrame, top_n=TOP_N) -> pd.DataFrame:
    return (
        df.groupby("Description", as_index=False)["TotalPrice"].sum()
        .rename(columns={"TotalPrice": "revenue"})
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def top_products_by_segment(df: pd.DataFrame, top_n_per_segment=5) -> pd.DataFrame:
    segments_path = PROCESSED_DIR / "customer_segments.csv"
    if not segments_path.exists():
        logger.info("  customer_segments.csv not found — run segmentation.py first; skipping this step.")
        return pd.DataFrame()

    segments = pd.read_csv(segments_path)[["CustomerID", "segment"]]
    merged = df.merge(segments, on="CustomerID", how="inner")

    grouped = (
        merged.groupby(["segment", "Description"], as_index=False)["Quantity"].sum()
        .sort_values(["segment", "Quantity"], ascending=[True, False])
    )
    return grouped.groupby("segment").head(top_n_per_segment).reset_index(drop=True)


if __name__ == "__main__":
    logger.info("Loading cleaned transactions...")
    df = load_clean_transactions()

    logger.info("Finding top products (keeps Apriori fast on CPU)...")
    top_products = get_top_products(df)
    logger.info(f"  restricting basket-building to top {len(top_products)} products")

    logger.info("Building baskets from invoices...")
    baskets = build_baskets(df, top_products)
    logger.info(f"  {len(baskets):,} multi-item baskets")

    logger.info(f"Running Apriori (min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE})...")
    rules = run_apriori(baskets)
    logger.info(f"  found {len(rules)} association rules")
    rules.to_csv(PROCESSED_DIR / "association_rules.csv", index=False)

    logger.info("Computing popular products...")
    popular_products(df).to_csv(PROCESSED_DIR / "popular_products.csv", index=False)

    logger.info("Computing top revenue products...")
    top_revenue_products(df).to_csv(PROCESSED_DIR / "top_revenue_products.csv", index=False)

    logger.info("Computing top products by segment...")
    seg_top = top_products_by_segment(df)
    if not seg_top.empty:
        seg_top.to_csv(PROCESSED_DIR / "segment_top_products.csv", index=False)

    logger.info(f"\nSaved outputs -> {PROCESSED_DIR}")
