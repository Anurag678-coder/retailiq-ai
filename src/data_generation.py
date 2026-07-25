"""
data_generation.py
Generates a synthetic retail sales dataset for RetailIQ AI.

Simulates ~2 years of daily sales across multiple stores and product
categories, with realistic trend, weekly seasonality, yearly seasonality,
promotions, and noise. Also generates a customer transactions table used
for RFM segmentation and an Apriori-style basket table.

Run:
    python src/data_generation.py
Outputs:
    data/raw/sales_data.csv
    data/raw/customer_transactions.csv
    data/raw/basket_transactions.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
np.random.seed(RNG_SEED)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STORES = ["Store_A", "Store_B", "Store_C"]
CATEGORIES = ["Electronics", "Grocery", "Apparel", "Home_Goods", "Toys"]
PRODUCTS = {
    "Electronics": ["Headphones", "Smartwatch", "Bluetooth_Speaker"],
    "Grocery": ["Coffee", "Snack_Pack", "Cereal"],
    "Apparel": ["T_Shirt", "Jeans", "Jacket"],
    "Home_Goods": ["Blender", "Bedsheet_Set", "Lamp"],
    "Toys": ["Puzzle", "Action_Figure", "Board_Game"],
}


def generate_sales_data(start="2023-01-01", end="2024-12-31"):
    """Generate a daily sales dataset per store x category."""
    date_range = pd.date_range(start=start, end=end, freq="D")
    rows = []

    for store in STORES:
        store_multiplier = np.random.uniform(0.8, 1.3)
        for category in CATEGORIES:
            base_demand = np.random.uniform(40, 120)
            trend_slope = np.random.uniform(0.01, 0.05)

            for i, date in enumerate(date_range):
                # Trend: slow growth over time
                trend = base_demand + trend_slope * i

                # Weekly seasonality: weekends higher for most categories
                weekday = date.weekday()
                weekly_factor = 1.25 if weekday >= 5 else 1.0

                # Yearly seasonality: bump around Nov-Dec (holiday season)
                yearly_factor = 1.0
                if date.month in (11, 12):
                    yearly_factor = 1.4
                elif date.month in (1,):
                    yearly_factor = 0.85

                # Random promotion days (~5% of days) with a sales bump
                is_promo = np.random.rand() < 0.05
                promo_factor = np.random.uniform(1.3, 1.8) if is_promo else 1.0

                noise = np.random.normal(0, base_demand * 0.08)

                units_sold = max(
                    0,
                    trend
                    * weekly_factor
                    * yearly_factor
                    * promo_factor
                    * store_multiplier
                    + noise,
                )
                units_sold = int(round(units_sold))

                unit_price = np.random.uniform(8, 150)
                revenue = round(units_sold * unit_price, 2)

                rows.append(
                    {
                        "date": date,
                        "store": store,
                        "category": category,
                        "units_sold": units_sold,
                        "unit_price": round(unit_price, 2),
                        "revenue": revenue,
                        "is_promo": int(is_promo),
                    }
                )

    df = pd.DataFrame(rows)

    # Inject some realistic messiness for the cleaning step to handle
    missing_idx = df.sample(frac=0.01, random_state=RNG_SEED).index
    df.loc[missing_idx, "units_sold"] = np.nan

    dup_rows = df.sample(frac=0.005, random_state=RNG_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    outlier_idx = df.sample(frac=0.002, random_state=RNG_SEED + 1).index
    df.loc[outlier_idx, "units_sold"] = df.loc[outlier_idx, "units_sold"] * 15

    return df.sort_values("date").reset_index(drop=True)


def generate_customer_transactions(n_customers=800, n_transactions=6000, start="2023-01-01", end="2024-12-31"):
    """Generate a customer transaction table for RFM segmentation."""
    date_range = pd.date_range(start=start, end=end, freq="D")
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

    # Give customers different "activity profiles" so segments emerge naturally
    profile_weights = np.random.dirichlet(np.ones(n_customers) * 0.3)

    rows = []
    for _ in range(n_transactions):
        cust = np.random.choice(customer_ids, p=profile_weights)
        date = np.random.choice(date_range)
        amount = round(np.random.gamma(shape=2.0, scale=35), 2)
        rows.append({"customer_id": cust, "transaction_date": date, "amount": amount})

    df = pd.DataFrame(rows)
    return df.sort_values("transaction_date").reset_index(drop=True)


def generate_basket_transactions(n_baskets=1500):
    """Generate market-basket style transactions for Apriori."""
    all_products = [p for plist in PRODUCTS.values() for p in plist]
    # Define a few "hidden" association rules to make Apriori results meaningful
    rows = []
    for i in range(n_baskets):
        basket = set(np.random.choice(all_products, size=np.random.randint(1, 5), replace=False))
        # Inject association: Coffee -> Snack_Pack, Headphones -> Bluetooth_Speaker
        if "Coffee" in basket and np.random.rand() < 0.6:
            basket.add("Snack_Pack")
        if "Headphones" in basket and np.random.rand() < 0.5:
            basket.add("Bluetooth_Speaker")
        rows.append({"basket_id": f"B_{i:05d}", "items": ",".join(sorted(basket))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating sales_data.csv ...")
    sales_df = generate_sales_data()
    sales_df.to_csv(RAW_DIR / "sales_data.csv", index=False)
    print(f"  -> {len(sales_df)} rows written")

    print("Generating customer_transactions.csv ...")
    cust_df = generate_customer_transactions()
    cust_df.to_csv(RAW_DIR / "customer_transactions.csv", index=False)
    print(f"  -> {len(cust_df)} rows written")

    print("Generating basket_transactions.csv ...")
    basket_df = generate_basket_transactions()
    basket_df.to_csv(RAW_DIR / "basket_transactions.csv", index=False)
    print(f"  -> {len(basket_df)} rows written")

    print("Done. Files saved to:", RAW_DIR)
