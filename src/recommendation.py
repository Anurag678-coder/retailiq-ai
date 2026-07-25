"""
recommendation.py  (BONUS — only build/run after core pipeline is stable)
Market-basket analysis using the Apriori algorithm to find "customers who
bought X also bought Y" style association rules.

Run:
    python src/recommendation.py
Input:
    data/raw/basket_transactions.csv
Output:
    data/processed/association_rules.csv
"""

import pandas as pd
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_baskets():
    df = pd.read_csv(RAW_DIR / "basket_transactions.csv")
    transactions = df["items"].apply(lambda s: s.split(",")).tolist()
    return transactions


def run_apriori(transactions, min_support=0.03, min_confidence=0.3):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found — try lowering min_support")

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["confidence", "support"], ascending=False).reset_index(drop=True)

    # Convert frozensets to readable strings for CSV/dashboard use
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    return rules[cols]


if __name__ == "__main__":
    print("Loading basket transactions...")
    transactions = load_baskets()
    print(f"  {len(transactions)} baskets")

    print("Running Apriori (min_support=0.03, min_confidence=0.3)...")
    rules = run_apriori(transactions)

    print(f"\nFound {len(rules)} association rules. Top 10 by confidence:")
    print(rules.head(10).to_string(index=False))

    rules.to_csv(PROCESSED_DIR / "association_rules.csv", index=False)
    print(f"\nSaved -> {PROCESSED_DIR / 'association_rules.csv'}")
