"""
segmentation.py
Builds RFM (Recency, Frequency, Monetary) scores per customer, then clusters
customers with KMeans into labeled segments (VIP, Loyal, At Risk, New/Low-Value).

Run:
    python src/segmentation.py
Input:
    data/raw/customer_transactions.csv
Output:
    data/processed/customer_segments.csv
    models/kmeans_model.pkl
    models/elbow_plot.png
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def load_transactions():
    df = pd.read_csv(RAW_DIR / "customer_transactions.csv", parse_dates=["transaction_date"])
    return df


def compute_rfm(df, snapshot_date=None):
    """Recency = days since last purchase (lower is better).
    Frequency = number of transactions.
    Monetary = total amount spent."""
    if snapshot_date is None:
        snapshot_date = df["transaction_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("transaction_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("transaction_date", "count"),
        monetary=("amount", "sum"),
    ).reset_index()

    return rfm


def score_rfm(rfm):
    """Quintile-score each dimension 1-5 (5 = best) so it's human-readable in the dashboard."""
    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    return rfm


def find_optimal_k(X_scaled, k_range=range(2, 9)):
    """Elbow method: track inertia (within-cluster sum of squares) as k grows."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "elbow_plot.png", dpi=150)
    plt.close()
    return inertias


def cluster_customers(rfm, n_clusters=4):
    features = rfm[["recency", "frequency", "monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    find_optimal_k(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    return rfm, kmeans, scaler


def label_clusters(rfm):
    """Rank clusters by mean RFM_score and assign business-friendly labels.
    Highest combined score = VIP, lowest = At Risk / Low Value."""
    cluster_rank = (
        rfm.groupby("cluster")["RFM_score"].mean().sort_values(ascending=False).index.tolist()
    )
    n = len(cluster_rank)
    label_pool = ["VIP", "Loyal", "Potential Loyalist", "At Risk", "Low Value"]
    labels = label_pool[:n] if n <= len(label_pool) else [f"Segment {i}" for i in range(n)]

    label_map = {cluster_id: labels[i] for i, cluster_id in enumerate(cluster_rank)}
    rfm["segment"] = rfm["cluster"].map(label_map)
    return rfm, label_map


if __name__ == "__main__":
    print("Loading customer transactions...")
    txns = load_transactions()
    print(f"  {len(txns)} transactions, {txns['customer_id'].nunique()} unique customers")

    print("Computing RFM...")
    rfm = compute_rfm(txns)
    rfm = score_rfm(rfm)

    print("Clustering with KMeans...")
    rfm, kmeans_model, scaler = cluster_customers(rfm, n_clusters=4)
    rfm, label_map = label_clusters(rfm)

    print("\nSegment sizes:")
    print(rfm["segment"].value_counts().to_string())

    print("\nSegment profile (mean R/F/M):")
    print(rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1).to_string())

    rfm.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)
    joblib.dump({"kmeans": kmeans_model, "scaler": scaler, "label_map": label_map},
                MODELS_DIR / "kmeans_model.pkl")

    print(f"\nSaved -> {PROCESSED_DIR / 'customer_segments.csv'}")
    print(f"Saved -> {MODELS_DIR / 'kmeans_model.pkl'}")
    print(f"Saved -> {MODELS_DIR / 'elbow_plot.png'}")
