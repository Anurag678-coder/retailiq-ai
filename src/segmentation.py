"""
segmentation.py
Clusters customers into business-friendly segments using their RFM profile
(Recency, Frequency, Monetary), computed in feature_engineering.py.

Number of clusters (k) is chosen using both the elbow method (inertia) and
the silhouette score, so the choice isn't just eyeballed off one chart.

Run:
    python src/segmentation.py
Input:
    data/processed/customer_features.csv
Output:
    data/processed/customer_segments.csv
    models/kmeans_model.pkl
    models/elbow_plot.png
"""

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RFM_COLS = ["recency", "frequency", "monetary"]
K_RANGE = range(2, 9)

# Ordered best -> worst; used when labeling clusters by their mean RFM rank.
LABEL_POOL = [
    "Champions", "Loyal Customers", "Big Spenders",
    "Potential Loyalists", "At Risk", "Lost Customers",
]


def load_customer_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "customer_features.csv")


def score_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Quintile-score each RFM dimension 1-5 (5 = best) — human-readable in the dashboard."""
    # Rank first, THEN qcut on the ranks (not the raw values). Real data has
    # many tied recency/monetary values (e.g. many customers active "7 days
    # ago"); qcut on raw values with ties can collapse bins below 5 and then
    # crash because `labels` still expects exactly 5. Ranking guarantees
    # unique values going into qcut, so this never happens, regardless of
    # how many ties the real dataset has.
    df["R_score"] = pd.qcut(df["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_score"] = pd.qcut(df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["RFM_score"] = df["R_score"] + df["F_score"] + df["M_score"]
    return df


def find_optimal_k(X_scaled, k_range=K_RANGE):
    """Elbow method (inertia) + silhouette score, plotted side by side.
    We still fit the final model with a fixed, sensible k (see cluster_customers)
    rather than fully automating k selection — that keeps cluster count stable
    and easy to explain, while the plot documents *why* that k is reasonable."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="darkorange")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score by k")

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "elbow_plot.png", dpi=150)
    plt.close()

    best_k_by_silhouette = list(k_range)[int(pd.Series(silhouettes).idxmax())]
    logger.info(f"  silhouette score suggests k={best_k_by_silhouette}")
    return inertias, silhouettes, best_k_by_silhouette


def cluster_customers(rfm: pd.DataFrame, n_clusters=5):
    features = rfm[RFM_COLS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    find_optimal_k(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    return rfm, kmeans, scaler


def label_clusters(rfm: pd.DataFrame):
    """Rank clusters by mean RFM_score and assign business-friendly labels.
    Highest combined score = best customers, lowest = churn risk."""
    cluster_rank = (
        rfm.groupby("cluster")["RFM_score"].mean().sort_values(ascending=False).index.tolist()
    )
    n = len(cluster_rank)
    labels = LABEL_POOL[:n] if n <= len(LABEL_POOL) else [f"Segment {i}" for i in range(n)]

    label_map = {cluster_id: labels[i] for i, cluster_id in enumerate(cluster_rank)}
    rfm["segment"] = rfm["cluster"].map(label_map)
    return rfm, label_map


if __name__ == "__main__":
    logger.info("Loading customer features...")
    cust = load_customer_features()
    logger.info(f"  {len(cust):,} customers")

    logger.info("Scoring RFM...")
    cust = score_rfm(cust)

    logger.info("Clustering with KMeans...")
    cust, kmeans_model, scaler = cluster_customers(cust, n_clusters=5)
    cust, label_map = label_clusters(cust)

    logger.info("\nSegment sizes:\n" + cust["segment"].value_counts().to_string())
    logger.info(
        "\nSegment profile (mean R/F/M):\n"
        + cust.groupby("segment")[RFM_COLS].mean().round(1).to_string()
    )

    cust.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)
    joblib.dump(
        {"kmeans": kmeans_model, "scaler": scaler, "label_map": label_map},
        MODELS_DIR / "kmeans_model.pkl",
    )

    logger.info(f"\nSaved -> {PROCESSED_DIR / 'customer_segments.csv'}")
    logger.info(f"Saved -> {MODELS_DIR / 'kmeans_model.pkl'}")
    logger.info(f"Saved -> {MODELS_DIR / 'elbow_plot.png'}")
