"""
churn_prediction.py
Predicts which customers are likely to churn (stop buying) — a classic,
widely-used retail/telecom ML problem, and a genuinely different skill from
the sales forecasting module (classification vs. regression).

Approach (temporal holdout — avoids the most common beginner mistake, which
is leaking future purchase behavior into the features):

  1. Pick a cutoff date a few months before the end of the dataset.
  2. Compute each customer's RFM profile using ONLY transactions before
     that cutoff (their behavior "so far").
  3. Label churn = 1 if that customer made ZERO purchases in the holdout
     window AFTER the cutoff, churn = 0 if they came back at least once.
  4. Train a classifier on the pre-cutoff RFM features to predict that label.

This mirrors how churn prediction is actually done in industry: you're
always predicting behavior you haven't seen yet from behavior you have.

Run:
    python src/churn_prediction.py
Input:
    data/processed/sales_clean.csv
Output:
    data/processed/churn_predictions.csv     (every customer + churn probability)
    models/churn_model_comparison.csv
    models/churn_model.pkl
    models/churn_feature_importance.csv
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

HOLDOUT_DAYS = 90  # "did they come back in the last 3 months of the dataset?"
FEATURE_COLS = ["recency", "frequency", "monetary", "avg_basket_size", "purchase_frequency_per_month"]


def load_clean_transactions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "sales_clean.csv", parse_dates=["InvoiceDate"])


def build_labeled_dataset(df: pd.DataFrame, holdout_days=HOLDOUT_DAYS):
    """Returns (features_df, cutoff_date) — one row per customer who had
    already purchased at least once before the cutoff."""
    df = df[df["CustomerID"].notna()].copy()

    max_date = df["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days)

    before = df[df["InvoiceDate"] < cutoff_date]
    after = df[df["InvoiceDate"] >= cutoff_date]

    if before["CustomerID"].nunique() == 0:
        raise ValueError(
            "No customers have transactions before the cutoff date — "
            "the dataset may be shorter than HOLDOUT_DAYS. Reduce HOLDOUT_DAYS."
        )

    # --- RFM + basket features computed ONLY from pre-cutoff data ---
    per_invoice = before.groupby(["CustomerID", "InvoiceNo"]).agg(
        invoice_date=("InvoiceDate", "min"),
        invoice_value=("TotalPrice", "sum"),
        basket_size=("StockCode", "nunique"),
    ).reset_index()

    span_days = before.groupby("CustomerID")["InvoiceDate"].agg(lambda s: (s.max() - s.min()).days)

    feats = per_invoice.groupby("CustomerID").agg(
        recency=("invoice_date", lambda x: (cutoff_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("invoice_value", "sum"),
        avg_basket_size=("basket_size", "mean"),
    ).reset_index()

    active_months = np.clip(span_days.reindex(feats["CustomerID"]).values / 30.0, 1, None)
    feats["purchase_frequency_per_month"] = (feats["frequency"] / active_months).round(2)
    feats["monetary"] = feats["monetary"].round(2)
    feats["avg_basket_size"] = feats["avg_basket_size"].round(1)

    # --- Label: did they NOT come back in the holdout window? ---
    returned_customers = set(after["CustomerID"].unique())
    feats["churned"] = (~feats["CustomerID"].isin(returned_customers)).astype(int)

    return feats, cutoff_date


def train_churn_models(feats: pd.DataFrame):
    X = feats[FEATURE_COLS]
    y = feats["churned"]

    # Real churn data is often imbalanced (far more loyal customers than
    # churned ones, or vice versa). Stratified split preserves that ratio in
    # both train/test — but if a class has fewer than 2 members (can happen
    # on very small datasets), stratification itself isn't possible, so we
    # fall back to a plain split rather than crashing.
    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if can_stratify else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    models = {}

    # Logistic Regression: simple, interpretable baseline.
    # class_weight="balanced" matters here — without it, a model can get a
    # deceptively high accuracy just by always predicting the majority class.
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    pred_lr = log_reg.predict(X_test_scaled)
    results["Logistic Regression"] = evaluate(y_test, pred_lr, proba_lr)
    models["Logistic Regression"] = (log_reg, scaler)

    # Random Forest: non-linear, handles feature interactions
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_train, y_train)  # tree models don't need scaling
    proba_rf = rf.predict_proba(X_test)[:, 1]
    pred_rf = rf.predict(X_test)
    results["Random Forest"] = evaluate(y_test, pred_rf, proba_rf)
    models["Random Forest"] = (rf, None)

    comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    comparison_df = comparison_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

    best_name = comparison_df.iloc[0]["model"]
    best_model, best_scaler = models[best_name]

    return comparison_df, best_name, best_model, best_scaler


def evaluate(y_true, y_pred, y_proba) -> dict:
    # roc_auc_score is undefined if the test split ends up with only one
    # class present (possible on small/imbalanced datasets) — guard it
    # rather than letting the whole pipeline crash on an edge case.
    try:
        roc_auc = round(roc_auc_score(y_true, y_proba), 4)
    except ValueError:
        roc_auc = float("nan")

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": roc_auc,
    }


def score_all_customers(feats: pd.DataFrame, best_name, best_model, best_scaler) -> pd.DataFrame:
    """Apply the winning model to every customer (not just the test split) to
    produce an actionable churn-risk list."""
    X_all = feats[FEATURE_COLS]
    X_scored = best_scaler.transform(X_all) if best_scaler is not None else X_all

    feats = feats.copy()
    feats["churn_probability"] = best_model.predict_proba(X_scored)[:, 1].round(4)
    feats["risk_level"] = pd.cut(
        feats["churn_probability"], bins=[-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
    )
    return feats.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def get_feature_importance(best_name, best_model) -> pd.DataFrame:
    if best_name == "Random Forest":
        importance = best_model.feature_importances_
    else:  # Logistic Regression — use absolute coefficient magnitude
        importance = np.abs(best_model.coef_[0])
    return (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    logger.info("Loading cleaned transactions...")
    transactions = load_clean_transactions()

    logger.info(f"Building labeled churn dataset (holdout = last {HOLDOUT_DAYS} days)...")
    feats, cutoff_date = build_labeled_dataset(transactions)
    logger.info(f"  cutoff date: {cutoff_date.date()}")
    logger.info(f"  {len(feats):,} customers, churn rate: {feats['churned'].mean():.1%}")

    logger.info("Training churn models (Logistic Regression, Random Forest)...")
    comparison_df, best_name, best_model, best_scaler = train_churn_models(feats)
    comparison_df.to_csv(MODELS_DIR / "churn_model_comparison.csv", index=False)
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info(f"\nBest model: {best_name}")

    joblib.dump(
        {"model": best_model, "scaler": best_scaler, "model_name": best_name, "feature_cols": FEATURE_COLS},
        MODELS_DIR / "churn_model.pkl",
    )

    logger.info("Scoring every customer...")
    scored = score_all_customers(feats, best_name, best_model, best_scaler)
    scored.to_csv(PROCESSED_DIR / "churn_predictions.csv", index=False)
    logger.info("\nRisk level breakdown:\n" + scored["risk_level"].value_counts().to_string())

    logger.info("Computing feature importance...")
    importance_df = get_feature_importance(best_name, best_model)
    importance_df.to_csv(MODELS_DIR / "churn_feature_importance.csv", index=False)
    logger.info("\n" + importance_df.to_string(index=False))

    logger.info(f"\nSaved -> {PROCESSED_DIR / 'churn_predictions.csv'}")
    logger.info(f"Saved -> {MODELS_DIR / 'churn_model.pkl'}")
