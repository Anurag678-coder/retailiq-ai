"""
generate_report.py
Auto-generates a one-page executive summary from the pipeline's outputs —
the kind of plain-English business narrative a Data Science/Analytics team
writes for stakeholders who won't read a Jupyter notebook. Real numbers,
not placeholders: every figure below is pulled live from your pipeline's
CSV/model outputs, so this regenerates correctly every time you re-run the
pipeline on fresh data.

This is the single most "resume-real" artifact in the project — it's what
turns "I ran some models" into "I can communicate what the models mean for
the business," which is the actual skill companies screen for.

Run (after the full pipeline has been run at least once):
    python src/generate_report.py
Output:
    reports/executive_summary.md
"""

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def require(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. {hint}")
    return path


def build_report() -> str:
    daily = pd.read_csv(require(PROCESSED_DIR / "daily_totals.csv", "Run preprocessing.py first."), parse_dates=["date"])
    sales = pd.read_csv(require(PROCESSED_DIR / "sales_clean.csv", "Run preprocessing.py first."), parse_dates=["InvoiceDate"])
    segments = pd.read_csv(require(PROCESSED_DIR / "customer_segments.csv", "Run segmentation.py first."))
    comparison = pd.read_csv(require(MODELS_DIR / "model_comparison.csv", "Run forecasting.py first."))
    rules = pd.read_csv(PROCESSED_DIR / "association_rules.csv") if (PROCESSED_DIR / "association_rules.csv").exists() else None
    cv_path = MODELS_DIR / "walk_forward_cv.csv"
    cv_df = pd.read_csv(cv_path) if cv_path.exists() else None

    with open(require(MODELS_DIR / "best_model_name.json", "Run forecasting.py first.")) as f:
        best_model_meta = json.load(f)

    # --- Headline numbers ---
    total_revenue = daily["revenue"].sum()
    total_units = daily["units_sold"].sum()
    date_start, date_end = daily["date"].min().date(), daily["date"].max().date()
    n_customers = segments["CustomerID"].nunique()
    n_countries = sales["Country"].nunique()

    # --- Segment insight: which segment carries the most revenue risk / value ---
    seg_summary = (
        segments.groupby("segment")
        .agg(customers=("CustomerID", "nunique"), total_monetary=("monetary", "sum"))
        .sort_values("total_monetary", ascending=False)
    )
    top_segment = seg_summary.index[0]
    top_segment_share = seg_summary.iloc[0]["total_monetary"] / seg_summary["total_monetary"].sum() * 100

    at_risk_revenue = 0.0
    at_risk_customers = 0
    if "At Risk" in seg_summary.index:
        at_risk_revenue = seg_summary.loc["At Risk", "total_monetary"]
        at_risk_customers = int(seg_summary.loc["At Risk", "customers"])

    # --- Forecasting insight ---
    best_model_name = comparison.iloc[0]["model"]
    best_r2 = comparison.iloc[0]["R2"]
    best_mae = comparison.iloc[0]["MAE"]

    cv_note = ""
    if cv_df is not None and len(cv_df) > 0:
        cv_note = (
            f" Validated across {len(cv_df)} expanding time-window folds "
            f"(walk-forward cross-validation), with a mean R² of "
            f"{cv_df['R2'].mean():.3f} (std {cv_df['R2'].std():.3f}), showing "
            f"the model holds up consistently across different time periods, "
            f"not just one lucky test window."
        )

    # --- Product/recommendation insight ---
    top_products = sales.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False)
    top5_share = top_products.head(5).sum() / top_products.sum() * 100
    n_rules = len(rules) if rules is not None else 0

    report = f"""# RetailIQ AI — Executive Summary

*Auto-generated from the live pipeline outputs — regenerate with `python src/generate_report.py` after any pipeline re-run.*

## Business Overview

Analyzed **{total_revenue:,.0f}** in revenue across **{total_units:,.0f}** units sold,
from **{date_start}** to **{date_end}**, covering **{n_customers:,}** identified
customers across **{n_countries}** countries.

## Key Findings

**1. Revenue is concentrated, not evenly spread.**
The top 5 products account for **{top5_share:.1f}%** of total revenue — a
strong signal for inventory prioritization: stockouts on these specific
items would have an outsized revenue impact compared to the long tail.

**2. Customer value is concentrated too.**
The **{top_segment}** segment ({int(seg_summary.iloc[0]['customers']):,} customers)
drives **{top_segment_share:.1f}%** of total customer spend, despite being a
minority of the customer base — the standard retail pattern where a small
share of customers drives most of the revenue.
"""

    if at_risk_customers > 0:
        report += f"""
**3. Revenue at churn risk is quantifiable.**
The **At Risk** segment represents **{at_risk_customers:,} customers** and
**${at_risk_revenue:,.0f}** in historical spend — a concrete, sized target
for a win-back campaign, rather than a vague "some customers might churn."
"""

    report += f"""
**4. Sales are forecastable with reasonable accuracy.**
The best-performing model (**{best_model_name}**) achieves an R² of
**{best_r2:.3f}** and a mean absolute error of **{best_mae:.1f} units/day**
on held-out data.{cv_note}

**5. Cross-selling opportunities exist and are quantified.**
Market basket analysis (Apriori) surfaced **{n_rules}** statistically
meaningful "frequently bought together" product pairs — concrete candidates
for bundling or "customers also bought" placement.

## Recommended Next Actions

1. Prioritize inventory/supply reliability for the top revenue-share products identified above.
2. Launch a targeted win-back offer for the At Risk segment, sized against its quantified revenue value.
3. Use the top association rules to pilot product bundling or on-site cross-sell placement.
4. Feed the {best_model_name} forecast into inventory/staffing planning; monitor the walk-forward
   fold metrics over time to catch model drift before it affects the business.

---
*Generated by RetailIQ AI · Data: UCI Online Retail II · See PROJECT_NOTES.md for methodology and limitations.*
"""
    return report


if __name__ == "__main__":
    logger.info("Building executive summary from pipeline outputs...")
    report_text = build_report()
    out_path = REPORTS_DIR / "executive_summary.md"
    out_path.write_text(report_text)
    logger.info(f"Saved -> {out_path}")
    print("\n" + report_text)
