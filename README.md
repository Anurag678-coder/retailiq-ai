# 🛒 RetailIQ AI

An end-to-end retail analytics project built on **real transaction data**: cleaning → feature engineering → sales forecasting (3–4 models) → SHAP explainability → RFM + KMeans customer segmentation → product recommendations (Apriori + rank-based) → an interactive Streamlit dashboard.

No synthetic data, no fake pipeline — this runs on the **Online Retail II** dataset (UCI Machine Learning Repository), a real two-year transaction log from a UK-based online gift retailer.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Running](#running)
- [Folder Structure](#folder-structure)
- [What Each Module Does](#what-each-module-does)
- [Results](#results)
- [Screenshots](#screenshots)
- [Team](#team--origin-point)
- [Future Improvements](#future-improvements)
- [Resume Bullets](#resume-bullets)
- [Notes](#notes)

---

## Features

- **Data cleaning** for real-world messiness: cancelled orders, missing customer IDs, invalid quantities/prices, duplicates, outliers
- **Feature engineering**: lag/rolling/calendar features for forecasting; RFM + simple CLV + basket size at the customer level
- **Cohort retention analysis**: monthly cohort heatmap showing what % of each cohort of new customers returns in later months
- **Churn prediction**: classification models (Logistic Regression, Random Forest) predicting which customers are likely to stop buying, using a proper temporal holdout (train on behavior before a cutoff, label from what happened after it — no future-data leakage)
- **Sales forecasting**: Linear Regression, Random Forest, XGBoost (+ optional LightGBM), compared on a chronological train/test split, with 5-fold walk-forward cross-validation
- **SHAP explainability**: global feature importance + single-prediction waterfall breakdown for the tree-based model
- **Customer segmentation**: RFM scoring, KMeans (k chosen via elbow + silhouette score), labeled into Champions / Loyal Customers / At Risk / etc.
- **Recommendations**: Apriori "frequently bought together" rules, popular products, top revenue products, top products per segment
- **Streamlit dashboard**: KPIs, sales trends, a dedicated Top Performers tab (top areas, top products, and top customers — drillable by country), cohort retention, forecasting, churn prediction, segmentation, recommendations, and SHAP — all in one place. Built with Plotly for interactive charts, a sidebar date/country filter, CSV export buttons on key tables, and a what-if forecast simulator.
- **REST API** (`api.py`, FastAPI): the same pipeline outputs exposed as JSON endpoints with auto-generated Swagger docs — useful if you're presenting this project for a Backend or Data Analyst role, not only a pure Data Science one.

---

## Architecture

    Real Transaction Data (UCI Online Retail II)
            │
            ▼
    Preprocessing  (clean, dedupe, remove cancellations/invalid rows)
            │
            ▼
    Feature Engineering  (daily time-series features + customer RFM/CLV features)
            │
            ├──────────────┬───────────────┬───────────────┐
            ▼              ▼               ▼               ▼
      Forecasting    Segmentation    Recommendations   Churn Prediction
      (LR/RF/XGB)     (RFM+KMeans)   (Apriori + rankings)  (LogReg/RF)
            │              │               │               │
            ▼              │               │               │
      Explainability        │               │               │
       (SHAP)               │               │               │
            │              │               │               │
            └──────────────┴───────────────┴───────────────┘
                            │
                            ▼
             Streamlit Dashboard  +  FastAPI REST API

---

## Dataset

**Online Retail II** — UCI Machine Learning Repository (id 502), donated by Dr. Daqing Chen. Real transactions from a UK-based, non-store online retailer selling all-occasion gifts, 01/12/2009–09/12/2011 (~1M rows across two sheets/years).

| | |
|---|---|
| **UCI** | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| **Kaggle mirror** | https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci |
| **License** | CC BY 4.0 |
| **Citation** | Chen, D. (2015). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |

**Columns:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

---

## Installation

    # 1. Clone the repo
    git clone https://github.com/Anurag678-coder/retailiq-ai
    cd retailiq-ai

    # 2. Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate      # Windows: venv\Scripts\activate

    # 3. Install dependencies
    pip install -r requirements.txt

---

## Running

    # 1. Download the real dataset (auto-downloads via UCI; falls back to
    #    manual instructions if your network blocks it — see the script)
    python src/download_data.py

    # 2. Run the pipeline, in order
    python src/preprocessing.py
    python src/feature_engineering.py
    python src/cohort_analysis.py
    python src/forecasting.py
    python src/explainability.py
    python src/segmentation.py
    python src/recommendation.py
    python src/churn_prediction.py
    python src/generate_report.py

    # 3. Launch the dashboard
    streamlit run app.py

    # 4. (Optional) Launch the REST API — separate terminal
    uvicorn api:app --reload
    # then open http://127.0.0.1:8000/docs for interactive API docs

> **Note:** If `src/download_data.py` can't reach the internet (common on college/office networks), download the dataset manually from the links above and save it as `data/raw/online_retail.csv`.

---

## Folder Structure

    retailiq-ai/
    ├── src/
    │   ├── download_data.py         # Fetches the real Online Retail II dataset
    │   ├── preprocessing.py         # Cleaning pipeline
    │   ├── feature_engineering.py   # Daily + customer-level feature building
    │   ├── forecasting.py           # Multi-model sales forecasting
    │   ├── explainability.py        # SHAP-based model explanation
    │   ├── segmentation.py          # RFM + KMeans customer segmentation
    │   ├── recommendation.py        # Apriori + rank-based recommendations
    │   ├── cohort_analysis.py       # Monthly cohort retention analysis
    │   └── churn_prediction.py      # Classification: predicts customer churn risk
    ├── data/
    │   ├── raw/                     # online_retail.csv goes here (gitignored)
    │   └── processed/                # cleaned data + all pipeline outputs
    ├── models/                      # saved models, metrics, plots (gitignored)
    ├── reports/                     # exported charts/screenshots for write-ups
    ├── legacy/                      # old synthetic-data version, kept for reference only
    ├── .streamlit/                  # dashboard theme config
    ├── app.py                       # Streamlit dashboard
    ├── api.py                       # FastAPI REST API over the same outputs
    ├── verify_setup.py              # Checks your environment before you run anything
    ├── requirements.txt
    ├── PROJECT_NOTES.md             # design decisions, tradeoffs, viva notes
    ├── CHANGELOG.md                 # per-file change history
    └── README.md

---

## What Each Module Does

| Module | Purpose |
|---|---|
| `download_data.py` | Downloads the real Online Retail II dataset from UCI, saves as `data/raw/online_retail.csv`. |
| `preprocessing.py` | Removes duplicates, cancelled invoices, invalid quantity/price, and missing descriptions; normalizes column names/types; caps outliers; builds `TotalPrice`; aggregates a daily time series. |
| `feature_engineering.py` | Daily lag/rolling/calendar features for forecasting, plus per-customer RFM, simple CLV, purchase frequency, and average basket size. |
| `forecasting.py` | Trains Linear Regression, Random Forest, XGBoost (+ LightGBM if installed) on a chronological split, compares R²/RMSE/MAE, saves the best model. Also runs 5-fold walk-forward (expanding-window) cross-validation on the best model. |
| `explainability.py` | SHAP `TreeExplainer` on the tree-based model — global summary plot + single-prediction waterfall. |
| `segmentation.py` | RFM scoring, KMeans clustering (k picked via elbow + silhouette), labeled segments. |
| `recommendation.py` | Apriori "frequently bought together" rules, plus popularity/revenue/segment-based product rankings. |
| `cohort_analysis.py` | Monthly cohort retention table — % of each new-customer cohort that returns in each following month. |
| `churn_prediction.py` | Predicts customer churn using a temporal holdout: RFM features from before a cutoff date predict who doesn't come back after it. |
| `generate_report.py` | Auto-generates a plain-English executive summary (`reports/executive_summary.md`). |
| `app.py` | Streamlit dashboard pulling from all the saved outputs above. |
| `api.py` | FastAPI REST API exposing the same outputs as JSON with auto-generated Swagger docs at `/docs`. |

---

## Results

| Output | Location |
|---|---|
| Best forecasting model | `models/model_comparison.csv` |
| Customer segments found | `data/processed/customer_segments.csv` |
| Churn predictions | `data/processed/churn_predictions.csv` |
| Top association rules | `data/processed/association_rules.csv` |
| Auto-generated summary | `reports/executive_summary.md` |

---

## Screenshots

    reports/screenshot_overview.png
    reports/screenshot_forecasting.png
    reports/screenshot_segmentation.png

---

## Team — Origin Point

| Member | Role | GitHub | Files Owned |
|---|---|---|---|
| **Akshay** | Dashboard, API & Integration | [@Akshay-23A](| `app.py`, `api.py`, `verify_setup.py` |
| **Anurag** | ML Pipeline | [@Anurag678-coder](https://github.com/Anurag678-coder) | `feature_engineering.py`, `forecasting.py`, `explainability.py`, `churn_prediction.py` |
| **Daksh** | Data Pipeline & Analytics | [@dkumarbhp2006-cell](https://github.com/dkumarbhp2006-cell)| `download_data.py`, `preprocessing.py`, `cohort_analysis.py`, `segmentation.py`, `recommendation.py`, `generate_report.py` |

---

## Future Improvements

- Deep learning forecasting (LSTM/Transformer) for comparison against tree models
- Real-time / incremental pipeline instead of batch re-runs
- Multi-country breakdown in the dashboard (the dataset spans many countries)
- Deployment (Streamlit Community Cloud or similar) with a live demo link

---

## Resume Bullets

### Data Science / ML angle
- Built an end-to-end retail analytics pipeline on real transaction records (UCI Online Retail II), covering cleaning, feature engineering, forecasting, and customer segmentation
- Trained and compared 3–4 regression models (Linear Regression, Random Forest, XGBoost) for daily sales forecasting using a leakage-free chronological train/test split
- Applied SHAP to explain tree-based model predictions, surfacing the features driving demand forecasts
- Segmented customers via RFM analysis and KMeans clustering (k selected via elbow method + silhouette score) into actionable groups (Champions, At Risk, etc.)
- Built product recommendations using Apriori market-basket analysis and revenue/popularity ranking
- Ran monthly cohort retention analysis to separate customer acquisition from customer retention trends
- Built a customer churn classifier (Logistic Regression, Random Forest) using a leakage-free temporal holdout, producing a ranked at-risk customer list a retention team could act on directly

### Backend / API angle
- Designed and built a REST API (FastAPI) exposing ML pipeline outputs — KPIs, forecasts, customer segments, recommendations — as JSON endpoints with auto-generated Swagger/OpenAPI documentation
- Structured a modular Python pipeline (9 independent, single-responsibility scripts) with clean file-based I/O contracts between stages

### Data Analyst / BI angle
- Delivered results through an interactive Streamlit + Plotly dashboard with KPIs, filters (date range, country), CSV export, and a what-if simulator
- Translated raw transaction logs into business-ready segments and retention/cohort metrics a marketing team could act on directly

---

## Notes

- Time series data is **never** randomly shuffled — the train/test split is chronological to avoid leaking future information into training.
- Rolling features are computed on `shift(1)` first so a day's own value never leaks into its own rolling stats.
- `avg_order_value` is intentionally excluded from the forecasting feature set — it's derived from same-day revenue, so using it to predict same-day units sold would leak the answer.
- `data/`, `models/`, and `venv/` are excluded from git — they're regenerable by rerunning the pipeline.
- See `PROJECT_NOTES.md` for plain-language explanations of each technique and the design tradeoffs behind them.
