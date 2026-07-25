# RetailIQ AI

An end-to-end retail analytics project: synthetic sales data → cleaning →
feature engineering → forecasting (4 models) → SHAP explainability →
RFM + KMeans customer segmentation → Streamlit dashboard, with Apriori
product recommendations as a bonus module.

## Project Structure

```
retailiq-ai/
├── data/
│   ├── raw/                    # generated synthetic dataset (csv)
│   └── processed/              # cleaned + feature-engineered data
├── notebooks/                  # exploratory notebooks mirroring src/
├── src/
│   ├── data_generation.py      # synthetic sales + customer + basket data
│   ├── preprocessing.py        # cleaning: dedupe, impute, cap outliers
│   ├── feature_engineering.py  # lag / rolling / calendar features
│   ├── forecasting.py          # Linear Reg, Random Forest, XGBoost, Prophet
│   ├── explainability.py       # SHAP summary + waterfall plots
│   ├── segmentation.py         # RFM scoring + KMeans clustering
│   └── recommendation.py       # bonus: Apriori association rules
├── app.py                      # Streamlit dashboard (entry point)
├── reports/                    # bonus: PDF report generator output
├── models/                     # saved trained models + plots
├── requirements.txt
├── README.md
└── PROJECT_NOTES.md            # plain-language explain notes (viva prep)
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify
python -c "import pandas, sklearn, xgboost, prophet, shap, streamlit; print('all good')"
```

## Running the Pipeline

Run these once, in order, from the project root:

```bash
python src/data_generation.py       # creates data/raw/*.csv
python src/preprocessing.py         # creates data/processed/sales_clean.csv, daily_totals.csv
python src/feature_engineering.py   # creates data/processed/features.csv
python src/forecasting.py           # trains 4 models, saves models/best_model.pkl + comparison
python src/explainability.py        # generates SHAP plots + importance table
python src/segmentation.py          # RFM + KMeans, creates data/processed/customer_segments.csv
python src/recommendation.py        # bonus: Apriori rules -> data/processed/association_rules.csv
```

Then launch the dashboard:

```bash
streamlit run app.py
```

## What Each Module Does

| Module | Purpose |
|---|---|
| `data_generation.py` | Generates 2 years of synthetic daily sales (3 stores × 5 categories), a customer transaction log, and market-basket data. Deliberately injects missing values, duplicates, and outliers so the cleaning step has real work to do. |
| `preprocessing.py` | Removes duplicates, imputes missing `units_sold` with category medians, caps outliers (1st/99th percentile per category), aggregates to a single daily time series. |
| `feature_engineering.py` | Adds lag features (sales N days ago), rolling mean/std (trailing windows, shifted to avoid leakage), and calendar features (day of week, month, weekend flag, etc). |
| `forecasting.py` | Trains Linear Regression, Random Forest, XGBoost, and Prophet on a chronological train/test split (last 60 days held out). Compares R² / RMSE / MAE and saves the best model. |
| `explainability.py` | Runs SHAP `TreeExplainer` on the XGBoost model to produce a global summary plot and a single-prediction waterfall plot. |
| `segmentation.py` | Computes RFM (Recency, Frequency, Monetary) per customer, uses the elbow method to pick k, clusters with KMeans, and labels clusters (VIP / Loyal / Potential Loyalist / At Risk). |
| `recommendation.py` (bonus) | Apriori algorithm on market-basket data to surface "bought X → also bought Y" rules. |
| `app.py` | Interactive Streamlit dashboard with 6 tabs: Overview, Forecasting, Explainability, Segmentation, Recommendations, and What-If Predictor. See "Interactive Features" below. |

## Interactive Features

The dashboard isn't just static charts — it reruns parts of the pipeline live based on your inputs:

- **Sidebar filters**: store, category, and date range filter the Overview tab's charts and raw data table (with CSV download).
- **Forecasting tab**: a horizon slider (7–90 days) re-generates a live Prophet forecast with a confidence-interval band, plotted alongside actual history.
- **Explainability tab**: a top-N slider controls how many SHAP features are shown.
- **Segmentation tab**: a K slider re-runs KMeans clustering live (2–8 clusters) and updates segment sizes, a Recency-vs-Monetary bubble chart, and a customer table you can filter by clicking a segment.
- **Recommendations tab**: support/confidence sliders re-run Apriori live and plot rules on a support-vs-confidence scatter (bubble size/color = lift).
- **What-If Predictor tab**: build a custom scenario (day of week, month, lag values, rolling average, promo/holiday flags) with sliders, get a live XGBoost prediction, and see a live SHAP waterfall explaining *why* — no need to touch code to explore "what if" scenarios.

All charts use Plotly (zoom, pan, hover tooltips) instead of static matplotlib images.

## Notes

- Time series data is **never** randomly shuffled — the train/test split is chronological (last 60 days = test set) to avoid leaking future information into training.
- Rolling features are computed on `shift(1)` first so the current day's own value never leaks into its own rolling stats.
- `data/`, `models/`, and `venv/` are excluded from git (see `.gitignore`) — they're regenerable by rerunning the pipeline.
- See `PROJECT_NOTES.md` for plain-language explanations of each technique, written for viva/defense prep.
