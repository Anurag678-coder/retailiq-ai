# PROJECT_NOTES.md — Design Decisions, Tradeoffs & Viva Prep

## Why real data instead of synthetic

The original version of this project used a synthetic data generator so the
pipeline could be built and demoed without waiting on a dataset. That's fine
for a first pass, but a portfolio project is stronger — and more honest in
an interview — if it's proven against real, messy data with real problems:
cancellations, missing customer IDs, inconsistent column naming across
sources, genuine outliers mixed in with real bulk orders. The synthetic
generator is kept in `legacy/` for reference but is not used anywhere in the
default pipeline.

## Data Cleaning

Real transaction exports are messier than synthetic data in specific ways:
- **Cancelled invoices** (`InvoiceNo` starting with `C`) are returns/reversals,
  not sales — excluded entirely rather than netted against original orders,
  to keep the sales pipeline simple and explainable.
- **Missing `CustomerID`** usually means a guest or offline/POS sale. These
  rows are kept for sales/revenue totals (they're still real transactions)
  but dropped when building customer-level features, since RFM and CLV are
  meaningless without knowing who the customer is.
- **Outliers are capped, not dropped** (1st/99th percentile on Quantity and
  UnitPrice). Real retail data has genuine large wholesale orders mixed in
  with data-entry typos — winsorizing keeps the signal from real bulk buyers
  while limiting how much a handful of extreme rows can distort aggregates.

**Why:** garbage in, garbage out — every downstream model depends on this step.

## Feature Engineering

Two separate feature sets, because forecasting and segmentation need
different grains of data:

- **Daily features** (for forecasting): lag features ("sales N days ago"),
  rolling mean/std (smooths day-to-day noise, computed on `shift(1)` so a
  day's own value never leaks into its own rolling stats), and calendar
  features (day of week, month, weekend flag).
- **Customer features** (for segmentation): Recency, Frequency, Monetary,
  average basket size, and a simple observed Customer Lifetime Value
  (average order value × purchase frequency × active lifespan in months).
  This CLV is descriptive/historical, not predictive — a proper predictive
  CLV needs a survival or probabilistic model (e.g. BG/NBD), which is out of
  scope here but noted as a future improvement.

`avg_order_value` is computed for the dashboard but deliberately **excluded**
from the forecasting feature set — it's derived from the same day's revenue,
so using it to predict the same day's units sold would leak the answer
through the back door.

## Forecasting Models

- **Linear Regression**: simple baseline, interpretable coefficients, can't
  capture non-linear patterns.
- **Random Forest**: many decision trees on random subsets, averaged —
  captures non-linearity and feature interactions without much tuning.
- **XGBoost**: builds trees sequentially, each correcting the previous ones'
  errors — usually the strongest performer on tabular data like this.
- **LightGBM** (optional): another gradient-boosted tree implementation,
  included as a bonus comparison point if installed; the pipeline runs fine
  without it.

Compared with **R²**, **RMSE**, and **MAE** on a **chronological** train/test
split (last N days held out, never shuffled — shuffling a time series leaks
future information into training and makes results look better than they'd
be in real deployment).

**Why 3–4 models and not more:** enough to compare a linear baseline against
bagged and boosted tree ensembles, without turning the project into an
exhaustive model zoo that adds noise, not signal, to a portfolio piece.

## SHAP (Explainability)

SHAP shows how much each feature pushed a single prediction up or down,
relative to the average prediction.
- **Summary plot** = global view — which features matter most across all predictions.
- **Waterfall plot** = one prediction, broken into feature contributions.

SHAP's `TreeExplainer` only works on tree-based models, so the explained
model is always XGBoost/LightGBM/Random Forest — never Linear Regression,
even if it happened to score best on a given run.

**Why:** turns a "black box" model into something a business user can
question and trust — e.g. "this forecast is high mainly because it's a
weekend near the December peak."

## RFM + KMeans Segmentation

RFM = Recency (days since last order), Frequency (number of distinct
orders), Monetary (total spend). Each customer gets a 1–5 score on each
dimension; KMeans then groups customers with similar RFM profiles.

**How k was chosen:** both the elbow method (inertia vs. k) and the
silhouette score are plotted (`models/elbow_plot.png`). The final model
uses a fixed k=5 for stability and easy labeling (Champions, Loyal
Customers, Big Spenders, Potential Loyalists, At Risk / Lost) — the plot
documents that this is a reasonable choice rather than an arbitrary one,
without making cluster count non-deterministic between runs.

**Why:** turns raw transaction logs into segments a marketing team can act
on directly (e.g. a win-back campaign for "At Risk").

## Recommendations

Kept deliberately simple, per the project brief — no LLM, no vector
database, no RAG:
- **Apriori** on real per-invoice baskets → "frequently bought together" rules.
  - Support = how often an item combination appears across all baskets.
  - Confidence = given the antecedent was bought, how often the consequent was too.
  - Lift = how much more likely the consequent is given the antecedent, vs.
    random chance (lift > 1 = real association).
- **Popular products** = highest total quantity sold.
- **Top revenue products** = highest total revenue.
- **Top products by segment** = best sellers within each customer segment —
  ties recommendations back to the segmentation output.

## Dashboard (Streamlit)

`app.py` loads the saved `.csv`/`.pkl` outputs from every module above and
displays them across six tabs (Overview, Sales Dashboard, Forecasting,
Customer Segmentation, Recommendations, SHAP Explainability), with KPIs
pinned at the top. It doesn't retrain anything live — training happens once
via the `src/` scripts, and the dashboard only reads and visualizes results.

**Why Streamlit:** fastest way to turn a data science pipeline into an
interactive tool a non-technical stakeholder can click through, without
writing custom frontend code.

## Known Limitations

- CLV is a simple historical estimate, not a predictive model.
- The dataset covers one retailer over two years — forecasts and segments
  reflect that retailer's specific customer base and seasonality, and won't
  generalize to a different business without retraining.
- Apriori's `min_support`/`min_confidence` thresholds are fixed constants;
  a production system would likely tune these per-category rather than
  globally.
- No hyperparameter tuning (e.g. grid/Bayesian search) — models use
  reasonable defaults, which keeps the project simple and fast to run, at
  some cost to raw accuracy.

## Future Scope

See the README's "Future Improvements" section — deep learning forecasting,
predictive CLV / churn modeling, and a live deployment are the most natural
next steps.
