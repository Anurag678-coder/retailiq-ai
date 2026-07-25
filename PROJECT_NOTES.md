# PROJECT_NOTES.md — Plain-Language Explain Sheet

This is the viva script. Every module gets 2–3 plain-language lines,
written as you finish that piece — not crammed at the end.

---

## Data Cleaning & EDA

Raw data always has problems: missing values, duplicate rows, and outliers.
We impute missing `units_sold` using the category median (robust to skew),
drop exact duplicates, and cap extreme outliers at the 1st/99th percentile
per category so a few freak values don't distort the model.
**Why:** garbage in, garbage out — every model downstream depends on this step.

## Feature Engineering

Lag features = "sales from N days ago" used as input to predict today.
Rolling stats = "average / std of the last N days" — smooths out day-to-day noise.
Calendar features = day of week, month, weekend flag — captures patterns like
weekend spikes or holiday-season demand that a raw date column can't express directly.
**Why:** a model can't guess a trend or seasonal pattern from a single row — it
needs recent history and calendar context as explicit inputs.

## Forecasting Models

- **Linear Regression**: simple baseline. Assumes a straight-line relationship
  between features and sales. Fast, interpretable coefficients, but can't
  capture non-linear patterns.
- **Random Forest**: many decision trees trained on random subsets of data/features,
  averaged together. Captures non-linear relationships and feature interactions
  without much tuning.
- **XGBoost**: builds trees sequentially, where each new tree corrects the
  errors of the previous ones ("gradient boosting"). Usually the strongest
  performer on tabular data like this.
- **Prophet**: purpose-built for time series. Explicitly models trend +
  weekly/yearly seasonality, which fits daily retail sales very naturally.

We compare all four with **R²** (variance explained, higher is better),
**RMSE** (root mean squared error — penalizes big misses more), and **MAE**
(mean absolute error — easier to interpret in raw units). The train/test split
is **chronological** (last 60 days held out), never shuffled — shuffling a time
series leaks future information into training and makes results look better
than they'd be in real deployment.
**Why 4 models, not one:** to compare a linear approach, a bagged tree ensemble,
a boosted tree ensemble, and a dedicated time-series model — covering the main
families of forecasting techniques.

## SHAP (Explainability)

SHAP shows how much each feature pushed a single prediction up or down,
relative to the average prediction.
- **Summary plot** = global view — which features matter most across all predictions.
- **Waterfall plot** = one prediction, broken into feature contributions.
**Why:** turns a "black box" model into something a business user can question
and trust — e.g. "this forecast is high mainly because it's a weekend in December."

## RFM + KMeans Segmentation

RFM = **Recency** (days since last purchase), **Frequency** (number of
purchases), **Monetary** (total spend). Each customer gets a 1–5 score on
each dimension. KMeans then groups customers with similar RFM profiles into
clusters, which we label by their average score: VIP, Loyal, Potential
Loyalist, At Risk.
**How we picked cluster count (k):** the elbow method — we plotted inertia
(within-cluster spread) against k and picked the point where adding more
clusters stopped meaningfully reducing it.
**Why:** turns raw transaction logs into segments a marketing team can
actually act on (e.g. win-back campaign for "At Risk").

## Apriori (Bonus — Recommendations)

Apriori finds "customers who bought X also bought Y" rules from basket data.
- **Support** = how often an item combination appears across all baskets.
- **Confidence** = given the antecedent was bought, how often the consequent
  was too.
- **Lift** = how much more likely the consequent is given the antecedent, vs.
  random chance. Lift > 1 means a real association, not coincidence.
**Why:** powers "frequently bought together" style recommendations.

## Dashboard (Streamlit)

`app.py` loads the saved `.csv`/`.pkl` outputs from every module above and
displays them in tabs: Overview, Forecasting, Explainability, Segmentation,
Recommendations. It doesn't retrain anything live — training happens once via
the `src/` scripts, and the dashboard just reads and visualizes the results.
**Why Streamlit:** fastest way to turn a data science pipeline into an
interactive tool a non-technical stakeholder can click through, without
writing custom frontend code.
**How the dashboard connects to the models:** models are saved as `.pkl` files
during training; the dashboard loads them (or their pre-computed outputs) and
displays predictions/metrics — it never re-runs `.fit()`.
