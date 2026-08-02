# CHANGELOG — Production Refactor

Per-file summary of what changed and why. Compare against the original ZIP.

## Fixed / Added (latest pass)

| File | What changed | Why |
|---|---|---|
| `.streamlit/config.toml` | New. Explicit light theme with locked colors (`backgroundColor`, `textColor`, etc.) instead of relying on Streamlit's OS/browser dark-mode auto-detection. | The KPI metric cards and surrounding columns were rendering white-text-on-white-background for some viewers — a theme auto-detection mismatch between the OS/browser and Streamlit, not a bug in the metric code itself (which already avoided custom CSS for exactly this reason). Locking the theme removes the ambiguity entirely. |
| `src/feature_engineering.py` | `build_customer_features` now excludes `CustomerID == 0` in addition to `NaN`. | Some Online Retail dataset mirrors encode a missing/unknown CustomerID as the literal value `0` rather than `NaN`. Left unfiltered, this shows up as one fake "customer" with an impossibly high order count (1000+ orders), which single-handedly distorts the top RFM segment (e.g. inflating "Champions" into a 1-person segment with unrealistic averages). Excluding it is a real data-cleaning decision, not a workaround — always verify a dataset's placeholder/sentinel values before doing customer-level analysis. |
| `app.py` | New **🏆 Top Performers** tab: top areas/countries by revenue, top products (by units sold and by revenue) within the selected area, and top customers within the selected area — all driven by the existing sidebar Country + date-range filters, so picking a country drills every chart in the tab into that market. | Requested addition: surface "which area sold the most, which products drove it, and who the top customers are in that area" as one connected view, reusing the filtering already built into the sidebar rather than adding a parallel filter system. |

## Added

| File | Purpose |
|---|---|
| `src/download_data.py` | New. Downloads the real Online Retail II dataset from UCI (auto, via `ucimlrepo` or direct request), with manual-download instructions as a fallback. |
| `CHANGELOG.md` | New. This file. |
| `legacy/README.md` | New. Explains what's in `legacy/` and why it's not part of the pipeline. |

## Rewritten

| File | What changed | Why |
|---|---|---|
| `src/preprocessing.py` | Reads real `data/raw/online_retail.csv` instead of synthetic `sales_data.csv`. Normalizes column-name variants across UCI/Kaggle mirrors. Added: cancelled-invoice removal (`InvoiceNo` starting with `C`), invalid quantity/price removal, missing-description removal, nullable `CustomerID` handling (guest sales kept for revenue, excluded later for RFM), `TotalPrice` column. Outlier capping logic kept (still winsorized, now on real Quantity/UnitPrice). | Real data has cancellations, guest checkouts, and invalid rows that synthetic data didn't need to model. |
| `src/feature_engineering.py` | Still builds daily lag/rolling/calendar features (logic unchanged), but now also builds `customer_features.csv` — Recency, Frequency, Monetary, average basket size, simple CLV, purchase frequency — computed from real per-invoice data. Added `avg_order_value` as a daily business KPI (explicitly excluded from the forecasting feature list to avoid leakage). | Segmentation needs real per-customer RFM; the old version relied on a separate synthetic `customer_transactions.csv`. |
| `src/forecasting.py` | Same LR/RF/XGBoost logic and chronological split, `Prophet` replaced with optional `LightGBM` (try/except, same pattern as before). Test window now scales down automatically on smaller datasets (`min(TEST_DAYS, len(df)//5)`) instead of assuming a fixed 730-day synthetic history. | Prophet was the heaviest dependency for the smallest accuracy gain; LightGBM is lighter and still gradient-boosted for a meaningful 4th comparison point. Real daily-aggregated history from the dataset's ~2-year span needed a split that doesn't assume the exact synthetic date range. |
| `src/explainability.py` | Explainer model priority list updated to `XGBoost → LightGBM → Random Forest` (was `XGBoost → Random Forest`). Test-window sizing matches the new dynamic logic in `forecasting.py`. Everything else (SHAP summary/waterfall/importance generation) unchanged — it was already generic over feature columns. | Keep the explainer selection consistent with whichever tree models actually got trained. |
| `src/segmentation.py` | Loads `customer_features.csv` (real RFM) instead of recomputing from synthetic `customer_transactions.csv`. Added silhouette score alongside the elbow plot (`find_optimal_k` now plots both). Cluster count raised from 4 to 5 with an expanded, more standard label pool (`Champions, Loyal Customers, Big Spenders, Potential Loyalists, At Risk, Lost Customers`). | Elbow plots alone are subjective to read; silhouette score gives a second, more quantitative signal for the same chart. The 5-label pool matches commonly used RFM segment naming in retail analytics. |
| `src/recommendation.py` | Apriori logic kept (same library, same rules output), now built from real per-invoice `Description` baskets in `sales_clean.csv` instead of a separate synthetic `basket_transactions.csv`. Added three new simple recommenders: `popular_products`, `top_revenue_products`, `top_products_by_segment` (the last one joins with `customer_segments.csv`). Lowered default `min_support`/`min_confidence` (0.03→0.01, 0.3→0.2) since real per-invoice baskets are far more diverse than the synthetic basket generator's constrained item pool. | Spec asked for "frequently bought together, popular products, top revenue products, top products by segment" — only the first existed before. |
| `app.py` | **v2 rewrite.** Restructured into 6 tabs per spec: Overview, Sales Dashboard (new), Forecasting, Customer Segmentation, Recommendations (now shows 4 recommendation types instead of 1), SHAP Explainability. KPIs moved out of the Overview tab into a top-level row visible on every tab. All charts converted from basic `st.line_chart`/`st.bar_chart` to interactive **Plotly**. Added a sidebar date-range + country filter, CSV download buttons on key tables, and a "What-If Forecast Simulator" that lets you change day-of-week/month-start/month-end and see the best model's prediction update live. Custom CSS for KPI cards. | Spec explicitly requested these sections and "KPIs at top." The Plotly rewrite + filters + simulator address dashboard polish for a portfolio-quality presentation. |
| `requirements.txt` | Removed `prophet` and `seaborn` (unused — `matplotlib` covers the two static plots that exist). Pinned upper bounds on core packages. Added `ucimlrepo`, `requests` (for `download_data.py`) and `lightgbm` (optional model), clearly marked as optional in a comment. | Prophet was a heavy, sometimes finicky dependency for one model out of four; removing it also removed the `cmdstan` build-toolchain requirement that trips up a lot of first-time installs. |
| `README.md` | Full rewrite: real dataset section with citation/license, updated architecture diagram, updated folder structure (`download_data.py`, `legacy/`, `customer_features.csv`, etc.), Results/Screenshots/Future Improvements/Resume Bullets sections added per spec. | Spec required these sections explicitly. |
| `PROJECT_NOTES.md` | Full rewrite: added a "Why real data instead of synthetic" section, explicit tradeoff notes (CLV is historical not predictive, fixed Apriori thresholds, no hyperparameter tuning), and a "Known Limitations" + "Future Scope" section. | Spec required documented design decisions and tradeoffs, not just technique explanations. |

## Moved (not deleted — kept for reference, excluded from the active pipeline)

| File | Moved to |
|---|---|
| `src/data_generation.py` | `legacy/data_generation.py` |
| `notebooks/*.ipynb` | `legacy/notebooks/` |

## Unchanged in logic (only cosmetic: logging via `logging` module instead of bare `print`, minor docstring updates)

- Core preprocessing steps: dedupe, outlier capping approach
- Core forecasting steps: chronological split, LR/RF/XGBoost training, model comparison/save logic
- Core SHAP generation: `TreeExplainer`, summary plot, waterfall plot
- Core Apriori mechanics: `TransactionEncoder` → `apriori` → `association_rules`

## Not changed

- `data/raw/`, `data/processed/`, `models/`, `reports/` — regenerated by
  running the pipeline; nothing to diff (old synthetic CSVs excluded from
  this refactor's output; `.gitkeep` placeholders added instead)

## v3 — Bug fixes + new features (this round)

| File | Change | Why |
|---|---|---|
| `src/segmentation.py` | **Bug fix.** `pd.qcut` on raw `recency`/`monetary` with `duplicates="drop"` could silently produce fewer than 5 bins when real data has many tied values, then crash because `labels=[5,4,3,2,1]` still expected exactly 5. Fixed by ranking (`.rank(method="first")`) before `qcut`, same pattern already used for `frequency` — guarantees unique bin edges regardless of how many ties exist in the real data. Verified against a tie-heavy synthetic dataset (80 customers, many same-day purchases) that reliably triggered the old bug. | Small/toy test data rarely has enough ties to hit this; the real ~4,000-customer dataset almost certainly does. |
| `src/recommendation.py` | **Bug fix / performance.** Apriori was running against *all* unique products (4,000+ in the real dataset), which builds a huge one-hot basket matrix — slow to the point of hanging on a CPU-only laptop with no real accuracy benefit (long-tail products barely co-occur enough to form rules anyway). Restricted basket-building to the top 200 best-selling products (`TOP_N_PRODUCTS_FOR_APRIORI`). | Keeps Apriori fast and memory-safe with no GPU, while keeping the business-relevant rules. |
| `requirements.txt` | Removed upper-bound version pins (e.g. `numpy<2.0`) that could make `pip`'s resolver fail against currently-released package versions. Added `fastapi`, `uvicorn`. | Tight pins that were reasonable when this was written can silently become impossible to satisfy as the package ecosystem moves forward — loosening them avoids confusing install-time resolver errors. |
| `src/cohort_analysis.py` | **New.** Monthly cohort retention table — pandas-only (groupby + pivot), no new dependency. | Common, resume-recognizable technique missing from the original scope; deliberately kept simple (no new library) to stay easy to explain in a viva. |
| `api.py` | **New.** FastAPI REST layer over the same processed outputs (KPIs, daily sales, forecast comparison + a live what-if prediction endpoint, segments, recommendations). Read-only, no auth/database — a deliberately simple backend layer, not a production service. | Makes the project relevant for Backend/Data Analyst-flavored resume framing, not only pure DS, without adding real operational complexity. |
| `app.py` | Added a "Cohort Retention" tab (Plotly heatmap) between Sales Dashboard and Forecasting. | Surfaces the new cohort analysis in the dashboard, not just as a CSV. |
| `verify_setup.py` | **New.** Checks that every required package actually imports before you run the pipeline, and tells you exactly which one is missing/broken. | Turns vague "it's not working" install issues into a specific, actionable error message. |

## v4 — Validation rigor + business communication (this round)

| File | Change | Why |
|---|---|---|
| `src/forecasting.py` | **New: walk-forward (expanding-window) cross-validation.** After picking the best model on the original single train/test split, it's re-trained and re-evaluated across 5 expanding time windows (`walk_forward_splits`), saved to `models/walk_forward_cv.csv`. This is the same validation style used in Kaggle's M5 (Walmart) and Rossmann Store Sales forecasting competitions — it shows whether a model is *consistently* good across different time periods, not just lucky on one 60-day window. | A single train/test split is the most common thing beginner forecasting projects skip validating properly; walk-forward CV is one of the clearest, most recognizable signals of forecasting maturity in an interview. |
| `src/generate_report.py` | **New.** Auto-generates `reports/executive_summary.md` — a plain-English business narrative (revenue concentration, at-risk segment value in dollars, forecast accuracy, cross-sell opportunities, recommended actions) built entirely from real numbers pulled out of the pipeline's own outputs. | Turns "I trained some models" into "I can communicate what the models mean for the business" — the actual skill being screened for, and a good one-page artifact to show a teacher/interviewer directly instead of walking them through code. |
| `app.py` | Overview tab now renders the executive summary (if generated) directly in the dashboard, with a download button. | Makes the business narrative visible without needing to open a separate file. |

## v4 — Churn prediction + robust KPI card styling (this round)

| File | Change | Why |
|---|---|---|
| `src/churn_prediction.py` | **New.** Customer churn classifier (Logistic Regression + Random Forest) using a proper temporal holdout split — RFM features computed only from data *before* a cutoff date, labeled by whether the customer returned in the following 90 days. Handles class imbalance (`class_weight="balanced"`) and small/skewed test splits (falls back from stratified split, guards `roc_auc_score` against single-class edge cases) without crashing — verified against a synthetic dataset built specifically to include one-time "churned" customers mixed with loyal repeat customers. | A genuinely different ML skill (classification, not regression) from the existing forecasting module, and the exact kind of problem retail/telecom companies actually build. |
| `app.py` | Added a "Churn Prediction" tab: model comparison, risk-level KPI cards, churn probability histogram, feature importance chart, filterable at-risk customer list with CSV export, and a "$ at risk" callout. Also: **removed the fragile custom CSS** for KPI cards (it went invisible — white text on white background — under a dark Streamlit theme) and replaced it with `st.container(border=True)`, a built-in Streamlit primitive that always matches the active theme correctly instead of fighting it. | Custom `unsafe_allow_html` CSS doesn't adapt to theme automatically; a native Streamlit primitive does, so this class of bug can't recur. |
