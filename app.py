"""
app.py — RetailIQ Interactive Dashboard
Streamlit entry point tying together forecasting, SHAP explainability,
customer segmentation, and product recommendations — with live filters,
sliders, and what-if controls (not just static charts).

Run:
    streamlit run app.py

Expects the pipeline to have already been run once:
    python src/data_generation.py
    python src/preprocessing.py
    python src/feature_engineering.py
    python src/forecasting.py
    python src/explainability.py
    python src/segmentation.py
    python src/recommendation.py     # optional, bonus
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="RetailIQ", layout="wide", page_icon="🛒")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_sales_clean():
    return pd.read_csv(PROCESSED_DIR / "sales_clean.csv", parse_dates=["date"])


@st.cache_data
def load_features():
    return pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODELS_DIR / "model_comparison.csv")


@st.cache_data
def load_segments():
    return pd.read_csv(PROCESSED_DIR / "customer_segments.csv")


@st.cache_data
def load_shap_importance():
    path = MODELS_DIR / "shap_feature_importance.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_baskets_raw():
    path = RAW_DIR / "basket_transactions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["items"].apply(lambda s: s.split(",")).tolist()


@st.cache_resource
def load_all_models():
    return joblib.load(MODELS_DIR / "all_models.pkl")


@st.cache_resource
def load_meta():
    with open(MODELS_DIR / "best_model_name.json") as f:
        return json.load(f)


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


def files_missing():
    required = [
        PROCESSED_DIR / "features.csv",
        PROCESSED_DIR / "sales_clean.csv",
        MODELS_DIR / "model_comparison.csv",
        PROCESSED_DIR / "customer_segments.csv",
        MODELS_DIR / "all_models.pkl",
    ]
    return [str(p) for p in required if not p.exists()]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛒 RetailIQ — Retail Data Science & ML Dashboard")
st.caption("Sales forecasting · SHAP explainability · Customer segmentation · Recommendations")

missing = files_missing()
if missing:
    st.error("Pipeline outputs not found. Run the pipeline scripts first (see README.md).")
    for m in missing:
        st.write(f"- `{m}`")
    st.stop()

sales_df = load_sales_clean()
features_df = load_features()
comparison_df = load_model_comparison()
segments_df = load_segments()
importance_df = load_shap_importance()
all_models = load_all_models()
meta = load_meta()
FEATURE_COLS = meta["feature_cols"]

# ---------------------------------------------------------------------------
# Sidebar — global filters (drive the Overview tab's drill-down)
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")
stores = sorted(sales_df["store"].unique().tolist())
categories = sorted(sales_df["category"].unique().tolist())

sel_stores = st.sidebar.multiselect("Store", stores, default=stores)
sel_categories = st.sidebar.multiselect("Category", categories, default=categories)

min_date, max_date = sales_df["date"].min().date(), sales_df["date"].max().date()
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
)

filtered = sales_df[
    (sales_df["store"].isin(sel_stores))
    & (sales_df["category"].isin(sel_categories))
    & (sales_df["date"].dt.date >= date_range[0])
    & (sales_df["date"].dt.date <= date_range[1])
]

st.sidebar.caption(f"{len(filtered):,} rows match current filters")

tab_overview, tab_forecast, tab_explain, tab_segment, tab_recs, tab_whatif = st.tabs(
    ["📊 Overview", "📈 Forecasting", "🔍 Explainability", "👥 Segmentation",
     "🧺 Recommendations", "🧪 What-If Predictor"]
)

# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------
with tab_overview:
    if filtered.empty:
        st.warning("No data matches the current filters — widen your selection in the sidebar.")
    else:
        daily_f = filtered.groupby("date", as_index=False).agg(
            units_sold=("units_sold", "sum"), revenue=("revenue", "sum")
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Units Sold", f"{daily_f['units_sold'].sum():,.0f}")
        col2.metric("Total Revenue", f"${daily_f['revenue'].sum():,.0f}")
        col3.metric("Avg Daily Units", f"{daily_f['units_sold'].mean():,.0f}")
        col4.metric("Days in Range", f"{len(daily_f):,}")

        fig = px.line(daily_f, x="date", y="units_sold", title="Daily Units Sold (filtered)")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, width='stretch')

        fig2 = px.line(daily_f, x="date", y="revenue", title="Daily Revenue (filtered)")
        fig2.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig2, width='stretch')

        col_a, col_b = st.columns(2)
        with col_a:
            by_store = filtered.groupby("store", as_index=False)["units_sold"].sum()
            st.plotly_chart(
                px.bar(by_store, x="store", y="units_sold", title="Units Sold by Store", color="store"),
                width='stretch',
            )
        with col_b:
            by_cat = filtered.groupby("category", as_index=False)["units_sold"].sum()
            st.plotly_chart(
                px.bar(by_cat, x="category", y="units_sold", title="Units Sold by Category", color="category"),
                width='stretch',
            )

        with st.expander("View filtered raw rows"):
            st.dataframe(filtered, width='stretch')
            st.download_button(
                "Download filtered data as CSV",
                filtered.to_csv(index=False).encode("utf-8"),
                file_name="retailiq_filtered_sales.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Forecasting tab — live horizon slider, re-forecasts on the fly
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Model Comparison")
    st.dataframe(comparison_df, width='stretch')
    best_model_name = comparison_df.iloc[0]["model"]
    st.success(f"Best performing model: **{best_model_name}** (highest R²)")

    fig_r2 = px.bar(comparison_df, x="model", y="R2", color="model", title="R² by Model")
    st.plotly_chart(fig_r2, width='stretch')

    st.divider()
    st.subheader("Live Forecast")

    horizon = st.slider("Forecast horizon (days ahead)", min_value=7, max_value=90, value=30, step=7)

    if "Prophet" in all_models:
        prophet_model = all_models["Prophet"]
        future = prophet_model.make_future_dataframe(periods=horizon)
        forecast = prophet_model.predict(future)

        daily_hist = features_df[["date", "units_sold"]].rename(columns={"units_sold": "actual"})
        forecast_plot = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=daily_hist["date"], y=daily_hist["actual"],
                                     mode="lines", name="Actual", line=dict(color="#1f77b4")))
        fig_fc.add_trace(go.Scatter(x=forecast_plot["date"], y=forecast_plot["yhat"],
                                     mode="lines", name="Forecast (Prophet)", line=dict(color="#ff7f0e")))
        fig_fc.add_trace(go.Scatter(
            x=list(forecast_plot["date"]) + list(forecast_plot["date"][::-1]),
            y=list(forecast_plot["yhat_upper"]) + list(forecast_plot["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)", line=dict(color="rgba(255,255,255,0)"),
            name="Confidence interval", showlegend=True,
        ))
        fig_fc.update_layout(title=f"Prophet Forecast — next {horizon} days", xaxis_title="Date", yaxis_title="Units Sold")
        st.plotly_chart(fig_fc, width='stretch')

        with st.expander("View forecast values"):
            st.dataframe(
                forecast_plot.tail(horizon).rename(
                    columns={"yhat": "forecast", "yhat_lower": "lower_bound", "yhat_upper": "upper_bound"}
                ),
                width='stretch',
            )
    else:
        st.info("Prophet model not found in models/all_models.pkl — re-run src/forecasting.py.")

    with st.expander("What do R² / RMSE / MAE mean?"):
        st.markdown(
            "- **R²**: proportion of variance in sales explained by the model (closer to 1 is better)\n"
            "- **RMSE**: root mean squared error — penalizes big misses more\n"
            "- **MAE**: mean absolute error — easier to interpret directly"
        )

# ---------------------------------------------------------------------------
# Explainability tab
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("SHAP Feature Importance")
    if importance_df is not None:
        top_n = st.slider("Show top N features", min_value=3, max_value=len(importance_df),
                           value=min(10, len(importance_df)))
        top_imp = importance_df.head(top_n)
        fig_imp = px.bar(top_imp, x="mean_abs_shap", y="feature", orientation="h",
                          title="Mean |SHAP value| by feature")
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, width='stretch')
    else:
        st.warning("Run `python src/explainability.py` to generate SHAP results.")

    col_a, col_b = st.columns(2)
    with col_a:
        summary_png = MODELS_DIR / "shap_summary.png"
        if summary_png.exists():
            st.image(str(summary_png), caption="Global SHAP summary (all test predictions)")
    with col_b:
        waterfall_png = MODELS_DIR / "shap_waterfall.png"
        if waterfall_png.exists():
            st.image(str(waterfall_png), caption="Waterfall: one prediction, broken into feature contributions")

    with st.expander("What is SHAP, in plain language?"):
        st.markdown(
            "SHAP shows how much each feature pushed a single prediction up or down relative "
            "to the average prediction. Try the **What-If Predictor** tab to see a live SHAP "
            "waterfall for a scenario you build yourself."
        )

# ---------------------------------------------------------------------------
# Segmentation tab — live K slider, reruns KMeans on the fly
# ---------------------------------------------------------------------------
with tab_segment:
    st.subheader("Live Re-clustering")
    k = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=4)

    rfm_cols = ["recency", "frequency", "monetary"]
    X = segments_df[rfm_cols]
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    live_clusters = km.fit_predict(X_scaled)

    live_df = segments_df.copy()
    live_df["live_cluster"] = live_clusters
    rank = live_df.groupby("live_cluster")["RFM_score"].mean().sort_values(ascending=False).index.tolist()
    label_pool = ["VIP", "Loyal", "Potential Loyalist", "Needs Attention", "At Risk", "Low Value", "Dormant", "New"]
    labels = label_pool[:k] if k <= len(label_pool) else [f"Segment {i}" for i in range(k)]
    label_map = {c: labels[i] for i, c in enumerate(rank)}
    live_df["segment"] = live_df["live_cluster"].map(label_map)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        seg_counts = live_df["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        st.plotly_chart(px.bar(seg_counts, x="segment", y="count", color="segment", title="Segment Sizes"),
                         width='stretch')
    with col_b:
        fig_scatter = px.scatter(
            live_df, x="recency", y="monetary", color="segment", size="frequency",
            hover_data=["customer_id", "frequency"],
            title="Recency vs Monetary (bubble size = frequency)",
        )
        st.plotly_chart(fig_scatter, width='stretch')

    st.subheader("Click a segment to filter the customer table")
    chosen_segment = st.radio("Segment", ["All"] + labels, horizontal=True)
    table = live_df if chosen_segment == "All" else live_df[live_df["segment"] == chosen_segment]
    st.dataframe(table[["customer_id", "recency", "frequency", "monetary", "RFM_score", "segment"]],
                 width='stretch')
    st.download_button(
        "Download this segment view as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="retailiq_segments.csv",
        mime="text/csv",
    )

    profile = live_df.groupby("segment")[rfm_cols].mean().round(1)
    st.subheader("Segment Profiles (mean R/F/M)")
    st.dataframe(profile, width='stretch')

    with st.expander("What is RFM / how is K chosen?"):
        st.markdown(
            "- **Recency**: days since last purchase (lower = more active)\n"
            "- **Frequency**: number of purchases\n"
            "- **Monetary**: total amount spent\n\n"
            "Drag the K slider above to see how segments split differently — in practice, K is "
            "chosen with the elbow method (see `models/elbow_plot.png`), but this live view lets "
            "you sanity-check alternative values."
        )

# ---------------------------------------------------------------------------
# Recommendations tab — live support/confidence sliders, reruns Apriori
# ---------------------------------------------------------------------------
with tab_recs:
    st.subheader("Live Association Rule Mining (Apriori)")
    transactions = load_baskets_raw()

    if transactions is None:
        st.info("No basket data found — run `python src/data_generation.py` and "
                 "`python src/recommendation.py` first.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            min_support = st.slider("Minimum support", 0.01, 0.20, 0.03, step=0.01)
        with col_b:
            min_confidence = st.slider("Minimum confidence", 0.10, 0.90, 0.30, step=0.05)

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_array, columns=te.columns_)
        frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.warning("No frequent itemsets at this support level — try lowering it.")
        else:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            if rules.empty:
                st.warning("No rules at this confidence level — try lowering it.")
            else:
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
                rules = rules.sort_values(["confidence", "support"], ascending=False).reset_index(drop=True)
                display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]

                st.dataframe(rules[display_cols], width='stretch')
                st.plotly_chart(
                    px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                               hover_data=["antecedents", "consequents"],
                               title="Rules: support vs confidence (bubble size/color = lift)"),
                    width='stretch',
                )

        with st.expander("What do support / confidence / lift mean?"):
            st.markdown(
                "- **Support**: how often this item combination appears across all baskets\n"
                "- **Confidence**: given the antecedent was bought, how often the consequent was too\n"
                "- **Lift**: how much more likely the consequent is given the antecedent, vs. random "
                "chance (lift > 1 means a real association)"
            )

# ---------------------------------------------------------------------------
# What-If Predictor tab — build a custom scenario, get a live prediction + SHAP
# ---------------------------------------------------------------------------
with tab_whatif:
    st.subheader("Build a Scenario")
    st.caption("Adjust the inputs below to simulate a day and get a live prediction from the XGBoost model.")

    if "XGBoost" not in all_models:
        st.warning("XGBoost model not found — re-run `python src/forecasting.py`.")
    else:
        xgb_model = all_models["XGBoost"]
        stats = features_df[FEATURE_COLS].describe()

        col1, col2, col3 = st.columns(3)
        with col1:
            day_of_week = st.selectbox(
                "Day of week", options=list(range(7)),
                format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x], index=5,
            )
            month = st.slider("Month", 1, 12, 12)
            is_weekend = 1 if day_of_week >= 5 else 0
        with col2:
            lag_1 = st.slider("Sales yesterday (lag_1)", int(stats.loc["min", "lag_1"]),
                               int(stats.loc["max", "lag_1"]), int(stats.loc["mean", "lag_1"]))
            lag_7 = st.slider("Sales 7 days ago (lag_7)", int(stats.loc["min", "lag_7"]),
                               int(stats.loc["max", "lag_7"]), int(stats.loc["mean", "lag_7"]))
            rolling_mean_7 = st.slider("7-day rolling avg", int(stats.loc["min", "rolling_mean_7"]),
                                        int(stats.loc["max", "rolling_mean_7"]), int(stats.loc["mean", "rolling_mean_7"]))
        with col3:
            is_month_start = st.checkbox("Month start")
            is_month_end = st.checkbox("Month end")
            is_promo_season = st.checkbox("Holiday season boost (Nov/Dec pattern)", value=(month in (11, 12)))

        row = {c: float(stats.loc["mean", c]) for c in FEATURE_COLS}
        row.update({
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
            "is_month_start": int(is_month_start),
            "is_month_end": int(is_month_end),
        })
        if is_promo_season:
            row["month"] = 12

        X_input = pd.DataFrame([row])[FEATURE_COLS]
        prediction = xgb_model.predict(X_input)[0]

        st.divider()
        st.metric("Predicted Units Sold", f"{prediction:,.0f}")

        explainer = get_shap_explainer(xgb_model)
        shap_val = explainer(X_input)

        contrib_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "shap_value": shap_val.values[0],
        }).sort_values("shap_value", key=abs, ascending=False).head(10)

        fig_contrib = px.bar(
            contrib_df, x="shap_value", y="feature", orientation="h",
            color="shap_value", color_continuous_scale="RdBu",
            title="Why this prediction: top feature contributions (SHAP)",
        )
        fig_contrib.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_contrib, width='stretch')

        st.caption(
            "Positive (red) values push the prediction up; negative (blue) values pull it down, "
            "relative to the average forecast."
        )

st.divider()
st.caption("RetailIQ · interactive dashboard — filters, live re-forecasting, live re-clustering, "
           "live rule mining, and a what-if predictor with on-demand SHAP explanations")
