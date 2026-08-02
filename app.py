"""
app.py — RetailIQ AI Dashboard
Streamlit entry point tying together sales KPIs, forecasting, SHAP
explainability, customer segmentation, and recommendations. Uses Plotly
for interactive charts and includes a sidebar filter + a simple what-if
forecast simulator.

Run:
    streamlit run app.py

Expects the pipeline to have already been run once (see README):
    python src/download_data.py
    python src/preprocessing.py
    python src/feature_engineering.py
    python src/forecasting.py
    python src/explainability.py
    python src/segmentation.py
    python src/recommendation.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="RetailIQ AI", layout="wide", page_icon="🛒")

PRIMARY = "#4F6AF6"
ACCENT = "#22C7A9"
PLOTLY_TEMPLATE = "plotly_white"

# Note: intentionally NOT using custom CSS for the KPI cards below. Custom
# `st.markdown(..., unsafe_allow_html=True)` CSS is fragile — it can end up
# invisible (white-on-white) depending on whether the viewer's Streamlit
# theme is light or dark, since it fights the theme instead of using it.
# `st.container(border=True)` is a built-in Streamlit primitive that always
# adapts correctly to the active theme.


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit reruns the whole script on every interaction,
# so anything reading from disk should be cached to keep the app snappy.
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(name, parse_dates=None):
    path = PROCESSED_DIR / name
    return pd.read_csv(path, parse_dates=parse_dates) if path.exists() else None


@st.cache_resource
def load_pickle(name):
    path = MODELS_DIR / name
    return joblib.load(path) if path.exists() else None


def files_missing():
    required = [
        PROCESSED_DIR / "daily_totals.csv",
        PROCESSED_DIR / "features.csv",
        PROCESSED_DIR / "sales_clean.csv",
        MODELS_DIR / "model_comparison.csv",
        PROCESSED_DIR / "customer_segments.csv",
    ]
    return [str(p) for p in required if not p.exists()]


def download_button(df: pd.DataFrame, label: str, filename: str, key: str):
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv", key=key,
    )


st.title("🛒 RetailIQ AI")
st.caption("Real transaction data (UCI Online Retail II) · Sales forecasting · SHAP explainability · Customer segmentation · Recommendations")

missing = files_missing()
if missing:
    st.error("Pipeline outputs not found. Run these once from the project root before launching the dashboard:")
    st.code(
        "python src/download_data.py\n"
        "python src/preprocessing.py\n"
        "python src/feature_engineering.py\n"
        "python src/forecasting.py\n"
        "python src/explainability.py\n"
        "python src/segmentation.py\n"
        "python src/recommendation.py",
        language="bash",
    )
    st.write("Missing files:")
    for m in missing:
        st.write(f"- `{m}`")
    st.stop()

daily_full = load_csv("daily_totals.csv", parse_dates=["date"])
sales_clean = load_csv("sales_clean.csv", parse_dates=["InvoiceDate"])
segments = load_csv("customer_segments.csv")

# ---------------------------------------------------------------------------
# Sidebar filters — date range + country. Only affects the Overview and
# Sales Dashboard tabs (segmentation/forecasting/recommendations are
# precomputed pipeline outputs and stay global, same as a real BI tool
# where models aren't retrained on every filter click).
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

min_date, max_date = daily_full["date"].min().date(), daily_full["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

countries = ["All"] + sorted(sales_clean["Country"].dropna().unique().tolist()) if sales_clean is not None else ["All"]
selected_country = st.sidebar.selectbox("Country", countries)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Filters apply to the Overview and Sales Dashboard tabs. Forecasting, "
    "Segmentation, and Recommendations reflect the full trained pipeline."
)

daily = daily_full[
    (daily_full["date"].dt.date >= start_date) & (daily_full["date"].dt.date <= end_date)
].copy()

sales_filtered = sales_clean
if sales_clean is not None:
    sales_filtered = sales_clean[
        (sales_clean["InvoiceDate"].dt.date >= start_date) & (sales_clean["InvoiceDate"].dt.date <= end_date)
    ]
    if selected_country != "All":
        sales_filtered = sales_filtered[sales_filtered["Country"] == selected_country]

# --- KPIs at top, visible on every tab ---
total_revenue = sales_filtered["TotalPrice"].sum() if sales_filtered is not None else daily["revenue"].sum()
total_units = sales_filtered["Quantity"].sum() if sales_filtered is not None else daily["units_sold"].sum()
n_orders = sales_filtered["InvoiceNo"].nunique() if sales_filtered is not None else daily["num_invoices"].sum()
aov = total_revenue / n_orders if n_orders else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1.container(border=True):
    st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
with k2.container(border=True):
    st.metric("📦 Units Sold", f"{total_units:,.0f}")
with k3.container(border=True):
    st.metric("🧾 Orders", f"{n_orders:,.0f}")
with k4.container(border=True):
    st.metric("💳 Avg Order Value", f"${aov:,.2f}")
with k5.container(border=True):
    st.metric("👥 Customers", f"{segments['CustomerID'].nunique():,}" if segments is not None else "—")

tab_overview, tab_sales, tab_top, tab_cohort, tab_forecast, tab_churn, tab_segment, tab_recs, tab_explain = st.tabs(
    ["📊 Overview", "💰 Sales Dashboard", "🏆 Top Performers", "📅 Cohort Retention", "📈 Forecasting",
     "⚠️ Churn Prediction", "👥 Customer Segmentation", "🧺 Recommendations", "🔍 SHAP Explainability"]
)

# --- Overview tab ---
with tab_overview:
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.line(daily, x="date", y="units_sold", template=PLOTLY_TEMPLATE,
                       title="Daily Units Sold", color_discrete_sequence=[PRIMARY])
        fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(daily, x="date", y="revenue", template=PLOTLY_TEMPLATE,
                        title="Daily Revenue", color_discrete_sequence=[ACCENT])
        fig2.update_layout(margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        if sales_filtered is not None and len(sales_filtered) > 0:
            by_country = (
                sales_filtered.groupby("Country")["TotalPrice"].sum()
                .sort_values(ascending=False).head(8).reset_index()
            )
            fig3 = px.pie(by_country, names="Country", values="TotalPrice", hole=0.5,
                           template=PLOTLY_TEMPLATE, title="Revenue by Country (top 8)")
            fig3.update_layout(margin=dict(t=50, l=10, r=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Quick facts**")
        st.markdown(
            f"- {len(daily):,} days in selected range\n"
            f"- {sales_filtered['Country'].nunique() if sales_filtered is not None else '—'} countries represented\n"
            f"- Avg {total_units / max(len(daily), 1):,.0f} units/day"
        )

    st.divider()
    st.subheader("📄 Executive Summary")
    summary_path = BASE_DIR / "reports" / "executive_summary.md"
    if summary_path.exists():
        st.caption("Auto-generated from this pipeline's own outputs — real numbers, not placeholders.")
        st.markdown(summary_path.read_text())
        st.download_button(
            "⬇ Download Executive Summary (Markdown)",
            summary_path.read_text().encode("utf-8"),
            file_name="executive_summary.md", mime="text/markdown", key="dl_summary",
        )
    else:
        st.info("Run `python src/generate_report.py` to generate a plain-English business summary here.")

# --- Sales Dashboard tab ---
with tab_sales:
    features = load_csv("features.csv", parse_dates=["date"])

    daily_aov = daily.assign(avg_order_value=(daily["revenue"] / daily["num_invoices"].replace(0, pd.NA)))
    fig = px.line(daily_aov, x="date", y="avg_order_value", template=PLOTLY_TEMPLATE,
                  title="Average Order Value Over Time", color_discrete_sequence=[PRIMARY])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(daily, x="date", y="num_invoices", template=PLOTLY_TEMPLATE,
                      title="Invoices per Day", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(daily, x="date", y="num_customers", template=PLOTLY_TEMPLATE,
                      title="Active Customers per Day", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, use_container_width=True)

    if features is not None:
        monthly = features.assign(year_month=features["date"].dt.to_period("M").astype(str))
        monthly_rev = monthly.groupby("year_month", as_index=False)["revenue"].sum()
        fig = px.bar(monthly_rev, x="year_month", y="revenue", template=PLOTLY_TEMPLATE,
                      title="Monthly Revenue", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)

    if sales_filtered is not None and len(sales_filtered) > 0:
        st.subheader("Top Countries by Revenue")
        top_countries = (
            sales_filtered.groupby("Country")["TotalPrice"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(top_countries, x="TotalPrice", y="Country", orientation="h",
                      template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        download_button(sales_filtered, "⬇ Download filtered transactions (CSV)", "filtered_transactions.csv", "dl_sales")

# --- Top Performers tab ---
with tab_top:
    st.caption(
        "Respects the Country and date-range filters in the sidebar — pick a country to drill "
        "into that market's top products and top customers, or leave it on \"All\" for the "
        "global picture."
    )

    if sales_filtered is None or len(sales_filtered) == 0:
        st.warning("No data matches the current filters.")
    else:
        scope_label = selected_country if selected_country != "All" else "All Countries"

        st.subheader(f"🌍 Top Areas by Revenue ({scope_label if selected_country != 'All' else 'global'})")
        top_area_df = (
            sales_filtered.groupby("Country")["TotalPrice"].sum()
            .sort_values(ascending=False).head(10).reset_index()
            .rename(columns={"TotalPrice": "revenue"})
        )
        if len(top_area_df) > 0:
            leader = top_area_df.iloc[0]
            st.info(f"📍 **{leader['Country']}** is the top-performing area, with "
                    f"**${leader['revenue']:,.0f}** in revenue in the current filter range.")
        fig_area = px.bar(top_area_df, x="revenue", y="Country", orientation="h",
                           template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY],
                           title="Revenue by Area")
        fig_area.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_area, use_container_width=True)

        st.divider()
        st.subheader(f"📦 Top Products in {scope_label}")
        col1, col2 = st.columns(2)
        with col1:
            top_products_qty = (
                sales_filtered.groupby("Description")["Quantity"].sum()
                .sort_values(ascending=False).head(10).reset_index()
            )
            fig_pq = px.bar(top_products_qty, x="Quantity", y="Description", orientation="h",
                             template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT],
                             title="By Units Sold")
            fig_pq.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_pq, use_container_width=True)
        with col2:
            top_products_rev = (
                sales_filtered.groupby("Description")["TotalPrice"].sum()
                .sort_values(ascending=False).head(10).reset_index()
                .rename(columns={"TotalPrice": "revenue"})
            )
            fig_pr = px.bar(top_products_rev, x="revenue", y="Description", orientation="h",
                             template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY],
                             title="By Revenue")
            fig_pr.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_pr, use_container_width=True)

        st.divider()
        st.subheader(f"👑 Top Customers in {scope_label}")
        if "CustomerID" in sales_filtered.columns:
            top_customers = (
                sales_filtered.groupby("CustomerID")
                .agg(revenue=("TotalPrice", "sum"),
                     orders=("InvoiceNo", "nunique"),
                     units=("Quantity", "sum"))
                .sort_values("revenue", ascending=False)
                .head(15)
                .reset_index()
            )
            top_customers["avg_order_value"] = (top_customers["revenue"] / top_customers["orders"]).round(2)
            st.dataframe(top_customers, use_container_width=True)
            download_button(top_customers, "⬇ Download top customers (CSV)",
                             f"top_customers_{scope_label.replace(' ', '_')}.csv", "dl_top_customers")

            fig_cust = px.bar(top_customers.head(10), x="revenue", y="CustomerID", orientation="h",
                               template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT],
                               title=f"Top 10 Customers by Revenue — {scope_label}")
            fig_cust.update_layout(yaxis=dict(autorange="reversed", type="category"))
            st.plotly_chart(fig_cust, use_container_width=True)
        else:
            st.info("Customer ID column not available in this data.")

# --- Cohort Retention tab ---
with tab_cohort:
    cohort_df = load_csv("cohort_retention.csv")
    st.subheader("Monthly Cohort Retention (%)")
    st.caption(
        "Each row is a group of customers who made their first purchase in that "
        "month. Each column shows what % of that group came back N months later."
    )
    if cohort_df is not None and len(cohort_df) > 0:
        heat_df = cohort_df.set_index("cohort_month")
        heat_df.columns = [f"Month {c}" for c in heat_df.columns]
        fig = px.imshow(
            heat_df, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE, title="Retention Heatmap",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cohort_df, use_container_width=True)
        download_button(cohort_df, "⬇ Download cohort table (CSV)", "cohort_retention.csv", "dl_cohort")
        with st.expander("Why cohort retention matters"):
            st.markdown(
                "It separates *how many new customers you're getting* from *how well "
                "you're keeping them* — two different problems that a single revenue "
                "number hides. A shrinking 'Month 1' column across cohorts is an early "
                "warning sign, well before total revenue actually drops."
            )
    else:
        st.info("Run `python src/cohort_analysis.py` to generate this table.")

# --- Forecasting tab ---
with tab_forecast:
    comparison_df = pd.read_csv(MODELS_DIR / "model_comparison.csv")

    st.subheader("Model Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    best_model_name = comparison_df.iloc[0]["model"]
    st.success(f"Best performing model: **{best_model_name}** (highest R²)")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(comparison_df, x="model", y="R2", template=PLOTLY_TEMPLATE,
                      title="R² by Model (higher is better)", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(comparison_df, x="model", y="RMSE", template=PLOTLY_TEMPLATE,
                      title="RMSE by Model (lower is better)", color_discrete_sequence=[ACCENT])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **R²**: proportion of variance in daily units sold explained by the model\n"
            "- **RMSE**: root mean squared error — penalizes big misses more\n"
            "- **MAE**: mean absolute error — easier to interpret directly in units"
        )

    st.divider()
    st.subheader("🔮 What-If Forecast Simulator")
    st.caption(
        "Keeps the most recent known trend (lag/rolling features) fixed, and shows "
        "how the best model's prediction changes if the *next* day falls on a "
        "different day of week or time of month — a quick sensitivity check, not a "
        "true multi-day forecast."
    )

    with open(MODELS_DIR / "best_model_name.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    best_model = load_pickle("best_model.pkl")
    features_df = load_csv("features.csv", parse_dates=["date"])

    if best_model is not None and features_df is not None and len(features_df) > 0:
        last_row = features_df.iloc[-1]

        col1, col2 = st.columns(2)
        with col1:
            sim_dow = st.selectbox(
                "Day of week", options=list(range(7)),
                format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
            )
            sim_weekend = 1 if sim_dow >= 5 else 0
        with col2:
            sim_month_start = st.checkbox("Is month start?", value=bool(last_row.get("is_month_start", 0)))
            sim_month_end = st.checkbox("Is month end?", value=bool(last_row.get("is_month_end", 0)))

        sim_row = last_row.copy()
        sim_row["day_of_week"] = sim_dow
        sim_row["is_weekend"] = sim_weekend
        sim_row["is_month_start"] = int(sim_month_start)
        sim_row["is_month_end"] = int(sim_month_end)

        X_sim = pd.DataFrame([sim_row[feature_cols]])
        prediction = best_model.predict(X_sim)[0]

        st.metric("Predicted units sold", f"{prediction:,.0f}",
                   delta=f"{prediction - last_row['units_sold']:+.0f} vs. last known day")
    else:
        st.info("Run the full pipeline first to enable the simulator.")

# --- Churn Prediction tab ---
with tab_churn:
    churn_df = load_csv("churn_predictions.csv")
    churn_comparison_path = MODELS_DIR / "churn_model_comparison.csv"

    if churn_df is None or not churn_comparison_path.exists():
        st.info("Run `python src/churn_prediction.py` to generate churn predictions.")
    else:
        churn_comparison = pd.read_csv(churn_comparison_path)

        st.caption(
            "Predicts which customers are likely to stop buying, using only "
            "behavior from *before* a cutoff date to predict what happens "
            "*after* it — the same temporal-holdout approach used for real "
            "churn models, not a same-day leak."
        )

        st.subheader("Model Comparison")
        st.dataframe(churn_comparison, use_container_width=True)
        best_churn_model = churn_comparison.iloc[0]["model"]
        st.success(f"Best model: **{best_churn_model}** (highest ROC-AUC)")

        col1, col2, col3 = st.columns(3)
        risk_counts = churn_df["risk_level"].value_counts()
        col1.metric("🔴 High Risk", f"{risk_counts.get('High', 0):,}")
        col2.metric("🟡 Medium Risk", f"{risk_counts.get('Medium', 0):,}")
        col3.metric("🟢 Low Risk", f"{risk_counts.get('Low', 0):,}")

        fig = px.histogram(churn_df, x="churn_probability", nbins=20, template=PLOTLY_TEMPLATE,
                            title="Churn Probability Distribution", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)

        importance_path = MODELS_DIR / "churn_feature_importance.csv"
        if importance_path.exists():
            churn_importance = pd.read_csv(importance_path)
            fig = px.bar(churn_importance, x="importance", y="feature", orientation="h",
                          template=PLOTLY_TEMPLATE, title="What drives the churn prediction?",
                          color_discrete_sequence=[ACCENT])
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("At-Risk Customers")
        risk_filter = st.selectbox("Filter by risk level", ["All", "High", "Medium", "Low"])
        display_df = churn_df if risk_filter == "All" else churn_df[churn_df["risk_level"] == risk_filter]
        st.dataframe(
            display_df[["CustomerID", "recency", "frequency", "monetary", "churn_probability", "risk_level"]],
            use_container_width=True,
        )
        download_button(display_df, "⬇ Download churn predictions (CSV)", "churn_predictions.csv", "dl_churn")

        potential_loss = display_df[display_df["risk_level"] == "High"]["monetary"].sum() if risk_filter == "All" else None
        if potential_loss:
            st.warning(f"💸 High-risk customers represent **${potential_loss:,.0f}** in historical spend — worth a retention campaign.")

        with st.expander("How is 'churn' defined here?"):
            st.markdown(
                "A customer is labeled **churned** if they made zero purchases in the "
                "final 90 days of the dataset, based on their RFM profile from *before* "
                "that window. This avoids the most common churn-modeling mistake — using "
                "information from the future to predict the future."
            )

# --- Customer Segmentation tab ---
with tab_segment:
    col1, col2 = st.columns([1, 1])
    with col1:
        seg_counts = segments["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.bar(seg_counts, x="segment", y="count", template=PLOTLY_TEMPLATE,
                      title="Segment Sizes", color="segment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(seg_counts, names="segment", values="count", hole=0.5,
                      template=PLOTLY_TEMPLATE, title="Segment Share")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profiles (mean R/F/M)")
    profile = segments.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1).reset_index()
    fig = px.bar(profile, x="segment", y="monetary", template=PLOTLY_TEMPLATE,
                  title="Average Monetary Value by Segment", color_discrete_sequence=[PRIMARY])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(profile, use_container_width=True)

    st.subheader("Customer-Level Detail")
    selected_segment = st.selectbox("Filter by segment", ["All"] + sorted(segments["segment"].unique().tolist()))
    detail_df = segments if selected_segment == "All" else segments[segments["segment"] == selected_segment]
    st.dataframe(detail_df, use_container_width=True)
    download_button(detail_df, "⬇ Download segment data (CSV)", "customer_segments.csv", "dl_segments")

    elbow_png = MODELS_DIR / "elbow_plot.png"
    if elbow_png.exists():
        with st.expander("How was the number of clusters (k) chosen?"):
            st.image(str(elbow_png), caption="Elbow method (inertia) + silhouette score by k")

    with st.expander("What is RFM?"):
        st.markdown(
            "- **Recency**: days since the customer's last purchase (lower = more recently active)\n"
            "- **Frequency**: number of distinct orders placed\n"
            "- **Monetary**: total amount spent\n\n"
            "Each customer is scored 1–5 on each dimension, then clustered with KMeans into "
            "business-friendly segments like Champions, Loyal Customers, and At Risk."
        )

# --- Recommendations tab ---
with tab_recs:
    rules_df = load_csv("association_rules.csv")
    popular_df = load_csv("popular_products.csv")
    revenue_df = load_csv("top_revenue_products.csv")
    segment_top_df = load_csv("segment_top_products.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Popular Products")
        if popular_df is not None:
            fig = px.bar(popular_df.head(10), x="Quantity", y="Description", orientation="h",
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Top Revenue Products")
        if revenue_df is not None:
            fig = px.bar(revenue_df.head(10), x="revenue", y="Description", orientation="h",
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Frequently Bought Together (Apriori)")
    if rules_df is not None and len(rules_df) > 0:
        st.dataframe(rules_df, use_container_width=True)
        download_button(rules_df, "⬇ Download association rules (CSV)", "association_rules.csv", "dl_rules")
        with st.expander("What do support / confidence / lift mean?"):
            st.markdown(
                "- **Support**: how often this item combination appears across all baskets\n"
                "- **Confidence**: given the antecedent was bought, how often the consequent was too\n"
                "- **Lift**: how much more likely the consequent is given the antecedent, vs. random "
                "chance (lift > 1 means a real association)"
            )
    else:
        st.info("No association rules found at the current support/confidence thresholds.")

    if segment_top_df is not None and len(segment_top_df) > 0:
        st.subheader("Top Products by Customer Segment")
        chosen = st.selectbox("Segment", sorted(segment_top_df["segment"].unique().tolist()), key="seg_top")
        st.dataframe(segment_top_df[segment_top_df["segment"] == chosen], use_container_width=True)

# --- SHAP Explainability tab ---
with tab_explain:
    importance_path = MODELS_DIR / "shap_feature_importance.csv"
    importance_df = pd.read_csv(importance_path) if importance_path.exists() else None

    st.subheader("SHAP Feature Importance")
    if importance_df is not None:
        fig = px.bar(importance_df.head(15), x="mean_abs_shap", y="feature", orientation="h",
                      template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY],
                      title="Mean |SHAP value| per feature")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)
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
            "SHAP (SHapley Additive exPlanations) shows how much each feature pushed a "
            "single prediction up or down relative to the average prediction. It turns a "
            "'black box' model into something a business user can question and trust."
        )

st.divider()
st.caption("RetailIQ AI · Real transaction data (Online Retail II, UCI) · Streamlit + Plotly dashboard")
