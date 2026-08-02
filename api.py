"""
api.py — RetailIQ AI REST API
A small, beginner-readable FastAPI layer over the same pipeline outputs the
Streamlit dashboard reads. This exists to demonstrate backend/API skills on
top of the data science pipeline — useful if you're presenting this project
for a Backend or Data Analyst role, not just a pure DS role.

It's intentionally simple: no auth, no database, no background jobs — just
read-only endpoints over the CSV/pickle files the pipeline already produces.
That's a deliberate choice for a portfolio project (see PROJECT_NOTES.md).

Run:
    uvicorn api:app --reload
Then open:
    http://127.0.0.1:8000/docs   (interactive Swagger UI, auto-generated)
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="RetailIQ AI API",
    description="Read-only REST API over the RetailIQ AI analytics pipeline "
                "(sales KPIs, forecasting, customer segments, recommendations).",
    version="1.0.0",
)

# Wide-open CORS since this is a local portfolio demo, not a real deployment.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def read_csv_or_404(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found — run the pipeline first (see README).",
        )
    return pd.read_csv(path)


@app.get("/")
def root():
    return {
        "message": "RetailIQ AI API is running.",
        "docs": "/docs",
        "endpoints": [
            "/health", "/kpis", "/sales/daily", "/forecast/models",
            "/forecast/predict", "/segments", "/segments/{segment_name}",
            "/recommendations/popular", "/recommendations/top-revenue",
            "/recommendations/association-rules",
        ],
    }


@app.get("/health")
def health():
    """Simple liveness check — also reports whether the pipeline has been run."""
    required = ["daily_totals.csv", "customer_segments.csv"]
    missing = [f for f in required if not (PROCESSED_DIR / f).exists()]
    return {"status": "ok", "pipeline_ready": len(missing) == 0, "missing_files": missing}


@app.get("/kpis")
def get_kpis():
    """Top-line business KPIs — the same numbers shown at the top of the dashboard."""
    daily = read_csv_or_404("daily_totals.csv")
    segments = read_csv_or_404("customer_segments.csv")
    return {
        "total_revenue": round(float(daily["revenue"].sum()), 2),
        "total_units_sold": int(daily["units_sold"].sum()),
        "total_orders": int(daily["num_invoices"].sum()),
        "days_of_data": len(daily),
        "identified_customers": int(segments["CustomerID"].nunique()),
    }


@app.get("/sales/daily")
def get_daily_sales(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """Daily units/revenue time series, optionally filtered by date range."""
    daily = read_csv_or_404("daily_totals.csv")
    if start_date:
        daily = daily[daily["date"] >= start_date]
    if end_date:
        daily = daily[daily["date"] <= end_date]
    return daily.to_dict(orient="records")


@app.get("/forecast/models")
def get_model_comparison():
    """R2/RMSE/MAE for every forecasting model that was trained and compared."""
    path = MODELS_DIR / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run src/forecasting.py first.")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/forecast/predict")
def predict_units_sold(day_of_week: int = Query(..., ge=0, le=6, description="0=Mon ... 6=Sun")):
    """Predicts units sold for a hypothetical day, keeping the most recent
    known lag/rolling trend fixed and only varying day-of-week — a simple
    sensitivity check, matching the dashboard's What-If Simulator."""
    model_path = MODELS_DIR / "best_model.pkl"
    meta_path = MODELS_DIR / "best_model_name.json"
    features_path = PROCESSED_DIR / "features.csv"
    if not (model_path.exists() and meta_path.exists() and features_path.exists()):
        raise HTTPException(status_code=404, detail="Run the full pipeline first (see README).")

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    features_df = pd.read_csv(features_path)
    last_row = features_df.iloc[-1].copy()
    last_row["day_of_week"] = day_of_week
    last_row["is_weekend"] = 1 if day_of_week >= 5 else 0

    X = pd.DataFrame([last_row[feature_cols]])
    prediction = float(model.predict(X)[0])

    return {
        "model_used": meta["best_model"],
        "day_of_week": day_of_week,
        "predicted_units_sold": round(prediction, 1),
    }


@app.get("/segments")
def get_segment_summary():
    """Segment sizes and average RFM profile for each customer segment."""
    segments = read_csv_or_404("customer_segments.csv")
    summary = (
        segments.groupby("segment")
        .agg(customers=("CustomerID", "nunique"),
             avg_recency=("recency", "mean"),
             avg_frequency=("frequency", "mean"),
             avg_monetary=("monetary", "mean"))
        .round(1).reset_index()
    )
    return summary.to_dict(orient="records")


@app.get("/segments/{segment_name}")
def get_customers_in_segment(segment_name: str):
    """All customers belonging to one named segment (e.g. 'Champions')."""
    segments = read_csv_or_404("customer_segments.csv")
    result = segments[segments["segment"].str.lower() == segment_name.lower()]
    if result.empty:
        raise HTTPException(status_code=404, detail=f"No segment named '{segment_name}' found.")
    return result.to_dict(orient="records")


@app.get("/recommendations/popular")
def get_popular_products(top_n: int = Query(10, ge=1, le=100)):
    df = read_csv_or_404("popular_products.csv")
    return df.head(top_n).to_dict(orient="records")


@app.get("/recommendations/top-revenue")
def get_top_revenue_products(top_n: int = Query(10, ge=1, le=100)):
    df = read_csv_or_404("top_revenue_products.csv")
    return df.head(top_n).to_dict(orient="records")


@app.get("/recommendations/association-rules")
def get_association_rules(min_confidence: float = Query(0.2, ge=0, le=1)):
    df = read_csv_or_404("association_rules.csv")
    return df[df["confidence"] >= min_confidence].to_dict(orient="records")
