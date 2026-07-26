# 🛒 RetailIQ AI

Full-stack Retail Data Science pipeline — from raw sales data to AI-powered business insights, all in one interactive dashboard.

**Project Type:** College Minor Project (Data Science)
**Team:** Origin Point
**Status:** Completed ✅

## 📌 What is this?

RetailIQ AI is an end-to-end retail analytics system that takes raw sales, customer, and basket transaction data and turns it into actionable business intelligence. It cleans and processes data, engineers time-series features, forecasts future sales using multiple ML models, explains predictions with SHAP, segments customers using RFM + KMeans, and recommends products using Market Basket Analysis — all visualized in a single Streamlit dashboard.

## 🚨 Problem Statement

Retail businesses generate huge volumes of transactional data but rarely use it beyond basic reporting. Sales trends go unnoticed, customer value isn't quantified, and product bundling opportunities are missed. RetailIQ AI automates the full analytics pipeline so businesses can forecast demand, identify high-value customers, and boost cross-selling — without manual analysis.

## 🎯 Project Goals

- Clean and standardize raw retail data for downstream analysis
- Engineer meaningful time-series features (lags, rolling stats, calendar effects)
- Forecast future sales using and comparing multiple ML models
- Explain model predictions using SHAP for transparency
- Segment customers using RFM analysis + KMeans clustering
- Recommend frequently co-purchased products using Apriori
- Present everything in a single interactive dashboard

## 🏗️ System Architecture



Raw Data Generation (Daksh)
        │
        ▼
Data Preprocessing (Akshay)
        │
        ▼
Feature Engineering (Anurag)
        │
        ▼
Forecasting Models (Anurag)
        │
        ▼
SHAP Explainability (Anurag)
        │
        ▼
Customer Segmentation — RFM + KMeans (Daksh)
        │
        ▼
Recommendation System — Apriori (Daksh)
        │
        ▼
Streamlit Dashboard Integration (Akshay)
        │
        ▼
✅ RetailIQ AI — Live Dashboard


## 📁 Project Structure


RetailIQ-AI/
├── src/
│   ├── data_generation.py       # Synthetic retail dataset generation
│   ├── preprocessing.py         # Data cleaning pipeline
│   ├── feature_engineering.py   # Lag, rolling, calendar features
│   ├── forecasting.py           # Multi-model sales forecasting
│   ├── explainability.py        # SHAP-based model explanation
│   ├── segmentation.py          # RFM + KMeans customer segmentation
│   └── recommendation.py        # Apriori market basket analysis
├── data/
│   ├── raw/
│   │   ├── sales_data.csv
│   │   ├── customer_transactions.csv
│   │   └── basket_transactions.csv
│   └── processed/
│       ├── sales_clean.csv
│       ├── daily_totals.csv
│       ├── features.csv
│       ├── customer_segments.csv
│       └── association_rules.csv
├── models/
│   ├── best_model.pkl
│   ├── all_models.pkl
│   ├── best_model_name.json
│   ├── model_comparison.csv
│   ├── shap_values.pkl
│   ├── shap_feature_importance.csv
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_testing.ipynb
├── reports/                     # Graphs, screenshots, analysis reports
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md



## ⚙️ The Analytics Pipeline

| Step | Script | Action |
|------|--------|--------|
| 1 | `data_generation.py` | Generate synthetic sales, customer & basket datasets |
| 2 | `preprocessing.py` | Remove duplicates, handle missing values, cap outliers, aggregate daily totals |
| 3 | `feature_engineering.py` | Create lag features, rolling stats, calendar features |
| 4 | `forecasting.py` | Train & compare Linear Regression, Random Forest, XGBoost, Prophet |
| 5 | `explainability.py` | Generate SHAP summary & waterfall plots for the best model |
| 6 | `segmentation.py` | RFM analysis + KMeans clustering → VIP / Loyal / At Risk labels |
| 7 | `recommendation.py` | Apriori algorithm → frequently bought-together product rules |
| 8 | `app.py` | Streamlit dashboard tying everything together |

## 🖥️ Installation & Running

bash
# 1. Clone the repo
git clone https://github.com/<your-org>/RetailIQ-AI
cd RetailIQ-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python src/data_generation.py
python src/preprocessing.py
python src/feature_engineering.py
python src/forecasting.py
python src/explainability.py
python src/segmentation.py
python src/recommendation.py

# 4. Launch the dashboard
streamlit run app.py


## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost, Prophet |
| Explainability | SHAP |
| Clustering | KMeans, StandardScaler |
| Market Basket Analysis | Apriori (mlxtend) |
| Dashboard | Streamlit + Plotly |
| Notebooks | Jupyter (EDA & experimentation) |

## 👥 Team — Origin Point

| Owner | Responsibility | GitHub | Files/Folders |
|-------|----------------|--------|----------------|
| **Akshay** | Backend Integration + Dashboard + Data Processing | [@Akshay-23A](https://github.com/Akshay-23A) | `preprocessing.py`, `app.py`, `data/processed/`, README, requirements |
| **Anurag** | ML Pipeline — Forecasting + Explainability | [@Anurag678-coder](https://github.com/Anurag678-coder) | `feature_engineering.py`, `forecasting.py`, `explainability.py`, `models/` |
| **Daksh** | Data Generation + Customer Analytics |  | `data_generation.py`, `segmentation.py`, `recommendation.py`, `data/raw/`, `reports/` |

## 🔮 Future Directions

- Real retail dataset integration (beyond synthetic data)
- Deep learning-based forecasting (LSTM/Transformer)
- Real-time dashboard updates via live data feed
- Multi-store / multi-region comparative analytics
- Customer churn prediction module
- Automated report generation (PDF export)

## License

This project is provided under the MIT License – feel free to use, modify, and distribute.

