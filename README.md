# 🛒 RetailIQ AI

**Full-stack Retail Data Science pipeline — from raw sales data to AI-powered business insights, all in one interactive dashboard.**

**Project Type:** College Minor Project (Data Science)
**Team:** Origin Point
**Status:** ✅ Completed

---

# 📌 Overview

RetailIQ AI is an end-to-end retail analytics platform that transforms raw retail transaction data into actionable business insights.

The project automatically generates or processes retail sales, customer, and basket transaction data, cleans and prepares it for machine learning, engineers predictive features, forecasts future sales using multiple ML models, explains model predictions using SHAP, segments customers using RFM + KMeans clustering, and recommends products using Market Basket Analysis.

All outputs are presented through a modern interactive Streamlit dashboard.

---

# 🚨 Problem Statement

Retail businesses generate large volumes of transactional data, but much of it remains underutilized.

Most businesses struggle to:

* Identify future sales trends
* Understand customer value
* Segment customers effectively
* Discover cross-selling opportunities
* Explain AI model predictions

RetailIQ AI automates the complete analytics workflow so businesses can make data-driven decisions without manual analysis.

---

# 🎯 Project Objectives

* Generate realistic synthetic retail datasets
* Clean and preprocess raw retail data
* Engineer time-series features
* Forecast future sales using multiple ML models
* Explain predictions with SHAP Explainable AI
* Segment customers using RFM + KMeans
* Recommend products using Apriori Market Basket Analysis
* Visualize everything inside one interactive dashboard

---

# 🏗️ System Architecture

```
Raw Data Generation (Daksh)
        │
        ▼
Data Preprocessing (Akshay)
        │
        ▼
Feature Engineering (Anurag)
        │
        ▼
Forecasting Models
(Linear Regression
 Random Forest
 XGBoost
 Prophet)
        │
        ▼
SHAP Explainability
        │
        ▼
Customer Segmentation
(RFM + KMeans)
        │
        ▼
Product Recommendation
(Apriori Algorithm)
        │
        ▼
Interactive Streamlit Dashboard
        │
        ▼
✅ RetailIQ AI
```

---

# 📁 Project Structure

```text
retailiq-ai/
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── explainability.py
│   ├── segmentation.py
│   └── recommendation.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── reports/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_NOTES.md
└── .gitignore
```

---

# ⚙️ Analytics Pipeline

| Step | Script                   | Purpose                                                                 |
| ---- | ------------------------ | ----------------------------------------------------------------------- |
| 1    | `data_generation.py`     | Generate synthetic retail datasets                                      |
| 2    | `preprocessing.py`       | Clean data, remove duplicates, fill missing values, cap outliers        |
| 3    | `feature_engineering.py` | Create lag, rolling and calendar features                               |
| 4    | `forecasting.py`         | Train and compare Linear Regression, Random Forest, XGBoost and Prophet |
| 5    | `explainability.py`      | Generate SHAP summary and waterfall plots                               |
| 6    | `segmentation.py`        | Perform RFM analysis and KMeans clustering                              |
| 7    | `recommendation.py`      | Generate Apriori association rules                                      |
| 8    | `app.py`                 | Launch the interactive Streamlit dashboard                              |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/retailiq-ai.git
cd retailiq-ai
```

## 2. Create a virtual environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```cmd
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Verify installation

```bash
python -c "import pandas, sklearn, xgboost, prophet, shap, streamlit; print('All dependencies installed successfully!')"
```

---

# ▶️ Running the Project

Run the scripts in the following order:

```bash
python src/data_generation.py
python src/preprocessing.py
python src/feature_engineering.py
python src/forecasting.py
python src/explainability.py
python src/segmentation.py
python src/recommendation.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

---

# 🖥️ Dashboard Features

The Streamlit dashboard provides:

* Interactive sales overview
* Store and category filters
* Forecast visualization
* SHAP Explainable AI
* Customer segmentation
* Product recommendations
* What-if prediction simulator
* Interactive Plotly charts
* CSV export functionality

---

# 🛠️ Tech Stack

| Category              | Technology                  |
| --------------------- | --------------------------- |
| Language              | Python 3                    |
| Data Processing       | Pandas, NumPy               |
| Machine Learning      | Scikit-learn                |
| Forecasting           | XGBoost, Prophet            |
| Explainability        | SHAP                        |
| Customer Segmentation | RFM, KMeans                 |
| Recommendation System | Apriori (mlxtend)           |
| Visualization         | Plotly, Matplotlib, Seaborn |
| Dashboard             | Streamlit                   |
| Notebook              | Jupyter Notebook            |

---

# 👥 Team

| Member     | Responsibility                                                        |
| ---------- | --------------------------------------------------------------------- |
| **Akshay** | Data preprocessing, Streamlit dashboard integration, backend pipeline |
| **Anurag** | Feature engineering, forecasting models, SHAP explainability          |
| **Daksh**  | Data generation, customer segmentation, recommendation system         |

---

# 🔮 Future Improvements

* Real-world retail dataset integration
* Live sales dashboard
* Multi-store analytics
* Customer churn prediction
* Deep learning forecasting (LSTM / Transformer)
* Automated PDF report generation
* Cloud deployment

---

# 📄 License

This project is released under the **MIT License**.

Feel free to use, modify, and distribute it for educational or research purposes.
