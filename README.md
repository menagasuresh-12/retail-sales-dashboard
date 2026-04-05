# 📊 Retail Sales Analysis & Forecasting Dashboard

A full end-to-end data science project featuring data generation, cleaning, exploratory analysis, machine learning forecasting, and an interactive Streamlit dashboard — built entirely with free, open-source tools.

---

## 🗂 Project Structure

```
retail_dashboard/
├── data/                  ← Auto-generated (run setup.py first)
│   ├── raw_sales.csv      ← Synthetic retail transactions
│   ├── cleaned_sales.csv  ← After cleaning pipeline
│   ├── model.pkl          ← Trained Linear Regression model
│   ├── forecast.csv       ← 6-month revenue predictions
│   └── plots/             ← Static chart exports
│
├── src/
│   ├── generate_data.py   ← Synthetic dataset generator
│   ├── data_processing.py ← Cleaning & feature engineering
│   ├── eda.py             ← EDA, KPIs, static chart export
│   └── ml_model.py        ← ML training, evaluation, forecasting
│
├── app/
│   └── dashboard.py       ← Streamlit interactive dashboard
│
├── notebooks/
│   └── 01_eda_notebook.py ← VS Code-compatible notebook cells
│
├── setup.py               ← One-shot data + model setup script
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd retail_dashboard
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the one-shot setup (data + model)

```bash
python setup.py
```

This will:
- Generate 5,000 synthetic retail transactions (2021–2023)
- Clean the data (handle nulls, duplicates, type conversions)
- Train a Linear Regression forecasting model
- Save everything to `/data/`

### 5. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open your browser at **http://localhost:8501**

---

## 📋 Features

| Feature | Details |
|---|---|
| **Synthetic Dataset** | 5,000 rows · 5 categories · 5 regions · 3 years |
| **Data Cleaning** | Null imputation · duplicate removal · datetime parsing |
| **EDA** | Monthly trends · category/region analysis · correlation heatmap |
| **ML Model** | Linear Regression · MAE/MSE/R² evaluation · pickle saved |
| **Forecasting** | 6-month future revenue prediction with lag features |
| **Dashboard** | 4 pages · interactive Plotly charts · sidebar filters |
| **Filters** | Date range · Category · Region |

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| Data | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Persistence | pickle (model), CSV (data) |

---

## 💡 Key Business Insights

1. **Seasonality is significant** — Revenue spikes ~50% in Nov/Dec (holiday effect) and dips in Jan/Feb.
2. **Electronics** drives the highest revenue but has the tightest margins (8–20%).
3. **Clothing & Sports** deliver the best profit margins (30–55%), ideal for margin improvement.
4. **All 5 regions** are balanced — no single region dominates, suggesting uniform market coverage.
5. **Top 5 products** account for a disproportionate share of revenue (Pareto principle holds).
6. **Model R² > 0.85** confirms the seasonal + trend pattern is learnable from time features alone.

---

## 📸 Dashboard Pages

- **Overview** — KPI cards (revenue, profit, units, avg order value) + donut + bubble charts
- **Sales Analysis** — Monthly trend line · category grouped bars · seasonal heatmap
- **Product Insights** — Top 10 bar chart · treemap · sortable full product table
- **Forecast** — Actual vs predicted overlay · 6-month future projection · model metrics

---

## 🔧 Run Individual Modules (optional)

```bash
# Only generate data
python src/generate_data.py

# Only clean data
python src/data_processing.py

# Only run EDA (saves charts to data/plots/)
python src/eda.py

# Only train model
python src/ml_model.py
```

---

## 📄 License

MIT — free to use, modify, and distribute.
