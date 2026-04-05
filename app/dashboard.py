"""
dashboard.py  —  Step 5: Streamlit Interactive Dashboard
---------------------------------------------------------
Run with:
    streamlit run app/dashboard.py

Pages:
  1. Overview      — KPI cards
  2. Sales Analysis— Time series, category & region charts
  3. Product Insights — Top / bottom products
  4. Forecast      — Actual vs predicted revenue

All charts use Plotly for interactivity.
Filters (sidebar): date range, category, region.
"""

import sys
from pathlib import Path

# ── Make src/ importable from app/ ───────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEANED_PATH  = ROOT / "data" / "cleaned_sales.csv"
MODEL_PATH    = ROOT / "data" / "model.pkl"
FORECAST_PATH = ROOT / "data" / "forecast.csv"


# ═════════════════════════════════════════════════════════════════════════════
#  Data & Model Loading  (cached so re-runs are instant)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load cleaned sales data. Cached so it's only read once."""
    df = pd.read_csv(CLEANED_PATH, parse_dates=["date"])
    return df


@st.cache_resource
def load_model():
    """Load trained model bundle. Cached as a resource (not serialised)."""
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]


@st.cache_data
def load_forecast() -> pd.DataFrame:
    return pd.read_csv(FORECAST_PATH)


# ═════════════════════════════════════════════════════════════════════════════
#  Page Configuration
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS — subtle background, card shadow
st.markdown("""
<style>
    .kpi-card {
        background: #f8f9fa;
        border-left: 4px solid #2196F3;
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 13px; color: #666; margin-bottom: 4px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
    .kpi-delta { font-size: 12px; color: #4CAF50; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Sidebar — Navigation + Filters
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Retail Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        ["🏠 Overview", "📈 Sales Analysis", "🛍 Product Insights", "🔮 Forecast"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")

    df_full = load_data()

    # Date range filter
    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )

    # Category filter
    categories    = ["All"] + sorted(df_full["category"].unique().tolist())
    sel_category  = st.selectbox("Category", categories)

    # Region filter
    regions    = ["All"] + sorted(df_full["region"].unique().tolist())
    sel_region = st.selectbox("Region", regions)

    st.markdown("---")
    st.caption("Data: 2021–2023 synthetic retail")


# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full.copy()

if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]

if sel_category != "All":
    df = df[df["category"] == sel_category]

if sel_region != "All":
    df = df[df["region"] == sel_region]


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: KPI card
# ═════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, delta: str = "") -> None:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1: Overview
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🏠 Business Overview")
    st.caption("High-level KPIs for the selected filters.")
    st.markdown("---")

    # Compute KPIs
    total_rev    = df["sales_amount"].sum()
    total_profit = df["profit"].sum()
    total_qty    = df["quantity"].sum()
    avg_margin   = df["profit_margin"].mean()
    total_orders = len(df)
    avg_order    = df["sales_amount"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Total Revenue",     f"${total_rev:,.0f}",    "All-time filtered")
        kpi_card("Total Orders",      f"{total_orders:,}",     "Transactions")
    with c2:
        kpi_card("Total Profit",      f"${total_profit:,.0f}", "Gross profit")
        kpi_card("Avg Order Value",   f"${avg_order:,.2f}",    "Per transaction")
    with c3:
        kpi_card("Units Sold",        f"{total_qty:,}",        "Total quantity")
        kpi_card("Avg Profit Margin", f"{avg_margin:.1f}%",    "Gross margin")

    st.markdown("---")

    # Revenue split by category — donut chart
    col_a, col_b = st.columns(2)
    with col_a:
        cat_rev = df.groupby("category")["sales_amount"].sum().reset_index()
        fig = px.pie(cat_rev, names="category", values="sales_amount",
                     hole=0.45, title="Revenue Split by Category",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Profit by region — horizontal bar
        reg_profit = df.groupby("region")["profit"].sum().sort_values().reset_index()
        fig2 = px.bar(reg_profit, x="profit", y="region", orientation="h",
                      title="Profit by Region", color="profit",
                      color_continuous_scale="Blues", text_auto="$.3s")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2: Sales Analysis
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈 Sales Analysis":
    st.title("📈 Sales Analysis")
    st.caption("Time-series trends, category performance, and regional breakdown.")
    st.markdown("---")

    # Monthly revenue line chart
    monthly = (df.groupby("year_month")["sales_amount"]
                 .sum()
                 .reset_index()
                 .rename(columns={"sales_amount": "Revenue"}))
    monthly = monthly.sort_values("year_month")

    fig = px.line(monthly, x="year_month", y="Revenue",
                  title="Monthly Revenue Trend",
                  markers=True, line_shape="spline",
                  color_discrete_sequence=["#2196F3"])
    fig.update_layout(xaxis_title="Month", yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Category bar chart — revenue + profit side by side
        cat_df = (df.groupby("category")[["sales_amount", "profit"]]
                    .sum()
                    .reset_index()
                    .sort_values("sales_amount", ascending=False))
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Revenue", x=cat_df["category"],
                              y=cat_df["sales_amount"], marker_color="#2196F3"))
        fig2.add_trace(go.Bar(name="Profit",  x=cat_df["category"],
                              y=cat_df["profit"], marker_color="#4CAF50"))
        fig2.update_layout(barmode="group", title="Category: Revenue vs Profit",
                           yaxis_tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Region scatter — revenue vs profit
        reg_df = df.groupby("region")[["sales_amount", "profit", "quantity"]].sum().reset_index()
        fig3 = px.scatter(reg_df, x="sales_amount", y="profit",
                          size="quantity", color="region", text="region",
                          title="Region: Revenue vs Profit (bubble = qty)",
                          color_discrete_sequence=px.colors.qualitative.Vivid)
        fig3.update_traces(textposition="top center")
        fig3.update_layout(xaxis_tickprefix="$", yaxis_tickprefix="$")
        st.plotly_chart(fig3, use_container_width=True)

    # Monthly heatmap — revenue by month and year
    st.markdown("---")
    st.subheader("Seasonal Heatmap")
    heat_df = (df.groupby(["year", "month"])["sales_amount"]
                 .sum()
                 .reset_index())
    # Pivot: rows=year, cols=month
    heat_pivot = heat_df.pivot(index="year", columns="month", values="sales_amount").fillna(0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    heat_pivot.columns = [month_labels[m-1] for m in heat_pivot.columns]

    fig4 = px.imshow(heat_pivot, text_auto="$.3s",
                     color_continuous_scale="Blues",
                     title="Monthly Revenue Heatmap (Year × Month)")
    fig4.update_layout(coloraxis_colorbar=dict(title="Revenue $"))
    st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3: Product Insights
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🛍 Product Insights":
    st.title("🛍 Product Insights")
    st.caption("Identify your stars and underperformers.")
    st.markdown("---")

    prod_df = (df.groupby(["product", "category"])
                 .agg(revenue=("sales_amount", "sum"),
                      profit=("profit", "sum"),
                      quantity=("quantity", "sum"),
                      margin=("profit_margin", "mean"))
                 .reset_index()
                 .sort_values("revenue", ascending=False))

    # Top 10 by revenue
    top10 = prod_df.head(10)
    fig = px.bar(top10, x="revenue", y="product", orientation="h",
                 color="category", title="Top 10 Products by Revenue",
                 text_auto="$.3s",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      xaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Bottom 5 — warning table
        st.subheader("⚠️ Low-Performing Products")
        bottom5 = prod_df.tail(5)[["product", "category", "revenue", "quantity"]]
        bottom5["revenue"] = bottom5["revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(bottom5.reset_index(drop=True), use_container_width=True)

    with col2:
        # Margin leaders
        st.subheader("💰 Highest Margin Products")
        top_margin = prod_df.nlargest(5, "margin")[["product", "category", "margin", "revenue"]]
        top_margin["margin"]  = top_margin["margin"].apply(lambda x: f"{x:.1f}%")
        top_margin["revenue"] = top_margin["revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(top_margin.reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # Treemap — all products sized by revenue, coloured by margin
    fig2 = px.treemap(prod_df,
                      path=["category", "product"],
                      values="revenue",
                      color="margin",
                      color_continuous_scale="RdYlGn",
                      title="Product Portfolio Treemap (size=revenue, color=margin%)")
    st.plotly_chart(fig2, use_container_width=True)

    # Full sortable table
    st.subheader("Complete Product Table")
    display_df = prod_df.copy()
    display_df["revenue"]  = display_df["revenue"].apply(lambda x: f"${x:,.2f}")
    display_df["profit"]   = display_df["profit"].apply(lambda x: f"${x:,.2f}")
    display_df["margin"]   = display_df["margin"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4: Forecast
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Forecast":
    st.title("🔮 Sales Forecast")
    st.caption("Linear Regression model trained on monthly revenue — actual vs predicted + future outlook.")
    st.markdown("---")

    # ── Reconstruct the same monthly series the model was trained on ──────────
    from ml_model import load_monthly_series, engineer_features, load_model as _load_model

    monthly    = load_monthly_series(CLEANED_PATH)
    monthly_fe = engineer_features(monthly)

    model, scaler = load_model()

    FEATURES = ["t", "year", "sin_month", "cos_month", "lag_1", "lag_2"]
    monthly_fe["predicted"] = model.predict(
        scaler.transform(monthly_fe[FEATURES])
    )

    # ── Actual vs Predicted chart ─────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_fe["year_month"], y=monthly_fe["revenue"],
        mode="lines+markers", name="Actual Revenue",
        line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(
        x=monthly_fe["year_month"], y=monthly_fe["predicted"],
        mode="lines+markers", name="Predicted Revenue",
        line=dict(color="#FF5722", width=2, dash="dash")))
    fig.update_layout(
        title="Actual vs Predicted Monthly Revenue",
        xaxis_title="Month",
        yaxis_tickprefix="$",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Model metrics ─────────────────────────────────────────────────────────
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    split_idx = int(len(monthly_fe) * 0.8)
    y_test    = monthly_fe["revenue"].iloc[split_idx:]
    y_pred    = monthly_fe["predicted"].iloc[split_idx:]

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    st.markdown("### Model Metrics (test set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"${mae:,.2f}", help="Mean Absolute Error — avg $ off per month")
    m2.metric("RMSE", f"${np.sqrt(mse):,.2f}", help="Root Mean Squared Error")
    m3.metric("R²",   f"{r2:.4f}", help="1.0 = perfect fit; >0.85 is excellent")

    st.markdown("---")

    # ── Future 6-month forecast ───────────────────────────────────────────────
    forecast = load_forecast()
    st.subheader("📅 Next 6 Months Forecast")

    fig2 = go.Figure()
    # Historical (last 12 months as context)
    last12 = monthly_fe.tail(12)
    fig2.add_trace(go.Scatter(
        x=last12["year_month"], y=last12["revenue"],
        mode="lines+markers", name="Historical",
        line=dict(color="#2196F3", width=2)))
    # Future
    fig2.add_trace(go.Scatter(
        x=forecast["year_month"], y=forecast["predicted_revenue"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#FF9800", width=2, dash="dot"),
        marker=dict(symbol="star", size=10)))
    fig2.update_layout(
        title="Revenue Forecast — Next 6 Months",
        xaxis_title="Month",
        yaxis_tickprefix="$",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Forecast table
    fc_display = forecast.copy()
    fc_display["predicted_revenue"] = fc_display["predicted_revenue"].apply(lambda x: f"${x:,.2f}")
    fc_display.columns = ["Month", "Predicted Revenue"]
    st.dataframe(fc_display, use_container_width=True, hide_index=True)

    st.info("ℹ️ Model: Linear Regression with time-index, cyclic month encoding, and 2-month lag features.")
