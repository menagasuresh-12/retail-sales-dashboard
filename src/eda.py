"""
eda.py  —  Step 3: Exploratory Data Analysis
--------------------------------------------
Run after data_processing.py:
    python src/eda.py

What it does:
  - Computes high-level KPIs (revenue, profit, quantity)
  - Analyses trends by month, category, region, and product
  - Saves 4 chart images to data/plots/ for reference
  - Prints business insights as comments in the output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path(__file__).parent.parent / "data" / "cleaned_sales.csv"
PLOTS_DIR  = Path(__file__).parent.parent / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# KPI Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_kpis(df: pd.DataFrame) -> None:
    total_revenue  = df["sales_amount"].sum()
    total_profit   = df["profit"].sum()
    total_qty      = df["quantity"].sum()
    avg_margin     = df["profit_margin"].mean()

    print("=" * 55)
    print("  KEY PERFORMANCE INDICATORS")
    print("=" * 55)
    print(f"  Total Revenue      : ${total_revenue:>12,.2f}")
    print(f"  Total Profit       : ${total_profit:>12,.2f}")
    print(f"  Total Units Sold   : {total_qty:>13,}")
    print(f"  Avg Profit Margin  : {avg_margin:>12.1f}%")
    print("=" * 55)

    # Business insight: margin < 15% = low-margin business; retail norm ~20-40%
    if avg_margin < 15:
        print("  INSIGHT: Low avg margin — push higher-margin categories.")
    else:
        print("  INSIGHT: Healthy margins across the portfolio.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Monthly Sales Trend
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_trend(df: pd.DataFrame) -> None:
    monthly = (df.groupby("year_month")["sales_amount"]
                 .sum()
                 .reset_index()
                 .rename(columns={"sales_amount": "revenue"}))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["year_month"], monthly["revenue"], marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(range(len(monthly)), monthly["revenue"], alpha=0.1, color="#2196F3")
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Monthly Revenue Trend (2021–2023)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Revenue (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "monthly_trend.png")
    plt.close()
    # INSIGHT: Revenue spikes visibly in Nov/Dec every year — confirm holiday effect.
    print("  [Chart] Monthly trend saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Sales by Category
# ─────────────────────────────────────────────────────────────────────────────
def plot_category_sales(df: pd.DataFrame) -> None:
    cat = (df.groupby("category")[["sales_amount", "profit"]]
             .sum()
             .sort_values("sales_amount", ascending=False)
             .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Revenue bar
    sns.barplot(data=cat, x="category", y="sales_amount", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Revenue by Category")
    axes[0].set_ylabel("Revenue (USD)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Profit bar
    sns.barplot(data=cat, x="category", y="profit", ax=axes[1], palette="Greens_d")
    axes[1].set_title("Profit by Category")
    axes[1].set_ylabel("Profit (USD)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.suptitle("Category Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_sales.png")
    plt.close()
    # INSIGHT: Electronics likely dominates revenue but Clothing/Sports have better margins.
    print("  [Chart] Category sales saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Sales by Region
# ─────────────────────────────────────────────────────────────────────────────
def plot_region_sales(df: pd.DataFrame) -> None:
    region = (df.groupby("region")["sales_amount"]
                .sum()
                .sort_values(ascending=False)
                .reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(region))
    bars = ax.barh(region["region"], region["sales_amount"], color=colors)
    ax.bar_label(bars, fmt="$%.0f", padding=5)
    ax.set_title("Revenue by Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue (USD)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "region_sales.png")
    plt.close()
    # INSIGHT: Fairly balanced across regions — no single region dominates.
    print("  [Chart] Region sales saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[["sales_amount", "quantity", "profit", "profit_margin"]].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap — Numeric Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()
    # INSIGHT: sales_amount & profit should correlate strongly (~0.9+).
    # Low correlation between quantity and margin = volume ≠ profitability.
    print("  [Chart] Correlation heatmap saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Top & Bottom Products
# ─────────────────────────────────────────────────────────────────────────────
def print_product_ranking(df: pd.DataFrame) -> None:
    prod = (df.groupby("product")["sales_amount"]
              .sum()
              .sort_values(ascending=False))

    print("  TOP 5 PRODUCTS BY REVENUE:")
    for i, (prod_name, rev) in enumerate(prod.head(5).items(), 1):
        print(f"    {i}. {prod_name:<20} ${rev:>10,.2f}")

    print("\n  BOTTOM 5 PRODUCTS (underperformers):")
    for i, (prod_name, rev) in enumerate(prod.tail(5).items(), 1):
        print(f"    {i}. {prod_name:<20} ${rev:>10,.2f}")
    # INSIGHT: Push marketing spend toward top 5; evaluate discontinuing bottom 5.
    print()


if __name__ == "__main__":
    print("\nRunning EDA...\n")
    df = load(DATA_PATH)
    print_kpis(df)
    plot_monthly_trend(df)
    plot_category_sales(df)
    plot_region_sales(df)
    plot_correlation_heatmap(df)
    print_product_ranking(df)
    print(f"\nAll plots saved to: {PLOTS_DIR}")
