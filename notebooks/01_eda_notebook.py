"""
01_eda_notebook.py
------------------
A standalone script version of the EDA notebook.
Run with:  python notebooks/01_eda_notebook.py

This file is meant to be readable like a Jupyter notebook
using the '# %%' cell separator convention (works in VS Code).
"""

# %% [markdown]
# # Retail Sales — Exploratory Data Analysis
# This notebook walks through the full EDA step by step.

# %% Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load data
df = pd.read_csv(Path(__file__).parent.parent / "data" / "cleaned_sales.csv",
                 parse_dates=["date"])
print(df.head())
print(df.describe())

# %% KPIs
print(f"Revenue : ${df['sales_amount'].sum():,.2f}")
print(f"Profit  : ${df['profit'].sum():,.2f}")
print(f"Qty     : {df['quantity'].sum():,}")

# %% Monthly trend
monthly = df.groupby("year_month")["sales_amount"].sum()
monthly.plot(kind="line", figsize=(12,4), title="Monthly Revenue", marker="o")
plt.tight_layout(); plt.show()

# %% Category revenue
cat = df.groupby("category")["sales_amount"].sum().sort_values(ascending=False)
cat.plot(kind="bar", title="Revenue by Category", color="steelblue")
plt.tight_layout(); plt.show()
