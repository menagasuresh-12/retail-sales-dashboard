"""
data_processing.py  —  Step 2: Clean & validate the raw dataset
---------------------------------------------------------------
Run after generate_data.py:
    python src/data_processing.py

What it does:
  1. Loads raw_sales.csv
  2. Converts date column to proper datetime type
  3. Drops duplicate rows
  4. Fills missing numeric values with column medians
  5. Validates data types and ranges
  6. Saves cleaned data to data/cleaned_sales.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH     = Path(__file__).parent.parent / "data" / "raw_sales.csv"
CLEANED_PATH = Path(__file__).parent.parent / "data" / "cleaned_sales.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV and display a quick summary."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline — returns a clean DataFrame."""

    original_len = len(df)

    # ── 1. Parse dates ────────────────────────────────────────────────────────
    # Convert string dates to proper datetime objects so we can do time maths
    df["date"] = pd.to_datetime(df["date"])

    # ── 2. Drop exact duplicates ──────────────────────────────────────────────
    df = df.drop_duplicates()
    print(f"Removed {original_len - len(df)} duplicate rows")

    # ── 3. Fill missing numeric values with median ────────────────────────────
    # Median is more robust than mean for skewed financial data
    for col in ["sales_amount", "quantity", "profit"]:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {n_null} nulls in '{col}' with median={median_val:.2f}")

    # ── 4. Sanity checks / data integrity ────────────────────────────────────
    # Remove any rows where sales or quantity somehow ended up <= 0
    before = len(df)
    df = df[(df["sales_amount"] > 0) & (df["quantity"] > 0)]
    print(f"Removed {before - len(df)} rows with non-positive sales/quantity")

    # ── 5. Add useful derived columns ─────────────────────────────────────────
    df["year"]          = df["date"].dt.year
    df["month"]         = df["date"].dt.month
    df["month_name"]    = df["date"].dt.strftime("%b")         # Jan, Feb, …
    df["year_month"]    = df["date"].dt.to_period("M").astype(str)  # 2021-01
    df["profit_margin"] = (df["profit"] / df["sales_amount"] * 100).round(2)

    # Reset index after all the dropping
    df = df.reset_index(drop=True)
    print(f"\nFinal clean dataset: {len(df):,} rows")
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data -> {path}")


if __name__ == "__main__":
    df_raw   = load_data(RAW_PATH)
    df_clean = clean_data(df_raw)
    save_data(df_clean, CLEANED_PATH)
