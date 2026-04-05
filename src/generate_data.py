"""
generate_data.py  —  Step 1: Create a realistic synthetic retail dataset
------------------------------------------------------------------------
Run this FIRST before anything else:
    python src/generate_data.py

What it does:
  - Generates 5,000 sales transactions across 3 years (2021-2023)
  - Adds realistic seasonality (holiday peaks, post-holiday dips)
  - Intentionally introduces ~1% nulls and ~0.5% duplicates for cleaning demo
  - Saves raw CSV to data/raw_sales.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Seed for reproducibility ──────────────────────────────────────────────────
np.random.seed(42)

N_ROWS = 5000

# Product catalog: category → list of products
CATEGORIES = {
    "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch", "Monitor", "Webcam"],
    "Clothing":    ["T-Shirt", "Jeans", "Jacket", "Shoes", "Dress", "Sweater", "Shorts"],
    "Groceries":   ["Bread", "Milk", "Rice", "Olive Oil", "Coffee", "Juice", "Cereal"],
    "Furniture":   ["Chair", "Desk", "Bookshelf", "Sofa", "Lamp", "Wardrobe", "Bed Frame"],
    "Sports":      ["Yoga Mat", "Dumbbell Set", "Running Shoes", "Bicycle", "Tennis Racket"],
}

REGIONS = ["North", "South", "East", "West", "Central"]

# Unit price ranges (USD) per category
PRICE_RANGE = {
    "Electronics": (50, 1500),
    "Clothing":    (10,  200),
    "Groceries":   (1,    30),
    "Furniture":   (80,  800),
    "Sports":      (15,  400),
}

# Gross profit margin ranges per category
MARGIN_RANGE = {
    "Electronics": (0.08, 0.20),
    "Clothing":    (0.30, 0.55),
    "Groceries":   (0.05, 0.15),
    "Furniture":   (0.20, 0.40),
    "Sports":      (0.25, 0.45),
}


def seasonal_multiplier(month: int) -> float:
    """
    Models real retail seasonality.
    Nov/Dec spike = holiday shopping; Jan/Feb dip = post-holiday slow.
    """
    curve = {1:0.75, 2:0.70, 3:0.85, 4:0.90, 5:0.95, 6:1.00,
             7:1.05, 8:1.10, 9:1.00, 10:1.05, 11:1.30, 12:1.50}
    return curve[month]


def generate_dataset() -> pd.DataFrame:
    records = []
    start = pd.Timestamp("2021-01-01")
    end   = pd.Timestamp("2023-12-31")
    span  = (end - start).days

    for _ in range(N_ROWS):
        date     = start + pd.Timedelta(days=int(np.random.randint(0, span)))
        category = np.random.choice(list(CATEGORIES.keys()))
        product  = np.random.choice(CATEGORIES[category])
        region   = np.random.choice(REGIONS)

        pmin, pmax  = PRICE_RANGE[category]
        unit_price  = np.random.uniform(pmin, pmax)
        quantity    = max(1, int(np.random.poisson(3 * seasonal_multiplier(date.month))))
        sales       = round(unit_price * quantity, 2)

        mmin, mmax  = MARGIN_RANGE[category]
        profit      = round(sales * np.random.uniform(mmin, mmax), 2)

        records.append({"date": date.strftime("%Y-%m-%d"), "product": product,
                        "category": category, "region": region,
                        "sales_amount": sales, "quantity": quantity, "profit": profit})

    return pd.DataFrame(records)


def add_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Inject nulls + duplicates so the cleaning step has real work to do."""
    null_idx = df.sample(frac=0.01, random_state=1).index
    df.loc[null_idx, "sales_amount"] = np.nan
    df.loc[null_idx[:len(null_idx)//2], "profit"] = np.nan
    dupes = df.sample(frac=0.005, random_state=2)
    return pd.concat([df, dupes], ignore_index=True)


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "raw_sales.csv"
    out.parent.mkdir(exist_ok=True)
    print("Generating dataset...")
    df = add_noise(generate_dataset())
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")
