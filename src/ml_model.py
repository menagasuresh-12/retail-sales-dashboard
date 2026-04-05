"""
ml_model.py  —  Step 4: Sales Forecasting with Linear Regression
-----------------------------------------------------------------
Run after data_processing.py:
    python src/ml_model.py

What it does:
  - Aggregates daily → monthly revenue time series
  - Engineers time-based features (month number, year, lag features)
  - Trains a Linear Regression model on 80% of data
  - Evaluates on the remaining 20% (MAE, MSE, R²)
  - Saves trained model + month scaler to data/model.pkl
  - Predicts next 6 months and saves to data/forecast.csv
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH     = Path(__file__).parent.parent / "data" / "cleaned_sales.csv"
MODEL_PATH    = Path(__file__).parent.parent / "data" / "model.pkl"
FORECAST_PATH = Path(__file__).parent.parent / "data" / "forecast.csv"


def load_monthly_series(path: Path) -> pd.DataFrame:
    """
    Aggregate transaction-level data into a monthly revenue time series.
    Each row = one month's total revenue.
    """
    df = pd.read_csv(path, parse_dates=["date"])

    monthly = (df.groupby("year_month")["sales_amount"]
                 .sum()
                 .reset_index()
                 .rename(columns={"sales_amount": "revenue"}))

    # Convert 'year_month' string (e.g. "2021-01") to a proper Period, then integer index
    monthly["period"]  = pd.PeriodIndex(monthly["year_month"], freq="M")
    monthly = monthly.sort_values("period").reset_index(drop=True)
    monthly["t"]       = monthly.index          # simple integer time index
    monthly["year"]    = monthly["period"].dt.year
    monthly["month"]   = monthly["period"].dt.month

    return monthly


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML features from the time index.
    - t          : raw integer (trend direction)
    - month      : seasonality (cyclical month number)
    - year       : inter-year trend
    - sin/cos    : smooth cyclical encoding of month (better than raw integer)
    - lag_1/lag_2: previous 1 and 2 months' revenue (autoregressive signal)
    """
    df = df.copy()
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["lag_1"]     = df["revenue"].shift(1)     # last month
    df["lag_2"]     = df["revenue"].shift(2)     # two months ago

    # Drop rows where lag features are NaN (first 2 rows)
    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate(df: pd.DataFrame):
    """
    Train Linear Regression and return model, scaler, and metrics dict.
    """
    FEATURES = ["t", "year", "sin_month", "cos_month", "lag_1", "lag_2"]
    TARGET   = "revenue"

    X = df[FEATURES]
    y = df[TARGET]

    # 80/20 split — keep chronological order (no shuffle)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Standardise features (helps linear models converge and generalise)
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_sc)

    # Evaluation metrics
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print("=" * 45)
    print("  MODEL EVALUATION  (test set)")
    print("=" * 45)
    print(f"  MAE   : ${mae:>10,.2f}")
    print(f"  RMSE  : ${rmse:>10,.2f}")
    print(f"  MSE   : ${mse:>10,.2f}")
    print(f"  R²    : {r2:>11.4f}")
    print("=" * 45)

    # Business interpretation:
    # R² > 0.85 = model explains >85% of revenue variance — solid for retail
    if r2 > 0.8:
        print("  INSIGHT: Good fit. Model captures seasonal + trend patterns well.")
    else:
        print("  INSIGHT: Moderate fit. Consider adding category/promo features.")
    print()

    # Attach predictions back to the test slice for dashboard use
    test_df = df.iloc[split_idx:].copy()
    test_df["predicted"] = y_pred

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2),
               "MSE": round(mse, 2), "R2": round(r2, 4)}
    return model, scaler, metrics, test_df


def forecast_future(model, scaler, df: pd.DataFrame, n_months: int = 6) -> pd.DataFrame:
    """
    Predict the next n_months beyond the training data.
    Uses the last known revenue values as lag features.
    """
    last_row   = df.iloc[-1]
    last_t     = int(last_row["t"])
    last_period = last_row["period"]

    future_rows = []
    lag_1 = df["revenue"].iloc[-1]
    lag_2 = df["revenue"].iloc[-2]

    for i in range(1, n_months + 1):
        next_period = last_period + i
        t      = last_t + i
        year   = next_period.year
        month  = next_period.month
        row = {
            "t":         t,
            "year":      year,
            "sin_month": np.sin(2 * np.pi * month / 12),
            "cos_month": np.cos(2 * np.pi * month / 12),
            "lag_1":     lag_1,
            "lag_2":     lag_2,
        }
        future_rows.append(row)
        # Update lags for the next iteration
        lag_2 = lag_1
        lag_1 = model.predict(scaler.transform(pd.DataFrame([row])))[0]

    future_df = pd.DataFrame(future_rows)
    FEATURES  = ["t", "year", "sin_month", "cos_month", "lag_1", "lag_2"]
    preds     = model.predict(scaler.transform(future_df[FEATURES]))

    forecast = pd.DataFrame({
        "year_month": [(last_period + i).strftime("%Y-%m") for i in range(1, n_months + 1)],
        "predicted_revenue": preds.round(2),
    })
    return forecast


def save_model(model, scaler, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"Model saved -> {path}")


def load_model(path: Path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]


if __name__ == "__main__":
    print("\nTraining forecasting model...\n")
    monthly    = load_monthly_series(DATA_PATH)
    monthly_fe = engineer_features(monthly)

    model, scaler, metrics, test_df = train_and_evaluate(monthly_fe)

    forecast = forecast_future(model, scaler, monthly_fe, n_months=6)
    forecast.to_csv(FORECAST_PATH, index=False)
    print(f"Forecast saved -> {FORECAST_PATH}")
    print(forecast.to_string(index=False))

    save_model(model, scaler, MODEL_PATH)
