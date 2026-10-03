"""
Feature Engineering Engine for Retail Demand Forecasting
Constructs candidate feature pools across 6 pillars:
- Autoregressive Lags
- Rolling Volatility & Mean (Zero-Leakage Shifted)
- Momentum & Velocity Ratios
- Price & Promotional Elasticity
- Calendar, Cyclical & Payday Signals
- Censored Demand / Stockout Feedback
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def build_retail_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw retail DataFrame and computes 30+ engineered predictive features.
    Ensures ZERO lookahead bias: all rolling statistics and lags are strictly
    computed using historical data up to t-1.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by=["sku_id", "date"]).reset_index(drop=True)

    # 1. Calendar & Cyclical Features
    data["day_of_week"] = data["date"].dt.weekday
    data["is_weekend"] = data["day_of_week"].isin([4, 5, 6]).astype(int)  # Fri, Sat, Sun
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter

    # Cyclical Sine/Cosine transformations
    data["cyclical_dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cyclical_dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["cyclical_month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["cyclical_month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # Payday window (1st-5th and 28th-31st)
    data["is_payday_window"] = (
        (data["day_of_month"] <= 5) | (data["day_of_month"] >= 28)
    ).astype(int)

    # Grouped operations per SKU
    engineered_dfs = []
    
    for sku_id, group in data.groupby("sku_id"):
        g = group.copy().sort_values("date").reset_index(drop=True)

        # Sales quantity (what was actually observed)
        sales = g["actual_sales"]

        # 2. Autoregressive Lags (strictly prior days)
        g["lag_1"] = sales.shift(1)
        g["lag_2"] = sales.shift(2)
        g["lag_3"] = sales.shift(3)
        g["lag_7"] = sales.shift(7)
        g["lag_14"] = sales.shift(14)
        g["lag_21"] = sales.shift(21)
        g["lag_28"] = sales.shift(28)

        # 3. Rolling Window Statistics (Shifted by 1 day to prevent leakage)
        sales_shift1 = sales.shift(1)
        
        g["rolling_mean_7"] = sales_shift1.rolling(window=7, min_periods=1).mean()
        g["rolling_mean_14"] = sales_shift1.rolling(window=14, min_periods=1).mean()
        g["rolling_mean_28"] = sales_shift1.rolling(window=28, min_periods=1).mean()

        g["rolling_std_7"] = sales_shift1.rolling(window=7, min_periods=1).std().fillna(0)
        g["rolling_std_14"] = sales_shift1.rolling(window=14, min_periods=1).std().fillna(0)

        g["rolling_min_7"] = sales_shift1.rolling(window=7, min_periods=1).min()
        g["rolling_max_7"] = sales_shift1.rolling(window=7, min_periods=1).max()

        # 4. Momentum & Acceleration Ratios
        g["momentum_ratio_7"] = (g["lag_1"] / (g["rolling_mean_7"] + 1e-4)).clip(0, 5.0)
        g["same_dow_ratio_7_14"] = (g["lag_7"] / (g["lag_14"] + 1e-4)).clip(0, 5.0)

        # 5. Price & Promo Elasticity Features
        price_shift1 = g["selling_price"].shift(1)
        g["rolling_avg_price_30"] = price_shift1.rolling(window=30, min_periods=1).mean()
        g["relative_price_ratio"] = (g["selling_price"] / (g["rolling_avg_price_30"] + 1e-4)).clip(0.4, 2.0)

        # Days since last promotion
        promo_series = g["is_promo"]
        days_since_promo = []
        last_p = -999
        for i, val in enumerate(promo_series):
            if val == 1:
                last_p = i
            days_since_promo.append(min(30, i - last_p) if last_p != -999 else 30)
        g["days_since_last_promo"] = days_since_promo

        # 6. Censored Stock-out & Inventory Feedback
        g["was_out_of_stock_lag1"] = g["was_out_of_stock"].shift(1).fillna(0).astype(int)
        stock_shift1 = g["starting_stock"].shift(1)
        g["stock_to_sales_ratio_lag1"] = (stock_shift1 / (g["rolling_mean_7"] + 1e-4)).clip(0, 10.0)

        engineered_dfs.append(g)

    result_df = pd.concat(engineered_dfs, ignore_index=True)

    # 7. Static Categorical Encodings
    category_map = {"Dairy": 0, "Bakery": 1, "Beverages": 2, "Produce": 3, "Snacks": 4, "Pantry": 5, "Household": 6, "Gourmet": 7}
    velocity_map = {"Fast": 0, "Medium": 1, "Slow": 2, "New": 3}

    result_df["category_code"] = result_df["category"].map(category_map).fillna(0).astype(int)
    result_df["velocity_code"] = result_df["velocity_class"].map(velocity_map).fillna(0).astype(int)

    # Ensure all engineered features are finite and free of NaNs or Infs
    candidate_cols = get_candidate_feature_names()
    for c in candidate_cols:
        if c in result_df.columns:
            result_df[c] = result_df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return result_df


def get_candidate_feature_names() -> List[str]:
    """Returns list of candidate features engineered for selection."""
    return [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_14",
        "rolling_min_7", "rolling_max_7",
        "momentum_ratio_7", "same_dow_ratio_7_14",
        "is_promo", "discount_percent", "consecutive_promo_days",
        "relative_price_ratio", "days_since_last_promo",
        "day_of_week", "is_weekend", "cyclical_dow_sin", "cyclical_dow_cos",
        "day_of_month", "month", "cyclical_month_sin", "cyclical_month_cos",
        "is_payday_window", "was_out_of_stock_lag1", "stock_to_sales_ratio_lag1",
        "lead_time_days", "shelf_life_days", "regular_price",
        "category_code", "velocity_code"
    ]


if __name__ == "__main__":
    import os
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "retail_store_sales.csv")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        df_feat = build_retail_features(df_raw)
        print(f"Engineered {len(get_candidate_feature_names())} features across {len(df_feat)} rows.")
        print(f"Sample columns:\n{get_candidate_feature_names()[:10]}")
