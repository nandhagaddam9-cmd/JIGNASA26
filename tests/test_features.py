"""Unit tests for feature engineering and zero data leakage."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import build_retail_features, get_candidate_feature_names


def test_zero_lookahead_leakage():
    """Verify that lag and rolling features at date t do not use data from date >= t."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sku_id": "SKU_TEST",
        "product_name": "Test Item",
        "category": "Dairy",
        "velocity_class": "Fast",
        "lead_time_days": 2,
        "shelf_life_days": 10,
        "regular_price": 2.0,
        "selling_price": 2.0,
        "discount_percent": 0.0,
        "is_promo": 0,
        "consecutive_promo_days": 0,
        "true_demand": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "actual_sales": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "unfulfilled_demand": [0] * 10,
        "starting_stock": [100] * 10,
        "ending_stock": [90] * 10,
        "was_out_of_stock": [0] * 10,
    })

    feat_df = build_retail_features(df)
    
    # At index 1 (day 2), lag_1 must equal day 1 sales (10)
    assert feat_df.loc[1, "lag_1"] == 10
    # At index 2 (day 3), rolling_mean_7 must be mean of day 1 and 2: (10 + 20) / 2 = 15
    assert feat_df.loc[2, "rolling_mean_7"] == 15.0


def test_candidate_features_finite():
    """Assert candidate features contain zero NaNs or Infs."""
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "retail_store_sales.csv")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path).head(300)
        feat_df = build_retail_features(df_raw)
        candidate_cols = get_candidate_feature_names()
        for col in candidate_cols:
            assert not feat_df[col].isna().any(), f"Column {col} contains NaN"
            assert not np.isinf(feat_df[col]).any(), f"Column {col} contains Inf"
