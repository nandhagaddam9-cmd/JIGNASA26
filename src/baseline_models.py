"""
Traditional Moving Average Baseline Forecasters
Implements:
1. Simple Moving Average 7-day (SMA-7)
2. Simple Moving Average 14-day (SMA-14)
3. Seasonal Same-Day-of-Week Moving Average (Seasonal-SMA)
4. Category Fallback for Cold-Start Items
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class MovingAverageForecaster:
    """
    Traditional retail moving average baseline forecaster.
    Provides SMA-7, SMA-14, and Seasonal-SMA predictions with cold-start category fallback.
    """

    def __init__(self, method: str = "sma_7", window: int = 7):
        self.method = method  # 'sma_7', 'sma_14', or 'seasonal_sma'
        self.window = window
        self.category_fallbacks: Dict[str, float] = {}
        self.global_fallback: float = 10.0

    def fit(self, train_df: pd.DataFrame) -> "MovingAverageForecaster":
        """
        Learns fallback category medians from historical training data
        to handle new or cold-start items with zero or sparse history.
        """
        cat_medians = train_df.groupby("category")["actual_sales"].median().to_dict()
        self.category_fallbacks = cat_medians
        self.global_fallback = float(train_df["actual_sales"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for each row in df based strictly on historical lags.
        """
        preds = []
        for _, row in df.iterrows():
            pred = None

            if self.method == "sma_7":
                # Average of last 7 days from pre-calculated rolling_mean_7 or lags
                val = row.get("rolling_mean_7", np.nan)
                if pd.notna(val) and val > 0:
                    pred = val
                else:
                    # Fallback to available short lags
                    lags = [row.get(f"lag_{i}", np.nan) for i in range(1, 8)]
                    valid = [l for l in lags if pd.notna(l) and l >= 0]
                    pred = np.mean(valid) if len(valid) > 0 else None

            elif self.method == "sma_14":
                val = row.get("rolling_mean_14", np.nan)
                if pd.notna(val) and val > 0:
                    pred = val
                else:
                    lags = [row.get(f"lag_{i}", np.nan) for i in range(1, 15)]
                    valid = [l for l in lags if pd.notna(l) and l >= 0]
                    pred = np.mean(valid) if len(valid) > 0 else None

            elif self.method == "seasonal_sma":
                # Same day of week across the previous 4 weeks (lag_7, lag_14, lag_21, lag_28)
                seasonal_lags = [row.get(f"lag_{k}", np.nan) for k in [7, 14, 21, 28]]
                valid = [l for l in seasonal_lags if pd.notna(l) and l >= 0]
                if len(valid) >= 2:
                    pred = np.mean(valid)
                else:
                    pred = row.get("rolling_mean_7", None)

            # Cold-start / fallback rule:
            if pred is None or pd.isna(pred) or pred <= 0:
                cat = row.get("category", "")
                pred = self.category_fallbacks.get(cat, self.global_fallback)

            preds.append(max(0.0, float(pred)))

        return np.array(preds)


if __name__ == "__main__":
    import os
    from feature_engineering import build_retail_features

    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "retail_store_sales.csv")
    df_raw = pd.read_csv(raw_path)
    df_feat = build_retail_features(df_raw)

    max_date = df_feat["date"].max()
    split_date = max_date - pd.Timedelta(days=30)
    train_df = df_feat[df_feat["date"] <= split_date].copy()
    test_df = df_feat[df_feat["date"] > split_date].copy()

    for method in ["sma_7", "sma_14", "seasonal_sma"]:
        model = MovingAverageForecaster(method=method)
        model.fit(train_df)
        preds = model.predict(test_df)
        mae = np.mean(np.abs(test_df["actual_sales"].values - preds))
        print(f"Baseline {method.upper()} Test MAE: {mae:.2f}")
