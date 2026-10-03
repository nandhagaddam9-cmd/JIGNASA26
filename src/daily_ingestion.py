"""
Daily POS Data Ingestion & Live Inference Pipeline
Simulates how a real retailer runs the system every morning:
1. Ingests yesterday's raw barcode/POS sales & closing stock records.
2. Automatically updates historical time-series logs.
3. Dynamically transforms raw records into model-ready engineered features (lags, rolling stats, calendar, promo dynamics).
4. Runs the trained Deep Learning ANN to forecast the upcoming 7-day demand.
5. Feeds the forecast into the Reorder Engine to generate today's actionable Purchase Orders.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from ann_model import ANNDemandForecaster
from feature_engineering import build_retail_features
from reorder_engine import InventoryReorderEngine


class DailyRetailIngestionEngine:
    """
    Automated daily pipeline that bridges raw Point-of-Sale (POS) receipt logs
    to live ANN demand forecasts and purchase order recommendations.
    """

    def __init__(
        self,
        historical_raw_path: str = "data/retail_store_sales.csv",
        model_path: str = "data/ann_model.pt",
        feature_rankings_path: str = "data/feature_rankings.csv",
    ):
        self.historical_raw_path = historical_raw_path
        self.model_path = model_path
        self.feature_rankings_path = feature_rankings_path

        # Load selected features
        if os.path.exists(feature_rankings_path):
            rankings = pd.read_csv(feature_rankings_path)
            self.selected_features = rankings["feature"].head(16).tolist()
        else:
            self.selected_features = []

        # Load model
        self.ann_model = None
        if os.path.exists(model_path):
            self.ann_model = ANNDemandForecaster(feature_names=self.selected_features)
            self.ann_model.load(model_path)

        self.reorder_engine = InventoryReorderEngine(service_level_z=1.65)

    def ingest_yesterdays_raw_pos(
        self,
        new_daily_pos_records: pd.DataFrame,
        persist: bool = True,
    ) -> pd.DataFrame:
        """
        Step 1 & 2: Ingests raw scanner logs from yesterday.
        Merges with historical store database and removes any accidental duplicates.
        """
        if os.path.exists(self.historical_raw_path):
            hist_df = pd.read_csv(self.historical_raw_path)
            hist_df["date"] = pd.to_datetime(hist_df["date"])
        else:
            hist_df = pd.DataFrame()

        new_df = new_daily_pos_records.copy()
        new_df["date"] = pd.to_datetime(new_df["date"])

        # Concatenate and sort
        combined = pd.concat([hist_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "sku_id"], keep="last")
        combined = combined.sort_values(by=["sku_id", "date"]).reset_index(drop=True)

        if persist:
            os.makedirs(os.path.dirname(self.historical_raw_path), exist_ok=True)
            combined.to_csv(self.historical_raw_path, index=False)
            print(f"[Ingestion] Saved {len(new_daily_pos_records)} new POS records to {self.historical_raw_path}")

        return combined

    def transform_raw_to_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Transforms raw historical + new daily logs into 37 engineered features.
        Calculates updated lags (t-1), rolling statistics, momentum, and promo signals.
        """
        print("[Transform] Engineering 37 predictive features from updated raw time-series...")
        feat_df = build_retail_features(raw_df)
        return feat_df

    def run_morning_replenishment(
        self,
        new_pos_records: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        End-to-End Morning Workflow:
        1. Ingest new raw POS data (if provided).
        2. Transform to model features.
        3. Predict upcoming 7-day demand using ANN.
        4. Generate today's executable Purchase Order Manifest.
        """
        # Load or ingest
        if new_pos_records is not None and not new_pos_records.empty:
            raw_data = self.ingest_yesterdays_raw_pos(new_pos_records, persist=True)
        else:
            raw_data = pd.read_csv(self.historical_raw_path)
            raw_data["date"] = pd.to_datetime(raw_data["date"])

        # Transform to feature matrix
        feat_df = self.transform_raw_to_features(raw_data)

        # Get latest day snapshot per SKU
        latest_date = feat_df["date"].max()
        latest_snapshot = feat_df[feat_df["date"] == latest_date].copy().drop_duplicates(subset=["sku_id"])

        print(f"[Forecast] Running Deep Learning ANN inference for latest date: {latest_date.strftime('%Y-%m-%d')}...")

        # Predict next 7 days demand per SKU
        predictions_7d: Dict[str, float] = {}

        if self.ann_model is not None and len(self.selected_features) > 0:
            # Predict using the latest available state vector
            for sku in latest_snapshot["sku_id"].unique():
                sku_row = latest_snapshot[latest_snapshot["sku_id"] == sku]
                # Scale daily rate to 7-day horizon
                daily_pred = float(self.ann_model.predict(sku_row)[0])
                predictions_7d[sku] = round(daily_pred * 7.0, 1)
        else:
            # Fallback to rolling average if model not loaded
            for sku in latest_snapshot["sku_id"].unique():
                row = latest_snapshot[latest_snapshot["sku_id"] == sku].iloc[0]
                predictions_7d[sku] = round(float(row.get("rolling_mean_7", 10.0)) * 7.0, 1)

        # Generate Reorder Manifest
        print("[Reorder Engine] Calculating Reorder Points, Safety Stocks, and Purchase Orders...")
        manifest = self.reorder_engine.generate_store_reorder_manifest(
            current_state_df=latest_snapshot,
            predictions_7d=predictions_7d,
        )

        manifest_path = os.path.join(os.path.dirname(self.historical_raw_path), "latest_reorder_manifest.csv")
        manifest.to_csv(manifest_path, index=False)
        print(f"[Success] Today's Purchase Order Manifest saved -> {manifest_path}")

        return latest_snapshot, manifest


def simulate_sample_new_day_data(historical_path: str = "data/retail_store_sales.csv") -> pd.DataFrame:
    """
    Utility helper that creates a realistic single-day POS batch representing 'yesterday's sales'
    to test the live daily ingestion workflow.
    """
    df = pd.read_csv(historical_path)
    df["date"] = pd.to_datetime(df["date"])
    latest_dt = df["date"].max()
    next_dt = latest_dt + pd.Timedelta(days=1)

    latest_records = df[df["date"] == latest_dt].copy()
    new_rows = []

    for _, row in latest_records.iterrows():
        # Simulate yesterday's sales with realistic variation
        base = max(1, int(row["actual_sales"]))
        simulated_sales = max(0, int(np.random.normal(base, scale=max(1, base * 0.15))))
        prev_end_stock = int(row["ending_stock"])
        
        # New starting stock and ending stock
        start_stock = max(prev_end_stock, 10)
        end_stock = max(0, start_stock - simulated_sales)
        out_of_stock = 1 if end_stock == 0 else 0

        new_rows.append({
            "date": next_dt.strftime("%Y-%m-%d"),
            "sku_id": row["sku_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "velocity_class": row["velocity_class"],
            "lead_time_days": row["lead_time_days"],
            "shelf_life_days": row["shelf_life_days"],
            "regular_price": row["regular_price"],
            "selling_price": row["selling_price"],
            "discount_percent": row["discount_percent"],
            "is_promo": row["is_promo"],
            "consecutive_promo_days": row["consecutive_promo_days"],
            "true_demand": simulated_sales,
            "actual_sales": simulated_sales if out_of_stock == 0 else start_stock,
            "unfulfilled_demand": 0,
            "starting_stock": start_stock,
            "ending_stock": end_stock,
            "was_out_of_stock": out_of_stock,
        })

    return pd.DataFrame(new_rows)


if __name__ == "__main__":
    engine = DailyRetailIngestionEngine()
    print("--- Simulating Daily Morning Ingestion Pipeline ---")
    
    # 1. Generate a mock batch of yesterday's POS transactions
    mock_yesterdays_batch = simulate_sample_new_day_data()
    print(f"Generated mock POS batch for: {mock_yesterdays_batch['date'].iloc[0]} ({len(mock_yesterdays_batch)} SKUs)")

    # 2. Run full morning replenishment cycle
    snapshot, manifest = engine.run_morning_replenishment(new_pos_records=mock_yesterdays_batch)

    print("\n--- Morning Purchase Orders Generated ---")
    print(manifest.head(6)[["sku_id", "product_name", "velocity_class", "current_stock", "predicted_7d_demand", "reorder_point", "recommended_order_qty", "status"]].to_string(index=False))
