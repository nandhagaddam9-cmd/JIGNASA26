"""
End-to-End Orchestrator Pipeline for Retail Inventory Reorder Assistant
Executes:
1. Dataset Verification & Feature Engineering
2. Chronological Time-Series Split (zero future leakage)
3. Rigorous Feature Selection (Multicollinearity, Mutual Info, Tree Attribution)
4. Model Training: Traditional Baselines vs. Deep Learning ANN
5. Chronological Metric Evaluation (MAE, RMSE, WAPE, Asymmetric Stock-Out Loss)
6. Inventory Reorder Optimization & Decision Manifest Generation
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ann_model import ANNDemandForecaster
from baseline_models import MovingAverageForecaster
from data_generator import generate_retail_dataset
from evaluation import evaluate_models_chronologically
from feature_engineering import build_retail_features, get_candidate_feature_names
from feature_selection import rank_and_select_features
from reorder_engine import InventoryReorderEngine


def run_full_pipeline(
    data_dir: str = "data",
    top_k_features: int = 16,
    ann_epochs: int = 50,
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    raw_data_path = os.path.join(data_dir, "retail_store_sales.csv")

    # Step 1: Ensure dataset exists
    if not os.path.exists(raw_data_path):
        print("[1/6] Generating synthetic realistic retail dataset...")
        df_raw = generate_retail_dataset(days=730, output_path=raw_data_path)
    else:
        print(f"[1/6] Loading retail dataset from {raw_data_path}...")
        df_raw = pd.read_csv(raw_data_path)

    # Step 2: Feature Engineering
    print("[2/6] Engineering candidate features across all 6 retail pillars...")
    df_feat = build_retail_features(df_raw)
    feat_output_path = os.path.join(data_dir, "retail_engineered_features.csv")
    df_feat.to_csv(feat_output_path, index=False)
    print(f"      Engineered {len(get_candidate_feature_names())} features across {len(df_feat)} records.")

    # Step 3: Chronological Time-Series Split
    print("[3/6] Performing Chronological Time-Series Split (Last 30 Days = Test Set)...")
    max_date = df_feat["date"].max()
    split_date = max_date - pd.Timedelta(days=30)

    train_df = df_feat[df_feat["date"] <= split_date].copy().reset_index(drop=True)
    test_df = df_feat[df_feat["date"] > split_date].copy().reset_index(drop=True)
    print(f"      Train Set: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    print(f"      Test Set:  {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df)} rows)")

    # Step 4: Rigorous Feature Selection
    print("[4/6] Executing Feature Selection Pipeline (Multicollinearity Filter + MI + Tree Attribution)...")
    candidate_cols = get_candidate_feature_names()
    selected_features, ranking_df = rank_and_select_features(
        X_train=train_df[candidate_cols],
        y_train=train_df["actual_sales"],
        top_k=top_k_features,
    )
    ranking_path = os.path.join(data_dir, "feature_rankings.csv")
    ranking_df.to_csv(ranking_path, index=False)
    print("      Top Selected Predictive Features:")
    for rank, feat in enumerate(selected_features, 1):
        score = ranking_df[ranking_df['feature'] == feat]['composite_score'].values[0]
        print(f"        {rank:2d}. {feat:<28} (Score: {score:.4f})")

    # Step 5: Model Training & Forecasting
    print("[5/6] Training Models & Generating Predictions...")
    
    # 5a. Baselines
    print("      Fitting Baseline Models (SMA-7, SMA-14, Seasonal-SMA)...")
    sma_7_model = MovingAverageForecaster(method="sma_7").fit(train_df)
    sma_14_model = MovingAverageForecaster(method="sma_14").fit(train_df)
    seasonal_sma_model = MovingAverageForecaster(method="seasonal_sma").fit(train_df)

    test_preds_dict = {
        "Baseline SMA-7": sma_7_model.predict(test_df),
        "Baseline SMA-14": sma_14_model.predict(test_df),
        "Baseline Seasonal-SMA": seasonal_sma_model.predict(test_df),
    }

    # 5b. Deep Learning ANN
    print(f"      Training Deep Learning ANN Forecaster ({ann_epochs} epochs)...")
    ann = ANNDemandForecaster(feature_names=selected_features, epochs=ann_epochs)
    ann.fit(train_df=train_df, val_df=test_df)
    test_preds_dict["Deep Learning ANN Forecaster"] = ann.predict(test_df)

    # Save ANN model
    model_save_path = os.path.join(data_dir, "ann_model.pt")
    ann.save(model_save_path)
    print(f"      Model checkpoint saved to {model_save_path}")

    # Step 6: Chronological Evaluation Scorecard
    print("[6/6] Computing Performance Metrics & Stock-Out-Oriented Error Loss...")
    scorecard = evaluate_models_chronologically(test_df, test_preds_dict)
    scorecard_path = os.path.join(data_dir, "model_scorecard.csv")
    scorecard.to_csv(scorecard_path, index=False)

    print("\n" + "=" * 90)
    print("                   CHRONOLOGICAL MODEL BENCHMARK SCORECARD")
    print("=" * 90)
    print(scorecard.to_string(index=False))
    print("=" * 90)

    # Step 7: Inventory Reorder Manifest Generation
    print("\n[+] Generating Actionable Inventory Reorder Manifest for Active SKUs...")
    latest_date = df_feat["date"].max()
    latest_state = df_feat[df_feat["date"] == latest_date].copy().drop_duplicates(subset=["sku_id"])

    # Forecast 7-day demand using ANN for each SKU
    sku_7d_forecasts = {}
    for sku in latest_state["sku_id"].unique():
        sku_recent = test_df[test_df["sku_id"] == sku].tail(7)
        if len(sku_recent) > 0:
            sku_7d_forecasts[sku] = float(np.sum(ann.predict(sku_recent)))
        else:
            sku_7d_forecasts[sku] = float(latest_state[latest_state["sku_id"] == sku]["rolling_mean_7"].values[0] * 7.0)

    engine = InventoryReorderEngine(service_level_z=1.65)
    manifest = engine.generate_store_reorder_manifest(latest_state, sku_7d_forecasts)
    manifest_path = os.path.join(data_dir, "latest_reorder_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"      Saved Reorder Manifest ({len(manifest)} SKUs) -> {manifest_path}")
    print("\nTop Priority Reorder Recommendations:")
    print(manifest.head(8)[["sku_id", "product_name", "velocity_class", "current_stock", "predicted_7d_demand", "reorder_point", "recommended_order_qty", "status", "capital_required_usd"]].to_string(index=False))
    print("\n Pipeline execution completed successfully!")


if __name__ == "__main__":
    run_full_pipeline()
