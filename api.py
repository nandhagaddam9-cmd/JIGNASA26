"""
FastAPI Production Backend REST API for Retail Inventory Reorder Assistant
Provides programmatic endpoints for:
- ERP & Point-of-Sale (POS) scanner data ingestion
- Live Deep Learning ANN demand forecasting
- Automated inventory reorder calculation and purchase order retrieval
- System health and catalog status
"""

import os
import sys
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ann_model import ANNDemandForecaster
from daily_ingestion import DailyRetailIngestionEngine
from reorder_engine import InventoryReorderEngine

app = FastAPI(
    title="Retail Inventory Reorder Assistant API",
    description="Production REST API backend for Track A (JIG26_06) providing ANN demand forecasting and risk-aware inventory replenishment.",
    version="1.0.0",
)

# Enable CORS for external frontends or ERP integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# Pydantic Request & Response Models
class POSRecordItem(BaseModel):
    date: str = Field(..., example="2026-01-01")
    sku_id: str = Field(..., example="SKU_FAST_001")
    product_name: str = Field(..., example="Fresh Whole Milk 1L")
    category: str = Field(..., example="Dairy")
    velocity_class: str = Field(..., example="Fast")
    lead_time_days: int = Field(..., example=2)
    shelf_life_days: int = Field(..., example=7)
    regular_price: float = Field(..., example=2.80)
    selling_price: float = Field(..., example=2.80)
    discount_percent: float = Field(0.0, example=0.0)
    is_promo: int = Field(0, example=0)
    consecutive_promo_days: int = Field(0, example=0)
    true_demand: int = Field(..., example=75)
    actual_sales: int = Field(..., example=75)
    unfulfilled_demand: int = Field(0, example=0)
    starting_stock: int = Field(..., example=120)
    ending_stock: int = Field(..., example=45)
    was_out_of_stock: int = Field(0, example=0)


class POSIngestRequest(BaseModel):
    records: List[POSRecordItem]


class PredictRequest(BaseModel):
    sku_id: str = Field(..., example="SKU_FAST_001")


# ----------------- ENDPOINTS -----------------

@app.get("/", tags=["Health"])
def root():
    """System health check and service overview."""
    return {
        "service": "Retail Inventory Reorder Assistant",
        "team": "TEAM-28",
        "track": "Track A - ANN & Predictive Analytics",
        "status": "ONLINE",
        "docs_url": "/docs",
    }


@app.get("/api/inventory/status", tags=["Inventory"])
def get_inventory_status():
    """Returns store-level inventory health summary and stockout alerts."""
    manifest_path = os.path.join(DATA_DIR, "latest_reorder_manifest.csv")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="Reorder manifest not found. Run pipeline first.")

    df = pd.read_csv(manifest_path)
    return {
        "total_active_skus": len(df),
        "critical_stockout_risks": len(df[df["status"] == "CRITICAL_STOCKOUT_RISK"]),
        "reorders_recommended": len(df[df["status"] == "REORDER_RECOMMENDED"]),
        "healthy_items": len(df[df["status"] == "HEALTHY"]),
        "overstocked_items": len(df[df["status"] == "OVERSTOCKED"]),
        "total_capital_required_usd": round(float(df["capital_required_usd"].sum()), 2),
    }


@app.get("/api/inventory/reorder-manifest", tags=["Inventory"])
def get_reorder_manifest(status_filter: Optional[str] = None):
    """Retrieves current purchase order recommendations across all SKUs."""
    manifest_path = os.path.join(DATA_DIR, "latest_reorder_manifest.csv")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="Reorder manifest not found.")

    df = pd.read_csv(manifest_path)
    if status_filter:
        df = df[df["status"] == status_filter]

    return df.to_dict(orient="records")


@app.get("/api/models/scorecard", tags=["Model Analytics"])
def get_benchmark_scorecard():
    """Returns the chronological benchmark scorecard comparing ANN against Moving Averages."""
    scorecard_path = os.path.join(DATA_DIR, "model_scorecard.csv")
    if not os.path.exists(scorecard_path):
        raise HTTPException(status_code=404, detail="Scorecard not found.")

    df = pd.read_csv(scorecard_path)
    return df.to_dict(orient="records")


@app.get("/api/features/rankings", tags=["Feature Intelligence"])
def get_feature_rankings():
    """Returns the top curated features and composite attribution scores."""
    rankings_path = os.path.join(DATA_DIR, "feature_rankings.csv")
    if not os.path.exists(rankings_path):
        raise HTTPException(status_code=404, detail="Feature rankings not found.")

    df = pd.read_csv(rankings_path)
    return df.to_dict(orient="records")


@app.post("/api/pos/ingest", tags=["Daily ETL"])
def ingest_pos_data(payload: POSIngestRequest):
    """
    Ingests new daily Point-of-Sale (POS) records, triggers feature transformation,
    runs the ANN demand model, and updates the reorder manifest.
    """
    records_data = [item.dict() for item in payload.records]
    df_new = pd.DataFrame(records_data)

    engine = DailyRetailIngestionEngine(
        historical_raw_path=os.path.join(DATA_DIR, "retail_store_sales.csv"),
        model_path=os.path.join(DATA_DIR, "ann_model.pt"),
        feature_rankings_path=os.path.join(DATA_DIR, "feature_rankings.csv"),
    )

    try:
        snapshot, manifest = engine.run_morning_replenishment(new_pos_records=df_new)
        return {
            "status": "SUCCESS",
            "ingested_records": len(df_new),
            "updated_reorder_manifest_count": len(manifest),
            "critical_stockout_risks": len(manifest[manifest["status"] == "CRITICAL_STOCKOUT_RISK"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
