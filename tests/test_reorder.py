"""Unit tests for the Inventory Reorder Engine."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reorder_engine import InventoryReorderEngine


def test_reorder_triggers():
    engine = InventoryReorderEngine(service_level_z=1.65)

    # 1. Critical stockout risk when stock is below lead time demand
    res_critical = engine.calculate_sku_reorder(
        sku_id="SKU_1", name="Milk", category="Dairy", velocity_class="Fast",
        current_stock=5, predicted_7d_demand=70.0, lead_time_days=2,
        shelf_life_days=7, historical_std=5.0, unit_price=2.0
    )
    assert res_critical["status"] == "CRITICAL_STOCKOUT_RISK"
    assert res_critical["recommended_order_qty"] > 0

    # 2. Healthy stock requires 0 order quantity
    res_healthy = engine.calculate_sku_reorder(
        sku_id="SKU_2", name="Cereal", category="Pantry", velocity_class="Medium",
        current_stock=100, predicted_7d_demand=20.0, lead_time_days=2,
        shelf_life_days=180, historical_std=3.0, unit_price=4.0
    )
    assert res_healthy["status"] in ["HEALTHY", "OVERSTOCKED"]
    assert res_healthy["recommended_order_qty"] == 0

    # 3. Slow-moving items do not over-order
    res_slow = engine.calculate_sku_reorder(
        sku_id="SKU_3", name="Truffle Oil", category="Gourmet", velocity_class="Slow",
        current_stock=2, predicted_7d_demand=1.5, lead_time_days=5,
        shelf_life_days=365, historical_std=0.5, unit_price=25.0
    )
    assert "Intermittent" in res_slow["rule_applied"]
    assert res_slow["recommended_order_qty"] <= 5
