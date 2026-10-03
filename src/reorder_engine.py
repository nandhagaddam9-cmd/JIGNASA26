"""
Retail Inventory Reorder Engine
Translates demand forecasts into actionable inventory purchase orders.
Implements:
- Dynamic Safety Stock (SS) and Reorder Point (ROP) for Fast & Medium items
- Min-Max / Poisson thresholding for Slow-Moving / Intermittent items
- Category-Analog initial batching for New / Cold-Start items
- Expiration & Shelf-Life bounds
- Clear Actionable Status Alerts (Critical Stockout Risk, Reorder, Healthy, Overstocked)
"""

import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class InventoryReorderEngine:
    """
    Supply chain decision engine translating 7-day demand predictions into
    precise SKU purchase order recommendations.
    """

    def __init__(self, service_level_z: float = 1.65):  # 1.65 corresponds to 95% cycle service level
        self.z = service_level_z

    def calculate_sku_reorder(
        self,
        sku_id: str,
        name: str,
        category: str,
        velocity_class: str,
        current_stock: int,
        predicted_7d_demand: float,
        lead_time_days: int,
        shelf_life_days: int,
        historical_std: float,
        unit_price: float,
        category_medians: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Calculates reorder point, safety stock, and recommended order quantity
        with distinct rule sets for standard, slow, and new items.
        """
        predicted_daily_rate = max(0.05, predicted_7d_demand / 7.0)
        lead_time_demand = predicted_daily_rate * lead_time_days

        # Rule Set 1: New / Cold-Start Items
        if velocity_class == "New":
            cat_median_rate = 25.0
            if category_medians and category in category_medians:
                cat_median_rate = category_medians[category] / 7.0
            
            # Category-Analog Heuristic with higher buffer
            safety_stock = int(math.ceil(1.5 * math.sqrt(lead_time_days) * (cat_median_rate * 0.35)))
            reorder_point = int(math.ceil(cat_median_rate * lead_time_days + safety_stock))
            target_stock = int(math.ceil(cat_median_rate * 7.0 + safety_stock))
            rule_applied = "Cold-Start Category-Analog Rule (1.5x Launch Buffer)"

        # Rule Set 2: Slow-Moving / Intermittent Items
        elif velocity_class == "Slow":
            # Poisson/Min-Max heuristic to avoid over-ordering low velocity SKUs
            safety_stock = 1  # 1 unit buffer is sufficient for 0.5-2 units/day items
            reorder_point = int(math.ceil(lead_time_demand)) + safety_stock
            target_stock = max(3, reorder_point + 2)
            rule_applied = "Intermittent / Croston-Poisson Min-Max Rule"

        # Rule Set 3: Fast & Medium Standard Items
        else:
            # Parametric Dynamic Safety Stock based on forecast volatility
            volatility = max(2.0, historical_std)
            safety_stock = int(math.ceil(self.z * volatility * math.sqrt(lead_time_days)))
            reorder_point = int(math.ceil(lead_time_demand + safety_stock))
            target_stock = int(math.ceil(predicted_7d_demand + safety_stock))
            rule_applied = "Standard Continuous (s, S) Dynamic Safety Stock Rule"

        # Shelf-life constraint: Do not order more than can be consumed before expiration
        max_shelf_stock = int(math.floor(predicted_daily_rate * max(1, shelf_life_days * 0.8)))
        if target_stock > max_shelf_stock and shelf_life_days <= 14:
            target_stock = max_shelf_stock
            rule_applied += " [Capped by Perishability Shelf-Life]"

        # Decision Logic: Reorder Trigger
        if current_stock <= lead_time_demand and current_stock < reorder_point:
            status = "CRITICAL_STOCKOUT_RISK"
            recommended_order_qty = max(0, target_stock - current_stock)
        elif current_stock <= reorder_point:
            status = "REORDER_RECOMMENDED"
            recommended_order_qty = max(0, target_stock - current_stock)
        elif current_stock > (target_stock * 1.8):
            status = "OVERSTOCKED"
            recommended_order_qty = 0
        else:
            status = "HEALTHY"
            recommended_order_qty = 0

        estimated_capital_required = round(recommended_order_qty * unit_price, 2)

        return {
            "sku_id": sku_id,
            "product_name": name,
            "category": category,
            "velocity_class": velocity_class,
            "current_stock": int(current_stock),
            "predicted_7d_demand": round(predicted_7d_demand, 1),
            "lead_time_days": lead_time_days,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "recommended_order_qty": recommended_order_qty,
            "status": status,
            "capital_required_usd": estimated_capital_required,
            "rule_applied": rule_applied,
        }

    def generate_store_reorder_manifest(
        self,
        current_state_df: pd.DataFrame,
        predictions_7d: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Generates full store reorder recommendation table for all active SKUs.
        """
        # Calculate category medians for fallback
        cat_medians = current_state_df.groupby("category")["actual_sales"].sum().to_dict()

        manifest_rows = []
        for _, row in current_state_df.iterrows():
            sku = row["sku_id"]
            pred_7d = predictions_7d.get(sku, row.get("rolling_mean_7", 10.0) * 7.0)

            rec = self.calculate_sku_reorder(
                sku_id=sku,
                name=row["product_name"],
                category=row["category"],
                velocity_class=row["velocity_class"],
                current_stock=int(row["ending_stock"]),
                predicted_7d_demand=pred_7d,
                lead_time_days=int(row["lead_time_days"]),
                shelf_life_days=int(row["shelf_life_days"]),
                historical_std=float(row.get("rolling_std_7", 3.0)),
                unit_price=float(row["selling_price"]),
                category_medians=cat_medians,
            )
            manifest_rows.append(rec)

        res_df = pd.DataFrame(manifest_rows)
        # Sort by urgency: CRITICAL first, then REORDER, then HEALTHY, then OVERSTOCKED
        status_order = {"CRITICAL_STOCKOUT_RISK": 0, "REORDER_RECOMMENDED": 1, "HEALTHY": 2, "OVERSTOCKED": 3}
        res_df["priority"] = res_df["status"].map(status_order)
        res_df = res_df.sort_values(by=["priority", "recommended_order_qty"], ascending=[True, False]).drop(columns=["priority"])
        return res_df


if __name__ == "__main__":
    engine = InventoryReorderEngine()
    
    # Test Fast item
    fast_test = engine.calculate_sku_reorder(
        sku_id="SKU_FAST_001", name="Fresh Milk 1L", category="Dairy", velocity_class="Fast",
        current_stock=18, predicted_7d_demand=120.0, lead_time_days=2, shelf_life_days=7,
        historical_std=14.0, unit_price=2.80
    )
    print("Fast Item Test:", fast_test)

    # Test Slow item
    slow_test = engine.calculate_sku_reorder(
        sku_id="SKU_SLOW_001", name="White Truffle Oil", category="Gourmet", velocity_class="Slow",
        current_stock=4, predicted_7d_demand=2.1, lead_time_days=6, shelf_life_days=365,
        historical_std=0.8, unit_price=24.00
    )
    print("Slow Item Test:", slow_test)

    # Test New item
    new_test = engine.calculate_sku_reorder(
        sku_id="SKU_NEW_001", name="Matcha Tea 330ml", category="Beverages", velocity_class="New",
        current_stock=0, predicted_7d_demand=35.0, lead_time_days=3, shelf_life_days=90,
        historical_std=5.0, unit_price=3.40
    )
    print("New Item Test:", new_test)
