"""
Retail Dataset Generator for Supply Chain & Inventory Reorder Assistant
Simulates a multi-SKU retail store with realistic consumer demand dynamics,
promotions, price elasticity, payday cycles, stockouts, and cold-start items.
"""

from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Define the 25 Product Catalog across 4 Behavioral Clusters
CATALOG: List[Dict] = [
    # 1. Fast-Moving Items (Staples, high daily volume, high stock-out risk)
    {"sku_id": "SKU_FAST_001", "name": "Fresh Whole Milk 1L", "category": "Dairy", "velocity": "Fast", "base_demand": 75, "price": 2.80, "lead_time": 2, "shelf_life": 7},
    {"sku_id": "SKU_FAST_002", "name": "Sliced White Bread 500g", "category": "Bakery", "velocity": "Fast", "base_demand": 65, "price": 2.20, "lead_time": 2, "shelf_life": 5},
    {"sku_id": "SKU_FAST_003", "name": "Farm Fresh Eggs 12-Pack", "category": "Dairy", "velocity": "Fast", "base_demand": 55, "price": 4.50, "lead_time": 3, "shelf_life": 21},
    {"sku_id": "SKU_FAST_004", "name": "Diet Cola 2L", "category": "Beverages", "velocity": "Fast", "base_demand": 50, "price": 2.50, "lead_time": 3, "shelf_life": 180},
    {"sku_id": "SKU_FAST_005", "name": "Organic Bananas 1kg", "category": "Produce", "velocity": "Fast", "base_demand": 80, "price": 1.90, "lead_time": 2, "shelf_life": 5},
    {"sku_id": "SKU_FAST_006", "name": "Salted Potato Chips 200g", "category": "Snacks", "velocity": "Fast", "base_demand": 45, "price": 3.20, "lead_time": 4, "shelf_life": 90},
    {"sku_id": "SKU_FAST_007", "name": "Natural Mineral Water 1.5L", "category": "Beverages", "velocity": "Fast", "base_demand": 90, "price": 1.10, "lead_time": 2, "shelf_life": 365},
    {"sku_id": "SKU_FAST_008", "name": "Plain Greek Yogurt 500g", "category": "Dairy", "velocity": "Fast", "base_demand": 40, "price": 3.60, "lead_time": 3, "shelf_life": 14},

    # 2. Promo-Sensitive Medium-Movers (Moderate baseline, massive spikes on discount)
    {"sku_id": "SKU_MED_001", "name": "Premium Ground Coffee 250g", "category": "Beverages", "velocity": "Medium", "base_demand": 24, "price": 7.50, "lead_time": 4, "shelf_life": 180},
    {"sku_id": "SKU_MED_002", "name": "Crunchy Breakfast Cereal 400g", "category": "Pantry", "velocity": "Medium", "base_demand": 28, "price": 4.90, "lead_time": 4, "shelf_life": 180},
    {"sku_id": "SKU_MED_003", "name": "Extra Virgin Olive Oil 750ml", "category": "Pantry", "velocity": "Medium", "base_demand": 18, "price": 11.50, "lead_time": 5, "shelf_life": 365},
    {"sku_id": "SKU_MED_004", "name": "Concentrated Laundry Detergent 1.5L", "category": "Household", "velocity": "Medium", "base_demand": 22, "price": 13.90, "lead_time": 4, "shelf_life": 365},
    {"sku_id": "SKU_MED_005", "name": "Dark Chocolate Bar 100g", "category": "Snacks", "velocity": "Medium", "base_demand": 30, "price": 2.90, "lead_time": 3, "shelf_life": 180},
    {"sku_id": "SKU_MED_006", "name": "Italian Penne Pasta 500g", "category": "Pantry", "velocity": "Medium", "base_demand": 35, "price": 1.70, "lead_time": 4, "shelf_life": 365},
    {"sku_id": "SKU_MED_007", "name": "All-in-1 Dishwasher Pods 30ct", "category": "Household", "velocity": "Medium", "base_demand": 16, "price": 12.50, "lead_time": 5, "shelf_life": 365},
    {"sku_id": "SKU_MED_008", "name": "Soft Toilet Paper 6-Mega Rolls", "category": "Household", "velocity": "Medium", "base_demand": 32, "price": 8.90, "lead_time": 3, "shelf_life": 365},

    # 3. Slow-Moving / Intermittent Items (Sporadic demand, zero-inflated)
    {"sku_id": "SKU_SLOW_001", "name": "Organic White Truffle Oil 100ml", "category": "Gourmet", "velocity": "Slow", "base_demand": 1.2, "price": 24.00, "lead_time": 7, "shelf_life": 365},
    {"sku_id": "SKU_SLOW_002", "name": "Pure Spanish Saffron Strands 1g", "category": "Gourmet", "velocity": "Slow", "base_demand": 0.8, "price": 18.50, "lead_time": 7, "shelf_life": 365},
    {"sku_id": "SKU_SLOW_003", "name": "Traditional Dijon Mustard 200g", "category": "Gourmet", "velocity": "Slow", "base_demand": 2.5, "price": 5.20, "lead_time": 5, "shelf_life": 180},
    {"sku_id": "SKU_SLOW_004", "name": "Madagascar Bourbon Vanilla Bean 2ct", "category": "Gourmet", "velocity": "Slow", "base_demand": 0.9, "price": 14.00, "lead_time": 7, "shelf_life": 365},
    {"sku_id": "SKU_SLOW_005", "name": "Himalayan Pink Salt Grinder 250g", "category": "Gourmet", "velocity": "Slow", "base_demand": 3.0, "price": 6.80, "lead_time": 5, "shelf_life": 365},
    {"sku_id": "SKU_SLOW_006", "name": "Aged Balsamic Vinegar 250ml", "category": "Gourmet", "velocity": "Slow", "base_demand": 1.5, "price": 19.50, "lead_time": 6, "shelf_life": 365},

    # 4. Cold-Start / Newly Launched Items (Introduced in the final 30 days)
    {"sku_id": "SKU_NEW_001", "name": "Matcha Sparkling Green Tea 330ml", "category": "Beverages", "velocity": "New", "base_demand": 28, "price": 3.40, "lead_time": 3, "shelf_life": 90, "launch_day_offset": 700},
    {"sku_id": "SKU_NEW_002", "name": "Keto Almond Crunch Bar 50g", "category": "Snacks", "velocity": "New", "base_demand": 22, "price": 2.70, "lead_time": 3, "shelf_life": 120, "launch_day_offset": 700},
    {"sku_id": "SKU_NEW_003", "name": "Barista Oat Milk 1L", "category": "Dairy", "velocity": "New", "base_demand": 38, "price": 3.80, "lead_time": 2, "shelf_life": 60, "launch_day_offset": 700},
]


def generate_retail_dataset(
    days: int = 730,
    start_date: str = "2024-01-01",
    seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates a realistic multi-SKU retail dataset over the specified number of days.
    Ground-truth dynamics include:
    - Day-of-week seasonality (Friday/Saturday/Sunday surges)
    - Monthly payday effect (days 1-5 and 28-31)
    - Promotional campaign spikes and post-promo demand dip (pantry loading)
    - Price elasticity (discounts drive non-linear demand lift)
    - Inventory stocking simulation with realistic stockouts and lead-time deliveries
    - Cold-start items introduced in the final period
    """
    np.random.seed(seed)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_range = [start_dt + timedelta(days=d) for d in range(days)]

    records = []

    for item in CATALOG:
        sku = item["sku_id"]
        base_demand = item["base_demand"]
        regular_price = item["price"]
        velocity = item["velocity"]
        category = item["category"]
        lead_time = item["lead_time"]
        launch_day = item.get("launch_day_offset", 0)

        # Initialize inventory state
        # Initial stock set comfortably to avoid day 1 artificial crash
        current_stock = int(base_demand * (lead_time + 4))
        pending_deliveries: Dict[int, int] = {}  # delivery_day -> qty
        consecutive_promo_days = 0
        last_promo_day = -999

        for day_idx, current_dt in enumerate(date_range):
            # Check cold-start availability
            if day_idx < launch_day:
                continue

            # 1. Receive incoming deliveries scheduled for today
            if day_idx in pending_deliveries:
                current_stock += pending_deliveries.pop(day_idx)

            # 2. Promotional Schedule:
            # Slow items rarely go on promo; Medium items frequently promo; Fast items occasionally
            promo_chance = 0.03 if velocity == "Slow" else (0.18 if velocity == "Medium" else 0.08)
            
            # Decide promo (ensure promo blocks of 2-4 days)
            if consecutive_promo_days > 0:
                if np.random.rand() < 0.65 and consecutive_promo_days < 4:
                    is_promo = 1
                    consecutive_promo_days += 1
                else:
                    is_promo = 0
                    consecutive_promo_days = 0
            else:
                if (day_idx - last_promo_day > 6) and (np.random.rand() < promo_chance):
                    is_promo = 1
                    consecutive_promo_days = 1
                    last_promo_day = day_idx
                else:
                    is_promo = 0

            # Discount calculation
            if is_promo:
                discount_pct = np.random.choice([0.15, 0.20, 0.30, 0.40])
                selling_price = round(regular_price * (1.0 - discount_pct), 2)
            else:
                discount_pct = 0.0
                selling_price = regular_price

            # 3. Demand Multipliers
            # Day of week multiplier (0=Mon, 6=Sun)
            dow = current_dt.weekday()
            dow_multipliers = {0: 0.90, 1: 0.88, 2: 0.92, 3: 0.95, 4: 1.15, 5: 1.45, 6: 1.35}
            dow_mult = dow_multipliers.get(dow, 1.0)

            # Payday window multiplier (days 1-5 and 28-31)
            dom = current_dt.day
            is_payday = 1 if (dom <= 5 or dom >= 28) else 0
            payday_mult = 1.25 if is_payday else 1.0

            # Promotional demand multiplier & elasticity
            # Non-linear surge with diminishing returns (promo fatigue)
            if is_promo:
                elasticity = 2.8 if velocity == "Medium" else 1.8
                fatigue_factor = max(0.65, 1.0 - (consecutive_promo_days - 1) * 0.12)
                promo_mult = (1.0 + (discount_pct * elasticity)) * fatigue_factor
            else:
                # Post-promo pantry dip: sales dip for 2 days after promo ends
                days_since_promo = day_idx - last_promo_day
                if 1 <= days_since_promo <= 2 and consecutive_promo_days == 0:
                    promo_mult = 0.82
                else:
                    promo_mult = 1.0

            # Annual trend / macro seasonality (Q4 holiday surge in Nov/Dec)
            month = current_dt.month
            seasonality_mult = 1.20 if month in [11, 12] else (1.10 if month in [6, 7] else 1.0)

            # 4. Generate Latent True Customer Demand
            expected_demand = base_demand * dow_mult * payday_mult * promo_mult * seasonality_mult

            if velocity == "Slow":
                # Intermittent zero-inflated Poisson demand
                true_demand = int(np.random.poisson(lam=expected_demand))
            else:
                noise = np.random.normal(0, scale=max(1.0, base_demand * 0.12))
                true_demand = max(0, int(round(expected_demand + noise)))

            # 5. Inventory Fulfillment & Stock-Out Execution
            start_stock = current_stock
            if current_stock >= true_demand:
                actual_sales = true_demand
                unfulfilled_demand = 0
                ending_stock = current_stock - true_demand
                was_out_of_stock = 0
            else:
                actual_sales = current_stock
                unfulfilled_demand = true_demand - current_stock
                ending_stock = 0
                was_out_of_stock = 1

            current_stock = ending_stock

            # 6. Basic Store Reorder Behavior (to keep inventory cycling realistically)
            # Reorder point: lead_time * base_demand * 1.5
            reorder_point = int(base_demand * lead_time * 1.4)
            if current_stock <= reorder_point:
                # Order enough for target horizon
                order_qty = int(base_demand * (lead_time + 5))
                arrival_day = day_idx + lead_time
                pending_deliveries[arrival_day] = pending_deliveries.get(arrival_day, 0) + order_qty

            # Record daily record
            records.append({
                "date": current_dt.strftime("%Y-%m-%d"),
                "sku_id": sku,
                "product_name": item["name"],
                "category": category,
                "velocity_class": velocity,
                "lead_time_days": lead_time,
                "shelf_life_days": item["shelf_life"],
                "regular_price": regular_price,
                "selling_price": selling_price,
                "discount_percent": round(discount_pct, 4),
                "is_promo": is_promo,
                "consecutive_promo_days": consecutive_promo_days,
                "true_demand": true_demand,
                "actual_sales": actual_sales,
                "unfulfilled_demand": unfulfilled_demand,
                "starting_stock": start_stock,
                "ending_stock": ending_stock,
                "was_out_of_stock": was_out_of_stock,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["sku_id", "date"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} records across {df['sku_id'].nunique()} SKUs -> {output_path}")

    return df


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "data", "retail_store_sales.csv")
    generate_retail_dataset(days=730, output_path=out_file)
